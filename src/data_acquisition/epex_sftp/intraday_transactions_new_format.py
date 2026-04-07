import logging
import os
import shutil
import tempfile
import warnings
import zipfile
from datetime import timedelta
from io import BytesIO
from pathlib import Path, PurePosixPath
import time
from typing import List, Optional

import numpy as np
import pandas as pd
import paramiko
import pytz
from dotenv import load_dotenv
from sqlalchemy import Engine, create_engine

#warnings.simplefilter(action="ignore", category=FutureWarning)
import platform

EPEX_CACHE_DIR = Path(os.getenv("EPEX_CACHE_DIR", Path.home() / ".cache" / "epex"))
EPEX_CACHE_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv()

TRANSACTION_HISTORICAL_DATA_PATH_PREFIX = PurePosixPath(
    "germany", "Intraday Continuous", "EOD", "Historical", "Transactions"
)
TRANSACTION_ZIP_FILE_NAME_PREFIX = "Continuous_Trades-DE"

SFTP_HOST = os.getenv("EPEX_SFTP_HOST")
SFTP_PORT = os.getenv("EPEX_SFTP_PORT")
SFTP_USERNAME = os.getenv("EPEX_SFTP_USER")
SFTP_PASSWORD = os.getenv("EPEX_SFTP_PW")

PASSWORD = os.getenv("SQL_PASSWORD")
if PASSWORD:
    password_for_url = f":{PASSWORD}"
else:
    password_for_url = ""

THESIS_DB_NAME = os.getenv("POSTGRES_DB_NAME")
POSTGRES_USERNAME = os.getenv("POSTGRES_USER")
POSTGRES_DB_HOST = os.getenv("POSTGRES_DB_HOST")
BERLIN_TZ = pytz.timezone("Europe/Berlin")


def sftp_connect() -> Optional[paramiko.SFTPClient]:
    """Establish a connection to the SFTP server and list files."""
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        logging.debug(f"Connecting to {SFTP_HOST}:{SFTP_PORT} as {SFTP_USERNAME}...")

        ssh.connect(
            hostname=SFTP_HOST,
            port=int(SFTP_PORT),
            username=SFTP_USERNAME,
            password=SFTP_PASSWORD,
            timeout=60,
            banner_timeout=60,
            auth_timeout=60,
            allow_agent=False,
            look_for_keys=False,
        )

        ssh.get_transport().set_keepalive(30)
        sftp = ssh.open_sftp()
        logging.debug(f"Connected to {SFTP_HOST}")
        return ssh, sftp

    except paramiko.AuthenticationException as e:
        logging.error("Authentication failed, please verify your credentials.")
    except paramiko.SSHException as sshException:
        logging.error(f"SSH connection failed: {sshException}")
    except paramiko.BadHostKeyException as badHostKeyException:
        logging.error(f"Unable to verify server's host key: {badHostKeyException}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")


def download_intraday_transaction_zip_archive(
    remote_path: PurePosixPath,
    local_path: Path,
    file_name_prefix: str,
    year: int = 2018,
) -> Path:
    """Download the ZIP file from the SFTP server."""
    ssh = None
    sftp = None
    try:
        ssh, sftp = sftp_connect()

        files = sftp.listdir(remote_path.as_posix())
        zip_files = [
            file
            for file in files
            if file.startswith(file_name_prefix)
            and file.endswith(".zip")
            and str(year) in file
        ]

        if not zip_files:
            raise FileNotFoundError(f"No ZIP files found with prefix {file_name_prefix}.")

        zip_file_name = sorted(zip_files)[-1]

        remote_file_path = PurePosixPath(remote_path, zip_file_name)
        local_path.mkdir(parents=True, exist_ok=True)
        local_file_path = Path(local_path, zip_file_name)

        if local_file_path.exists() and local_file_path.stat().st_size > 0:
            logging.info(f"[{year}] Using cached ZIP: {local_file_path}")
            return local_file_path

        logging.info(f"[{year}] Downloading {remote_file_path} -> {local_file_path}")

        # Try to download the file with prefetch disabled
        last_logged_mb = -1

        def _progress(transferred: int, total: int) -> None:
            nonlocal last_logged_mb
            # Log every ~50 MiB to avoid spamming
            transferred_mb = transferred // (50 * 1024 * 1024)
            if transferred_mb != last_logged_mb:
                last_logged_mb = transferred_mb
                if total:
                    pct = (transferred / total) * 100
                    logging.info(
                        f"[{year}] Download progress: {transferred/1024/1024:.0f} MiB / {total/1024/1024:.0f} MiB ({pct:.1f}%)"
                    )
                else:
                    logging.info(f"[{year}] Download progress: {transferred/1024/1024:.0f} MiB")

        sftp.get(
            remote_file_path.as_posix(),
            local_file_path.as_posix(),
            prefetch=False,
            callback=_progress,
        )

        return local_file_path

    except Exception as e:
        logging.exception(f"[{year}] Download failed: {e}")
        raise

    finally:
        # Close up, regardless of success or failure
        try:
            if sftp is not None:
                sftp.close()
        finally:
            if ssh is not None:
                ssh.close()



def unpack_archive(path: Path) -> Path:
    """Recursively unpack a ZIP archive, including any nested ZIP files."""
    with zipfile.ZipFile(path.as_posix(), "r") as archive:
        extract_path = path.parent
        archive.extractall(extract_path)

    nested_extract_path = extract_path
    Path(nested_extract_path).mkdir(parents=True, exist_ok=True)


    for extracted_file in extract_path.glob("**/*.zip"):
        with zipfile.ZipFile(extracted_file, "r") as nested_archive:
            nested_archive.extractall(nested_extract_path)
        extracted_file.unlink()

    return extract_path


def fetch_csv_file_names(path: Path) -> List[str]:
    """Fetch all CSV filenames from the directory."""
    files = os.listdir(path)
    csv_files = [file for file in files if file.endswith(".csv")]
    return csv_files


def _iter_csv_members_from_zip(zip_file: zipfile.ZipFile):
    for info in zip_file.infolist():
        if info.is_dir():
            continue
        name = info.filename
        lower = name.lower()
        if lower.endswith(".csv"):
            yield ("csv", name)
        elif lower.endswith(".zip"):
            yield ("zip", name)


def count_csv_files_in_zip_path(zip_path: Path) -> int:
    """Count CSV files inside a ZIP, including inside nested ZIPs."""
    total = 0
    with zipfile.ZipFile(zip_path.as_posix(), "r") as outer:
        for kind, member_name in _iter_csv_members_from_zip(outer):
            if kind == "csv":
                total += 1
                continue

            with outer.open(member_name, "r") as nested_fh:
                nested_bytes = nested_fh.read()
            with zipfile.ZipFile(BytesIO(nested_bytes), "r") as nested_zip:
                for nested_kind, _nested_member_name in _iter_csv_members_from_zip(
                    nested_zip
                ):
                    if nested_kind == "csv":
                        total += 1
    return total


def iter_csv_file_handles_from_zip_path(zip_path: Path):
    """
    Yield (display_name, file_handle) pairs for all CSV files found inside the given ZIP,
    including CSVs inside nested ZIPs, without extracting to disk.
    """
    with zipfile.ZipFile(zip_path.as_posix(), "r") as outer:
        for kind, member_name in _iter_csv_members_from_zip(outer):
            if kind == "csv":
                with outer.open(member_name, "r") as fh:
                    yield member_name, fh
            else:
                # Nested zip: read bytes once and iterate its CSVs
                with outer.open(member_name, "r") as nested_fh:
                    nested_bytes = nested_fh.read()
                with zipfile.ZipFile(BytesIO(nested_bytes), "r") as nested_zip:
                    for nested_kind, nested_member_name in _iter_csv_members_from_zip(
                        nested_zip
                    ):
                        if nested_kind != "csv":
                            continue
                        with nested_zip.open(nested_member_name, "r") as fh:
                            yield f"{member_name}::{nested_member_name}", fh


def extract_data_from_csv_file(path: Path, filename: str) -> pd.DataFrame:
    """Extract data from a specific CSV file."""
    full_path = Path(path, filename)
    df = pd.read_csv(
        full_path,
        skiprows=1,
    )
    # Optionally, you can convert columns to specific datetime formats if needed

    df["DeliveryStart"] = pd.DatetimeIndex(
        pd.to_datetime(df["DeliveryStart"], utc=True)
    ).tz_convert("Europe/Berlin")
    df["DeliveryEnd"] = pd.DatetimeIndex(
        pd.to_datetime(df["DeliveryEnd"], utc=True)
    ).tz_convert("Europe/Berlin")
    df["ExecutionTime"] = pd.DatetimeIndex(
        pd.to_datetime(df["ExecutionTime"], utc=True)
    ).tz_convert("Europe/Berlin")

    return df


def extract_data_from_csv_handle(file_handle) -> pd.DataFrame:
    """Extract data from a CSV file-like object (e.g., streamed from a ZIP)."""
    df = pd.read_csv(
        file_handle,
        skiprows=1,
    )
    df["DeliveryStart"] = pd.DatetimeIndex(
        pd.to_datetime(df["DeliveryStart"], utc=True)
    ).tz_convert("Europe/Berlin")
    df["DeliveryEnd"] = pd.DatetimeIndex(
        pd.to_datetime(df["DeliveryEnd"], utc=True)
    ).tz_convert("Europe/Berlin")
    df["ExecutionTime"] = pd.DatetimeIndex(
        pd.to_datetime(df["ExecutionTime"], utc=True)
    ).tz_convert("Europe/Berlin")
    return df


def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    df.rename(
        {
            "TradeId": "tradeid",
            "Side": "side",
            "Product": "product",
            "Volume": "volume",
            "Price": "price",
            "ExecutionTime": "executiontime",
            "DeliveryStart": "deliverystart",
            "DeliveryEnd": "deliveryend",
            "Currency": "currency",
            "VolumeUnit": "volumeunit",
        },
        axis=1,
        inplace=True,
    )

    df.drop(
        [
            "RemoteTradeId",
            "TradePhase",
            "UserDefinedBlock",
            "SelfTrade",
            "OrderID",
            "DeliveryArea",
        ],
        axis=1,
        inplace=True,
    )

    # Filter the DataFrame based on the conditions
    filtered_df = df[
        (df["product"].isin(["XBID_Quarter_Hour_Power", "Intraday_Quarter_Hour_Power"]))
    ].copy()

    # set seconds of execution time to zero, because we definetly do not need them
    filtered_df["executiontime"] = filtered_df["executiontime"].apply(
        lambda x: x.replace(second=0, microsecond=0)
    )

    # Calculate the weighted average price and group by deliverystart
    grouped_df = (
        filtered_df.groupby(["deliverystart", "executiontime"])
        .apply(
            lambda x: pd.Series(
                {
                    "weighted_avg_price": (x["price"] * x["volume"]).sum()
                    / x["volume"].sum(),
                    "volume": x["volume"].sum(),
                    "trade_count": x.shape[0],
                    "tradeid": x["tradeid"].iloc[0],
                    "side": x["side"].iloc[0],
                    "deliveryend": x["deliveryend"].iloc[0],
                    "product": x["product"].iloc[0],
                    "volumeunit": x["volumeunit"].iloc[0],
                    "currency": x["currency"].iloc[0],
                }
            )
        )
        .reset_index()
    )

    return grouped_df


def load_data(df: pd.DataFrame, database: Engine) -> None:
    conn = database.connect()
    df.to_sql(
        "transactions_intraday_de",
        conn,
        chunksize=10000,
        if_exists="append",
        index=False,
    )
    conn.close()


def execute_etl_transactions_new_format(years: List[int]) -> None:
    ## be aware: 2022 there was a change in data format
    # -> file for 2022 incomplete (new files "Continuous_Trades-MA-yyyymmdd-yyyymmddThhmmsssssZ")
    database = create_engine(
        f"postgresql://{POSTGRES_USERNAME}{password_for_url}@{POSTGRES_DB_HOST}/{THESIS_DB_NAME}"
    )
    for year in years:
        transaction_archive_location = download_intraday_transaction_zip_archive(
            remote_path=TRANSACTION_HISTORICAL_DATA_PATH_PREFIX,
            local_path=EPEX_CACHE_DIR,
            file_name_prefix=TRANSACTION_ZIP_FILE_NAME_PREFIX,
            year=year,
        )
        logging.info(f"[{year}] ZIP available at: {transaction_archive_location}")

        total_csv = None
        try:
            total_csv = count_csv_files_in_zip_path(transaction_archive_location)
            logging.info(f"[{year}] ZIP contains ~{total_csv} CSV files (incl. nested ZIPs)")
        except Exception as e:
            logging.warning(f"[{year}] Could not count CSV files in ZIP: {e}")

        t0 = time.time()
        processed = 0
        for display_name, csv_fh in iter_csv_file_handles_from_zip_path(
            transaction_archive_location
        ):
            processed += 1
            if processed == 1 or processed % 10 == 0:
                elapsed = max(time.time() - t0, 1e-6)
                rate = processed / elapsed
                if total_csv:
                    pct = (processed / total_csv) * 100
                    logging.info(
                        f"[{year}] Processing CSV {processed}/{total_csv} ({pct:.1f}%) at {rate:.2f} files/s. Latest: {display_name}"
                    )
                else:
                    logging.info(
                        f"[{year}] Processing CSV #{processed} at {rate:.2f} files/s. Latest: {display_name}"
                    )
            df = extract_data_from_csv_handle(csv_fh)
            df_for_db = transform_data(df)
            load_data(df_for_db, database)
        logging.info(f"[{year}] Done. Processed {processed} CSV files from ZIP.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    years = [2024, 2025]
    execute_etl_transactions_new_format(years)
