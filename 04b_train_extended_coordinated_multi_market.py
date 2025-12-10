import os
import warnings

import pandas as pd
import torch
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    CallbackList,
    BaseCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env

from src.coordinated_multi_market.basic_battery_dam_env import BasicBatteryDAM
from src.coordinated_multi_market.custom_ppo import CustomPPO
from src.coordinated_multi_market.learning_utils import (
    load_input_data,
    prepare_input_data,
    linear_schedule,
)
from src.shared.folder_versioning import create_new_dir_version
from src.shared.config import (
    COORDINATED_MODEL_NAME_QH,
    LOGGING_PATH_COORDINATED,
    MODEL_OUTPUT_PATH_COORDINATED,
    RTE,
    SCALER_OUTPUT_PATH_COORDINATED,
    SEED,
    TENSORBOARD_PATH_INTELLIGENT,
)

warnings.simplefilter(action="ignore", category=FutureWarning)



# ============================================================
# Konfiguration für das erweiterte Training (Phase 2)
# ============================================================

# Von welcher (DA-)Modellversion / welchem Checkpoint sollen die Gewichte geladen werden?
SOURCE_MODEL_NUMBER = "8"  # Ordnernummer deiner DA-Trainingsversion
SOURCE_MODEL_CHECKPOINT = "ppo_stacked_checkpoint_2000000_steps"  # Name ohne ".zip"

# Wie lange soll das erweiterte Training laufen?
EXTENDED_TRAINING_STEPS = 1_000_000  # z.B. genau die 500k Steps für λ: 1.0 -> 0.5

# Neue Hyperparameter für das weiterführende Training
NEW_LEARNING_RATE = 1e-4         # Beispielwert – bitte ggf. anpassen
NEW_ENT_COEF = 0.05             # Beispielwert – bitte ggf. anpassen

# λ-Schedule:
# Start: λ=1.0, alle 100k Steps -0.1, bis min. 0.5
LAMBDA_START = 0.9
LAMBDA_END = 0.5
LAMBDA_DECAY_STEP = 100_000
LAMBDA_DECAY = 0.1  # pro 100k Steps


# ============================================================
# Callback für λ-Schedule
# ============================================================

class LambdaScheduleCallback(BaseCallback):
    """
    Steuert self.model.lambda_val während des Trainings:

    - Start bei start_lambda
    - Alle decay_step Timesteps wird lambda um decay abgesenkt
    - Untergrenze bei end_lambda
    - Die Zählung beginnt bei dem num_timesteps-Wert, den das geladene Modell hat
      (also ab Start des erweiterten Trainings).
    """

    def __init__(
        self,
        start_lambda: float = 1.0,
        end_lambda: float = 0.5,
        decay_step: int = 100_000,
        decay: float = 0.1,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.start_lambda = start_lambda
        self.end_lambda = end_lambda
        self.decay_step = decay_step
        self.decay = decay
        self.start_timesteps: int | None = None

    def _on_training_start(self) -> None:
        # Merke dir, bei welchem num_timesteps das erweiterte Training beginnt
        self.start_timesteps = int(self.model.num_timesteps)
        # Explizit mit start_lambda starten
        if hasattr(self.model, "lambda_val"):
            self.model.lambda_val = float(self.start_lambda)
        if self.verbose > 0:
            print(
                f"[LambdaSchedule] Start at num_timesteps={self.start_timesteps}, "
                f"lambda={self.start_lambda}"
            )

    def _on_step(self) -> bool:
        if self.start_timesteps is None:
            # Falls aus irgendeinem Grund _on_training_start nicht gelaufen ist
            self.start_timesteps = int(self.model.num_timesteps)

        # Wie viele Steps seit Beginn von Phase 2?
        steps_since_start = int(self.model.num_timesteps - self.start_timesteps)
        if steps_since_start < 0:
            steps_since_start = 0

        # Wieviele "Decay-Stufen" sind bereits vergangen?
        n_decays = steps_since_start // self.decay_step

        new_lambda = max(
            self.end_lambda,
            self.start_lambda - self.decay * n_decays
        )

        if hasattr(self.model, "lambda_val"):
            self.model.lambda_val = float(new_lambda)

        # Für TensorBoard
        self.model.logger.record("curriculum/lambda", float(new_lambda))

        if self.verbose > 1:
            print(
                f"[LambdaSchedule] num_timesteps={self.model.num_timesteps}, "
                f"steps_since_start={steps_since_start}, lambda={new_lambda}"
            )

        return True


# ============================================================
# Hauptteil
# ============================================================

if __name__ == "__main__":

    # Ausgabeordner für diese neue Trainingsphase (NEUE Versionsnummer)
    os.makedirs(LOGGING_PATH_COORDINATED, exist_ok=True)
    os.makedirs(MODEL_OUTPUT_PATH_COORDINATED, exist_ok=True)
    os.makedirs(SCALER_OUTPUT_PATH_COORDINATED, exist_ok=True)

    versioned_log_path = create_new_dir_version(LOGGING_PATH_COORDINATED)
    versioned_model_path = create_new_dir_version(MODEL_OUTPUT_PATH_COORDINATED)
    versioned_scaler_path = create_new_dir_version(SCALER_OUTPUT_PATH_COORDINATED)

    print(f"[Extended Training] Logging path: {versioned_log_path}")
    print(f"[Extended Training] Model output path: {versioned_model_path}")
    print(f"[Extended Training] Scaler path: {versioned_scaler_path}")

    # Reward components logging path – wie im alten Script, nur im neuen versioned_log_path
    reward_log_path = os.path.join(versioned_log_path, "reward_components.csv")
    print(f"[Extended Training] Reward components CSV: {reward_log_path}")


    # Device wählen
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Daten laden und vorbereiten
    df_spot_train, df_spot_val, df_spot_test = load_input_data(write_test=False)
    input_data_train = prepare_input_data(
        df_spot_train,
        versioned_scaler_path,
        fit_scaler=True,
    )

    # Reward Logging (DA+IDC-Komponenten)
    reward_log_path = os.path.join(versioned_log_path, "reward_components_extended.csv")
    print(f"[Extended Training] Reward components CSV: {reward_log_path}")

    # Environment initialisieren
    env = BasicBatteryDAM(
        modus="train_extended",
        logging_path=versioned_log_path,
        input_data=input_data_train,
        round_trip_efficiency=RTE,
    )

    check_env(env)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])

    # Checkpoint-Callback: speichert Zwischenschritte
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=versioned_model_path,
        name_prefix="ppo_stacked_extended_checkpoint",
    )

    # λ-Schedule-Callback
    lambda_callback = LambdaScheduleCallback(
        start_lambda=LAMBDA_START,
        end_lambda=LAMBDA_END,
        decay_step=LAMBDA_DECAY_STEP,
        decay=LAMBDA_DECAY,
        verbose=1,
    )

    # Beide Callbacks kombinieren
    callback = CallbackList([checkpoint_callback, lambda_callback])

    # Policy-Architektur (wie zuvor)
    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=dict(pi=[64, 64, 64, 64], vf=[64, 64, 64, 64]),
        log_std_init=-0.5,
    )

    # Pfad zum bestehenden (DA-trainierten) Modell
    source_model_path = os.path.join(
        MODEL_OUTPUT_PATH_COORDINATED,
        SOURCE_MODEL_NUMBER,
        SOURCE_MODEL_CHECKPOINT + ".zip",
    )

    print(f"[Extended Training] Loading pretrained DA model from: {source_model_path}")
    

    # Modell laden und Hyperparameter überschreiben
    model = CustomPPO.load(
        source_model_path,
        env=env,
        device=device,
        custom_objects={
            # Learning-Rate-Config überschreiben
            "learning_rate": NEW_LEARNING_RATE,
            "lr_schedule": linear_schedule(NEW_LEARNING_RATE),
            # Entropie-Koeffizient überschreiben
            "ent_coef": NEW_ENT_COEF,
            # Falls im Original andere Werte gesetzt waren:
            "seed": SEED,
        },
    )

    model.reward_log_path = reward_log_path

    from stable_baselines3.common.logger import configure
    # Neuer TensorBoard-Ordner nur für das erweiterte Training
    TB_RUN_NAME = "PPO_extended_01"
    tb_log_path = os.path.join(TENSORBOARD_PATH_INTELLIGENT, TB_RUN_NAME)
    os.makedirs(tb_log_path, exist_ok=True)
    new_logger = configure(tb_log_path, ["stdout", "tensorboard"])
    model.set_logger(new_logger)


    # Sicherheitshalber Optimizer-LR direkt setzen
    if hasattr(model, "policy") and hasattr(model.policy, "optimizer"):
        for param_group in model.policy.optimizer.param_groups:
            param_group["lr"] = NEW_LEARNING_RATE

    # Zu Beginn der Phase 2 explizit λ=1.0 setzen;
    # LambdaScheduleCallback übernimmt dann im Training das weitere Update.
    if hasattr(model, "lambda_val"):
        model.lambda_val = float(LAMBDA_START)

    # Wichtig: num_timesteps NICHT zurücksetzen, damit der λ-Schedule
    # im Callback relativ zum bisherigen Training gezählt werden kann.
    reset_num_timesteps = False

    print(
        f"[Extended Training] Starting from num_timesteps={model.num_timesteps}, "
        f"lambda={getattr(model, 'lambda_val', None)}"
    )

    # Lernen
    model.learn(
        total_timesteps=EXTENDED_TRAINING_STEPS,
        callback=callback,
        reset_num_timesteps=reset_num_timesteps,
    )

    # Finales Modell speichern
    model.save(os.path.join(versioned_model_path, COORDINATED_MODEL_NAME_QH))

    print("[Extended Training] Finished training!")
