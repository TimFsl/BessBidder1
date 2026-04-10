# v2_counterfactual_reward

## Action space (3 actions)

| Index | Meaning | Normalized power |
|------|---------|------------------|
| `0` | **Idle** | `0` |
| `1` | **Full buy** (charge) | `-1` |
| `2` | **Full sell** (discharge) | `+1` |

**Checkpoints from v1 / 7-action policies are incompatible** (policy output size changes).

## Counterfactual RI (Option A)

- Total shaped return per episode:  
  `R = sum(DA step rewards) + RI_profit_eur / 10`  
  (same scaling as v1: total RI-shaped reward per episode = `RI_eur / 10`).
- **Default** `counterfactual_active_mode="first_buy_first_sell"`: at most **2** counterfactual replays per day — first hour where action `== 1` (buy) and first hour where action `== 2` (sell). Each gets margin `R_full − R_cf` with idle substituted at that hour; other hours keep DA-only step rewards.
- Alternative: `counterfactual_active_mode="volume_nonzero"` — one CF per hour with `|realized volume| > threshold` (optionally capped by `max_counterfactual_steps_per_episode`).

Shared RI code: `src/coordinated_multi_market/rolling_intrinsic/` (not duplicated).

## Run

From repository root:

```bash
PYTHONPATH=. python -m src.coordinated_multi_market.v2_counterfactual_reward.train
```

## TensorBoard (reward diagnostics)

These are logged from `CustomPPO.collect_rollouts` (per rollout, averaged over episodes in that rollout):

| Scalar | Meaning |
|--------|--------|
| `episode_profit/combined_eur` | Mean DA + RI **cash** profit (€) per episode. |
| `episode_profit/day_ahead_eur` | Mean DA-only profit (€). |
| `episode_profit/intraday_eur` | Mean RI simulator profit (€). |
| `episode_reward/combined_sum` | Mean **total shaped reward** sum per episode (DA + intrinsic/CF). |
| `episode_reward/day_ahead_sum` | Mean sum of **env DA step rewards** per episode. |
| `episode_reward/intrinsic_step_sum_mean` | Mean sum of **intrinsic / CF** per-step terms (uniform RI or margins). **Not** PPO’s GAE advantage. |
| `episode_reward/advantage_sum` | Same as `intrinsic_step_sum_mean` (legacy name; matches CSV column `advantage_reward`). |
| `episode_reward/cf_mean_margin` | Mean of per-episode mean margin (alias for below). |
| `counterfactual/mean_margin_within_episode` | Mean of \(\text{mean}(margin_t)\) inside each episode (strict CF). |
| `counterfactual/sum_margins_per_episode_mean` | Mean of \(\sum_t margin_t\) per episode (≈ total intrinsic sum in strict CF). |
| `counterfactual/active_hours_per_episode_mean` | Mean number of timesteps that received a nonzero margin. |
| `counterfactual/episodes_with_strict_cf` | Episodes that ran strict CF (not uniform fallback) in this rollout. |
| `counterfactual/episodes_uniform_ri_fallback` | Episodes with no CF hours → uniform RI spread like v1. |

**PPO dynamics** (policy entropy, value loss, clip fraction, etc.) are under the usual SB3 keys, e.g. `train/entropy_loss`, `train/approx_kl`, `train/clip_fraction`, `train/explained_variance`.

## Entropy collapse & two-phase training

- Low entropy means the policy is **almost deterministic**. New reward terms still **change the returns and advantages**, so learning can continue, but **exploration** is weak — raising `ent_coef` or adding noise / second-phase **fine-tuning** can help.
- **Option A — new run from checkpoint:**  
  `model = CustomPPO.load("...zip"); model.set_env(env); model.learn(...)` with new hyperparameters (`learning_rate`, `ent_coef`, `clip_range`, …). You can also **re-seed** exploration via higher `ent_coef` or a schedule.
- **Option B — change hyperparameters after X steps (like curriculum):**  
  Possible, but **not built into vanilla SB3** for all params. You typically use a **callback** (`_on_step`) that every N steps updates `self.model.learning_rate`, `self.model.ent_coef`, or swaps reward mode — or use **schedules** (`schedule(progress_remaining)`) for LR. For a **full policy re-init**, load a checkpoint or manually reset policy weights (unusual).

## `CustomPPO` knobs

- `use_counterfactual_ri_reward=True` — `False` → uniform RI spread per step (v1-style).
- `counterfactual_idle_action=0` — must match env idle index (`0` for `Discrete(3)`).
- `counterfactual_active_mode="first_buy_first_sell"` | `"volume_nonzero"`.
- `counterfactual_min_abs_volume` — used only in `volume_nonzero` mode.
- `max_counterfactual_steps_per_episode` — used only in `volume_nonzero` mode (cap + sort by \|volume\|).
