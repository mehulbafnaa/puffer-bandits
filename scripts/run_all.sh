#!/usr/bin/env bash
set -euo pipefail

#!/usr/bin/env bash
set -euo pipefail

# Run a comprehensive sweep of agents/environments using uv entrypoints.
#
# Usage:
#   scripts/run_all.sh [device]
#     device: optional single device override (cpu|mps|cuda). If omitted, uses $DEVICES or defaults to mps.
#
# Env knobs:
#   DEVICES="cpu mps cuda"   # device matrix
#   SEEDS="0 1 2"            # seeds to iterate
#   DO_SMOKE=1/0             # small, quick runs (default 1)
#   DO_THROUGHPUT=1/0        # larger MP runs (default 1)
#   DO_NEURAL=1/0            # neural agent runs (default 1)
#   LOGROOT=sweeps_logs       # root log directory
#   WORKERS=8                 # num-workers for MP
#   K=10 D=8                  # default problem sizes
#   (per-category T/RUNS can be overridden, see below)

if [[ $# -ge 1 ]]; then
  DEVICES="$1"
fi
DEVICES="${DEVICES:-mps}"

# Tunables (can be overridden via env)
K="${K:-10}"
D="${D:-8}"

# Ensure num-workers divides runs for PufferLib
WORKERS="${WORKERS:-8}"

SEEDS="${SEEDS:-0}"

# Category toggles
DO_SMOKE="${DO_SMOKE:-1}"
DO_THROUGHPUT="${DO_THROUGHPUT:-1}"
DO_NEURAL="${DO_NEURAL:-1}"

# Non-contextual (stationary / non-stationary)
RUNS_BASIC="${RUNS_BASIC:-1024}"
T_BASIC="${T_BASIC:-2000}"

# Contextual (linear)
RUNS_CTX="${RUNS_CTX:-512}"
T_CTX="${T_CTX:-600}"

# Neural agents (heavier)
RUNS_NEURAL="${RUNS_NEURAL:-256}"
T_NEURAL="${T_NEURAL:-400}"

# Hyperparameter defaults (override via env)
ALPHAS=${ALPHAS:-"1.0"}
VS=${VS:-"0.1"}
GAMMAS_EXP3=${GAMMAS_EXP3:-"0.07"}
GAMMAS_EXP3IX=${GAMMAS_EXP3IX:-"0.05"}
DUCB_C=${DUCB_C:-"2.0"}
DUCB_DISCOUNT=${DUCB_DISCOUNT:-"0.99"}
SW_C=${SW_C:-"2.0"}
SW_WINDOW=${SW_WINDOW:-"200"}
NEURAL_FEATURES=${NEURAL_FEATURES:-"32"}
NEURAL_ENSEMBLES=${NEURAL_ENSEMBLES:-"3"}
NEURAL_HIDDEN=${NEURAL_HIDDEN:-"128"}
NEURAL_DEPTH=${NEURAL_DEPTH:-"2"}
NEURAL_DROPOUT=${NEURAL_DROPOUT:-"0.1"}

LOGROOT="${LOGROOT:-sweeps_logs}"
STAMP="$(date +%Y%m%d_%H%M%S)"

check_tools() {
  command -v uv >/dev/null 2>&1 || { echo "[err] uv not found. Install via pipx install uv" >&2; exit 127; }
}

ensure_divisible() {
  local runs="$1" workers="$2"
  if (( workers <= 0 )); then echo "[err] --num-workers must be >=1" >&2; exit 2; fi
  if (( runs % workers != 0 )); then
    echo "[warn] runs ($runs) not divisible by num-workers ($workers); adjusting workers to 1" >&2
    WORKERS=1
  fi
}

run() {
  local name="$1"; shift
  local logfile="$1"; shift
  echo "\n[sweep] RUN ${name}: $*" | tee -a "$logfile"
  uv run "$@" 2>&1 | tee -a "$logfile"
}

check_tools

for DEV in $DEVICES; do
  for SEED in $SEEDS; do
    LOGDIR="${LOGROOT}/${STAMP}_${DEV}_s${SEED}"
    mkdir -p "$LOGDIR"
    echo "[sweep] logs -> ${LOGDIR}"
    ensure_divisible "$RUNS_BASIC" "$WORKERS"
    ensure_divisible "$RUNS_CTX" "$WORKERS"
    ensure_divisible "$RUNS_NEURAL" "$WORKERS"

    # --- Smoke (quick) ---
    if [[ "${DO_SMOKE}" == "1" ]]; then
      run "smoke_klucb" "$LOGDIR/smoke_klucb.log" \
        puffer-bandits-native --env bernoulli --algo klucb --k "$K" --T 300 --runs 256 \
        --vector serial --device "$DEV" --tui --log-every 50 --seed "$SEED"

      run "smoke_linucb" "$LOGDIR/smoke_linucb.log" \
        puffer-bandits-native --env contextual --algo linucb --k "$K" --d "$D" --T 300 --runs 128 \
        --vector serial --device "$DEV" --tui --log-every 50 --seed "$SEED" --alpha $(echo $ALPHAS | awk '{print $1}') --lam 1.0
    fi

    # --- Throughput (MP) ---
    if [[ "${DO_THROUGHPUT}" == "1" ]]; then
      run "tp_klucb" "$LOGDIR/tp_klucb.log" \
        puffer-bandits-native --env bernoulli --algo klucb --k "$K" --T "$T_BASIC" --runs "$RUNS_BASIC" \
        --vector mp --num-workers "$WORKERS" --device "$DEV" --tui --log-every 200 --seed "$SEED"

      for V in $VS; do
        run "tp_lints_v${V}" "$LOGDIR/tp_lints_v${V}.log" \
          puffer-bandits-native --env contextual --algo lints --k "$K" --d "$D" --T "$T_CTX" --runs "$RUNS_CTX" \
          --vector mp --num-workers "$WORKERS" --device "$DEV" --tui --log-every 100 --seed "$SEED" --v "$V" --lam 1.0
      done
    fi

    # --- Neural (heavier) ---
    if [[ "${DO_NEURAL}" == "1" ]]; then
      run "neuralts" "$LOGDIR/neuralts.log" \
        puffer-bandits-native --env contextual --algo neuralts --k "$K" --d "$D" --T "$T_NEURAL" --runs "$RUNS_NEURAL" \
        --vector mp --num-workers "$WORKERS" --device "$DEV" --tui --log-every 100 --seed "$SEED" \
        --ensembles $(echo $NEURAL_ENSEMBLES | awk '{print $1}') --hidden "$NEURAL_HIDDEN" --depth "$NEURAL_DEPTH" --dropout "$NEURAL_DROPOUT"

      run "neurallinear_f${NEURAL_FEATURES}" "$LOGDIR/neurallinear_f${NEURAL_FEATURES}.log" \
        puffer-bandits-native --env contextual --algo neurallinear --k "$K" --d "$D" --T "$T_NEURAL" --runs "$RUNS_NEURAL" \
        --vector mp --num-workers "$WORKERS" --device "$DEV" --tui --log-every 100 --seed "$SEED" \
        --features "$NEURAL_FEATURES" --linlam 1.0 --linv 0.1 --hidden "$NEURAL_HIDDEN" --depth "$NEURAL_DEPTH" --dropout "$NEURAL_DROPOUT"
    fi

    # --- Adversarial (plots + CSV, advanced runner) ---
    for G in $GAMMAS_EXP3; do
      run "exp3_g${G}" "$LOGDIR/exp3_g${G}.log" \
        puffer-bandits-advanced --env bernoulli --algo exp3 --k "$K" --T 1000 \
        --runs "$RUNS_BASIC" --device "$DEV" --num-workers "$WORKERS" --log-every 200 --save-csv --seed "$SEED" --gamma "$G"
    done
    for G in $GAMMAS_EXP3IX; do
      run "exp3ix_g${G}" "$LOGDIR/exp3ix_g${G}.log" \
        puffer-bandits-advanced --env bernoulli --algo exp3ix --k "$K" --T 1000 \
        --runs "$RUNS_BASIC" --device "$DEV" --num-workers "$WORKERS" --log-every 200 --save-csv --seed "$SEED" --gamma "$G"
    done

    # --- Non-contextual with plots ---
    run "klucb_plots" "$LOGDIR/klucb_plots.log" \
      puffer-bandits-runner --algo klucb --k "$K" --T "$T_BASIC" --runs "$RUNS_BASIC" --device "$DEV" --num-workers "$WORKERS" --log-every 200 --save-csv --seed "$SEED"

    for C in $DUCB_C; do for DIS in $DUCB_DISCOUNT; do
      run "ducb_c${C}_d${DIS}" "$LOGDIR/ducb_c${C}_d${DIS}.log" \
        puffer-bandits-runner --algo ducb --k "$K" --T 1500 --runs "$RUNS_BASIC" --device "$DEV" --num-workers "$WORKERS" --log-every 300 --save-csv --seed "$SEED" \
        --nonstationary --sigma 0.1 --c "$C" --discount "$DIS"
    done; done

    for C in $SW_C; do for W in $SW_WINDOW; do
      run "swucb_c${C}_w${W}" "$LOGDIR/swucb_c${C}_w${W}.log" \
        puffer-bandits-runner --algo swucb --k "$K" --T 1500 --runs "$RUNS_BASIC" --device "$DEV" --num-workers "$WORKERS" --log-every 300 --save-csv --seed "$SEED" \
        --nonstationary --sigma 0.1 --c "$C" --window "$W"
    done; done

  done
done

echo "\n[sweep] completed. See logs in ${LOGROOT}/${STAMP}_*"
