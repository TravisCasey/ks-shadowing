#!/usr/bin/env bash
# Run from the repository root.
#
# Regenerate the examples/data/delay_timescale/ fixtures behind
# plot_delay_timescale.py. Each of the four subdirectories holds one sampling of
# a single integration, so every sampling step sees the identical physical
# trajectory, plus one SSA reference run and a ten-point PHA-DELAY ladder whose
# embedding windows (m - 1) * dt are matched in time across steps.
#
# The committed trajectories are reused when present. Only if one is missing is
# the 11500 time unit master integrated (dt = 0.02, 150 MB, seeded from row 0 of
# examples/data/trajectory.h5) and subsampled, into OUT_DIR/master.h5.
# min_duration is held fixed at ~3.2 time units (7 rows at dt = 0.46), so it
# means the same physical duration at every step.

set -uo pipefail

OUT="${OUT_DIR:-examples/data/delay_timescale}"
RPO_FILE="data/rpos_selected.npz"
SPAN_TU="${SPAN_TU:-11500}"
RES="${RES:-2048}"
N_JOBS="${N_JOBS:--1}"

STRIDES=(46 23 12 6)
declare -A MIN_DURATION=([46]=4 [23]=7 [12]=13 [6]=27)
declare -A DELAYS=(
  [46]="1 3 5 7 9 11 13 15 17 21"
  [23]="1 5 9 13 17 21 25 29 33 41"
  [12]="1 9 15 23 31 39 47 55 61 77"
  [6]="1 15 31 47 61 77 93 107 123 153"
)

failures=()
stamp() { date "+%Y-%m-%d %H:%M:%S"; }
echo "=== Start $(stamp) | span=${SPAN_TU} tu | resolution=${RES} | n_jobs=${N_JOBS} ==="

missing_trajectory=false
for stride in "${STRIDES[@]}"; do
  [ -f "${OUT}/s${stride}/trajectory.h5" ] || missing_trajectory=true
done

if [ "${missing_trajectory}" = true ] && [ ! -f "${OUT}/master.h5" ]; then
  mkdir -p "${OUT}"
  echo "[$(stamp)] integrating master trajectory (${SPAN_TU} tu at dt=0.02)"
  SPAN_TU="${SPAN_TU}" OUT="${OUT}" uv run python - <<'PY' || { echo "FATAL: master integration failed"; exit 1; }
import os
from ks_shadowing import KSTrajectory
from ks_shadowing.core import INTEGRATION_DT, ksint

span = float(os.environ["SPAN_TU"])
out = os.environ["OUT"]
steps = round(span / INTEGRATION_DT)
seed_modes = KSTrajectory.load("examples/data/trajectory.h5", 512).modes[0]
modes = ksint(seed_modes, INTEGRATION_DT, steps, save_interval=1)
KSTrajectory(modes=modes, dt=INTEGRATION_DT, resolution=512).save(f"{out}/master.h5")
print(f"  master rows={modes.shape[0]} dt={INTEGRATION_DT}")
PY
fi

for stride in "${STRIDES[@]}"; do
  if [ ! -f "${OUT}/s${stride}/trajectory.h5" ]; then
    mkdir -p "${OUT}/s${stride}"
    STRIDE="${stride}" OUT="${OUT}" uv run python - <<'PY' || { echo "FATAL: subsample failed"; exit 1; }
import os
import numpy as np
from ks_shadowing import KSTrajectory

stride = int(os.environ["STRIDE"])
out = os.environ["OUT"]
master = KSTrajectory.load(f"{out}/master.h5", 512)
sub = KSTrajectory(modes=master.modes[::stride].copy(), dt=master.dt * stride, resolution=512)
sub.save(f"{out}/s{stride}/trajectory.h5")
assert np.array_equal(
    KSTrajectory.load(f"{out}/s{stride}/trajectory.h5", 512).modes, master.modes[::stride]
)
print(f"  stride {stride}: rows={sub.num_timesteps} dt={sub.dt:.2f}")
PY
  fi
done

# Detection: cheapest stride first, SSA reference then the delay ladder.
run() {  # run <label> <output> <args...>
  local label="$1" output="$2"; shift 2
  if [ -f "${output}" ]; then echo "[$(stamp)] skip ${label} (exists)"; return 0; fi
  echo "[$(stamp)] START ${label}"
  # --no-sync: repo edits during the run must not rebuild the binary between runs.
  if uv run --no-sync ks-detect "$@" --output "${output}"; then
    echo "[$(stamp)] OK    ${label}"
  else
    echo "[$(stamp)] FAIL  ${label}"
    failures+=("${label}")
    rm -f "${output}"
  fi
}

for stride in "${STRIDES[@]}"; do
  dir="${OUT}/s${stride}"
  traj="${dir}/trajectory.h5"
  md="${MIN_DURATION[${stride}]}"
  common=(--rpo-file "${RPO_FILE}" --trajectory "${traj}" --resolution "${RES}"
          --min-duration "${md}" --threshold-quantile 0.4 --n-jobs "${N_JOBS}" --show-progress)
  run "s${stride} SSA" "${dir}/ssa_r${RES}.h5" --method ssa "${common[@]}" --chunk-size 1000
  for m in ${DELAYS[${stride}]}; do
    run "s${stride} PHA m=${m}" "${dir}/pha_r${RES}_d${m}_o0.h5" \
        --method pha "${common[@]}" --chunk-size 2500 --delay "${m}" --max-derivative-order 0
  done
done

echo "=== End $(stamp) ==="
if [ ${#failures[@]} -eq 0 ]; then
  echo "ALL RUNS OK ($(find "${OUT}" -name "*_r${RES}*.h5" | wc -l) result files)"
else
  echo "FAILURES (${#failures[@]}): ${failures[*]}"
  echo "Re-run the same command to retry only these."
fi
