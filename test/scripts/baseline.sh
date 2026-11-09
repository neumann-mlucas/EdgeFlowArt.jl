#!/usr/bin/env bash
# Baseline sweep: fixture corpus × canonical configs.
# Skips outputs that already exist.
#
# Usage:  ./test/scripts/baseline.sh                # full corpus
#         ./test/scripts/baseline.sh moon.jpg       # subset (basename or path)
#
# Env: PHASE (default baseline), JULIA_NUM_THREADS (default 8), JULIA_BIN.

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# shellcheck source-path=SCRIPTDIR source=lib.sh disable=SC1091
source "${HERE}/lib.sh"

PHASE="${PHASE:-baseline}"
set_out_dir "$PHASE"
JULIA_OPTS=(-O3 -t "${JULIA_NUM_THREADS:-8}")
# shellcheck disable=SC2034  # consumed by run_sweep in lib.sh
CSV_HEADER="phase,image,config,size,steps,seconds,status,git_sha"

# name|size|steps
# shellcheck disable=SC2034  # consumed by run_sweep in lib.sh
CONFIGS=(
  "fast|256|500"
  "full|512|2000"
)

run_config() {
  local cfg="$1" input="$2" out_stem="$3"
  IFS='|' read -r name size steps <<<"$cfg"

  local stem_short
  stem_short="$(basename "$out_stem")"
  if [[ -f "${out_stem}.png" ]]; then
    echo "  SKIP  ${stem_short}.png"
    return
  fi

  julia_timed "${out_stem}.log" "${JULIA_OPTS[@]}" --project="$PROJECT_DIR" main.jl \
    -i "$input" -o "${out_stem}.png" -s "$size" --steps "$steps"

  local img
  img="$(basename "$input" .png)"
  echo "${PHASE},${img},${name},${size},${steps},${LAST_SECS},${LAST_STATUS},$(git_sha)" >>"$BENCH_CSV"
  echo "  ${LAST_STATUS^^}  ${stem_short}.png  ($(fmt_time "$LAST_SECS"))"
}

run_sweep "$@"
