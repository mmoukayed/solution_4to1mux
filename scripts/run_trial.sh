#!/usr/bin/env bash
# Run inside the ERC container after sourcing install/setup.bash.
set -euo pipefail

if ! command -v ros2 >/dev/null 2>&1; then
    echo 'Source /opt/erc_ws/install/setup.bash inside the ERC container first.' >&2
    exit 1
fi

package_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
trial_log_dir="$package_root/logs"
mkdir -p "$trial_log_dir/ros"
export ROS_LOG_DIR="$trial_log_dir/ros"
trial_log="$trial_log_dir/trial-$(date +%Y%m%d-%H%M%S)-$$.log"
printf 'Trial log: %s\n' "$trial_log"
set +e
ros2 launch solution_4to1mux solution.launch.py "$@" 2>&1 | tee "$trial_log"
trial_status=("${PIPESTATUS[@]}")
set -e
printf '\nSaved trial output: %s\n' "$trial_log"
if (( trial_status[1] != 0 )); then
    echo 'Could not write the complete trial log.' >&2
    exit "${trial_status[1]}"
fi
exit "${trial_status[0]}"
