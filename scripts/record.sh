#!/usr/bin/env bash
# Record a run for later analysis. Start it in a second terminal (same container) BEFORE launching,
# stop it with Ctrl-C after the run:
#
#   scripts/record.sh slot_try1            # -> bags/<date>_<time>_slot_try1/
#   scripts/record.sh slot_try1 --raw      # also the raw colour images (~80 MB/s: short runs only)
#   scripts/record.sh slot_try1 --light    # no depth images (if the link to the robot struggles)
#
# Records: the solution's own state/failure/poses and perception debug topics, /erc (the organisers'
# checker incl. rejections), joints, odom, both laser scans, cmd_vel, TF, the URDF, camera infos,
# compressed colour and the raw depth the stack uses. Enough to replay perception offline and to see
# why a run did what it did.
set -euo pipefail
name=${1:-run}; shift || true
raw=0; light=0
for a in "$@"; do
  case "$a" in --raw) raw=1;; --light) light=1;; *) echo "unknown option $a" >&2; exit 2;; esac
done
if ! command -v ros2 >/dev/null 2>&1; then
  echo 'Source /opt/ros/humble/setup.bash (inside the ERC container) first.' >&2; exit 1
fi
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
out="$root/bags/$(date +%Y%m%d_%H%M%S)_$name"
mkdir -p "$root/bags"

regex='^/(solution|perception|manipulation|erc)/.*'
regex+='|^/tf$|^/tf_static$|^/joint_states$|^/robot_description$|^/mobile_base_controller/odom$'
regex+='|^/scan.*|^/cmd_vel.*|^/head_front_camera/.*camera_info$'
regex+='|^/head_front_camera/color/image_raw/compressed$'
[ "$light" = 1 ] || regex+='|^/head_front_camera/depth/image_rect_raw$|^/head_front_camera/aligned_depth_to_color/image_raw$'
[ "$raw" = 1 ] && regex+='|^/head_front_camera/color/image_raw$'

storage=(-s sqlite3)
if ros2 bag record --help 2>/dev/null | grep -q mcap; then
  storage=(-s mcap --storage-preset-profile zstd_fast)
fi
echo "recording to $out (Ctrl-C to stop)"
ros2 bag record "${storage[@]}" -o "$out" --include-hidden-topics -e "$regex" &
pid=$!
trap 'kill -INT $pid 2>/dev/null; wait $pid 2>/dev/null; echo; ros2 bag info "$out" 2>/dev/null | head -20; echo "saved: $out"' INT TERM
wait $pid
