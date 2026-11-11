#!/usr/bin/env bash
# The ERC arena in MuJoCo for this solution: CPU only, no GPU or display needed.
#
#   sim/sim.sh build                      build the image (once; ~15-30 min, needs internet)
#   sim/sim.sh up [SEED] [ROUND]          start the simulator (default seed 42, round 1)
#   sim/sim.sh run COLUMN COLOUR [ARGS]   build and run the solution (Ctrl-C stops it)
#   sim/sim.sh record start               start recording the arena + head cameras
#   sim/sim.sh record stop [OUT.mp4]      stop and write the video (simulation speed)
#   sim/sim.sh shell                      a shell in the container (ROS sourced)
#   sim/sim.sh down                       stop everything
#
# Env: ROS_DOMAIN_ID (default 30), BOOK_MU=10 (Gazebo's sticky books; default: realistic friction),
#      SIM_SPEED (default 1.0 = real time; a slow CPU simply runs slower, the video still plays at
#      real speed), SIM_CPUS=2 (cap the container's CPUs, e.g. to try a weak machine),
#      DEPTH=d435 (the real D435's depth noise and holes, measured on the robot; default: exact, as Gazebo).
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
IMAGE=4to1mux-mujoco:latest
NAME=4to1mux_sim
DOMAIN=${ROS_DOMAIN_ID:-30}
ROS='source /opt/ros/humble/setup.bash; source /opt/erc_ws/install/setup.bash; [ -f /opt/sol_ws/install/setup.bash ] && source /opt/sol_ws/install/setup.bash; export CYCLONEDDS_URI="<CycloneDDS><Domain><Discovery><MaxAutoParticipantIndex>120</MaxAutoParticipantIndex></Discovery></Domain></CycloneDDS>";'
TTY=$([ -t 0 ] && echo -it || true)
cx() { docker exec "$NAME" bash -c "$ROS $*"; }

case "${1:-}" in
build)
  docker build -t "$IMAGE" "$REPO/sim/mujoco" ;;

up)
  SEED=${2:-42}; ROUND=${3:-1}
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  docker run -d --name "$NAME" --network host --ipc host ${SIM_CPUS:+--cpus $SIM_CPUS} \
    -e ROS_DOMAIN_ID="$DOMAIN" -e ROS_LOCALHOST_ONLY=1 -e MUJOCO_GL=egl -e ERC_DEPTH="${DEPTH:-exact}" \
    -v "$REPO":/opt/sol_ws/src/solution_4to1mux "$IMAGE" sleep infinity >/dev/null
  echo "[sim] arena seed $SEED round $ROUND (domain $DOMAIN), starting..."
  docker exec -d "$NAME" bash -c "$ROS ros2 launch erc_bringup mujoco_simulation.launch.py \
    seed:=$SEED round:=$ROUND sim_speed:=${SIM_SPEED:-1.0} headless:=true depth_cloud:=false \
    ${BOOK_MU:+book_mu:=$BOOK_MU} > /tmp/sim.log 2>&1"
  for _ in $(seq 1 120); do
    n=$(docker exec "$NAME" grep -ac 'Configured and activated' /tmp/sim.log 2>/dev/null || true)
    [ "${n:-0}" -ge 8 ] && { echo "[sim] ready: $n controllers active"; exit 0; }
    docker exec "$NAME" grep -aqE 'process has died|Traceback' /tmp/sim.log 2>/dev/null && break
    sleep 3
  done
  echo "[sim] did not come up; last lines of the log:"; docker exec "$NAME" tail -20 /tmp/sim.log; exit 1 ;;

run)
  [ $# -ge 3 ] || { sed -n 2,12p "$0"; exit 1; }
  COL=$2; COLOUR=$3; shift 3
  cx "cd /opt/sol_ws && colcon build --symlink-install --packages-select solution_4to1mux 2>&1 | tail -1"
  docker exec $TTY "$NAME" bash -c "$ROS source /opt/sol_ws/install/setup.bash; ros2 launch solution_4to1mux \
    solution.launch.py robot_profile:=simulation shelf_column_number:=$COL book_colour:=$COLOUR $*" ;;

record)
  case "${2:-}" in
  start)
    cx "rm -rf /tmp/rec; pkill -f '[r]ecord.py' || true"
    docker exec -d "$NAME" bash -c "$ROS python3 /opt/sol_ws/src/solution_4to1mux/sim/tools/record.py /tmp/rec > /tmp/rec.log 2>&1"
    echo "[sim] recording; stop with: sim/sim.sh record stop [OUT.mp4]" ;;
  stop)
    OUT=${3:-run_$(date +%Y%m%d_%H%M%S).mp4}
    cx "pkill -INT -f '[r]ecord.py'; sleep 3; tail -1 /tmp/rec.log"
    cx "python3 /opt/sol_ws/src/solution_4to1mux/sim/tools/make_video.py /tmp/rec /tmp/out.mp4 --label '$(basename "$OUT" .mp4)'"
    docker cp "$NAME:/tmp/out.mp4" "$OUT" && echo "[sim] video: $OUT" ;;
  *) sed -n 2,12p "$0"; exit 1 ;;
  esac ;;

shell)
  docker exec -it "$NAME" bash -c "$ROS exec bash" ;;

down)
  docker rm -f "$NAME" >/dev/null 2>&1 && echo "[sim] stopped" || true ;;

*) sed -n 2,15p "$0"; exit 1 ;;
esac
