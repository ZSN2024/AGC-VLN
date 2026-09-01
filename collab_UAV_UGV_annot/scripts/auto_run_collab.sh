#!/bin/bash
# Auto-run air-ground collaboration N times.
# CARLA-Air must be running already (terminal 1).
#
# Usage:
#   bash collab_UAV_UGV_annot/scripts/auto_run_collab.sh [N] [EPISODE_ID]
#   ANN=<level> to select the map-annotation richness level (default: full):
#   none | goal | car_goal | line | full
#
# Flow per run:
#   1. setup_scene.py spawns vehicles + positions drone
#   2. run_collab.py runs UAV (modified SPF) + UGV (VLM path planning) in parallel

N=${1:-5}
EP=${2:-town10hd_001}
HEADLESS=${HEADLESS:-0}
ANN=${ANN:-full}
SETUP_EXTRA=""
[ "$HEADLESS" = "1" ] && SETUP_EXTRA="--headless"
DIR="$(cd "$(dirname "$0")/../.." && pwd)"
SCRIPTS="$DIR/collab_UAV_UGV_annot/scripts"
TEMPLATES="$DIR/collab_UAV_UGV_annot/episodes/town10hd_templates.json"
RUNS="$DIR/collab_UAV_UGV_annot/runs"
# Batch dir: default is per-level under THIS experiment's runs dir. Only honor
# an inherited BATCH_DIR if it already points inside $RUNS; a stale BATCH_DIR
# from another experiment (e.g. collab_UAV_UGV_alt) would save results into the
# wrong directory, so discard it.
case "$BATCH_DIR" in
  "$RUNS"/*) : ;;   # valid — keep it
  *) BATCH_DIR="$RUNS/batch_annot_${ANN}_$(date +%Y%m%d_%H%M%S)" ;;
esac
mkdir -p "$BATCH_DIR"

echo "============================================"
echo "  Collab UAV+UGV Auto-Run: $N runs on $EP  (annot=$ANN)"
echo "  UAV: modified SPF + annotate birdview"
echo "  UGV: send position → birdview → VLM → drive"
echo "============================================"

# ── Scene setup ──
echo "[1/3] Setting up scene..."
python "$SCRIPTS/setup_scene.py" --episode-id "$EP" --episode-file "$TEMPLATES" $SETUP_EXTRA &
SCENE_PID=$!
sleep 6

# ── Dual view ──
VIEW_PID=""
if [ "$HEADLESS" = "1" ]; then
    echo "[2/3] Headless: skipping dual view"
else
    echo "[2/3] Starting dual view..."
    python "$SCRIPTS/show_dual_view.py" &
    VIEW_PID=$!
    sleep 3
fi

# ── Run collaboration N times ──
echo "[3/3] Running collaboration..."
SUCCESS=0; FAIL=0
UAV_SUCC=0; UGV_SUCC=0
TIMES=""; UAV_PATHS=""; UGV_PATHS=""; UAV_DISTS=""; UGV_DISTS=""
UAV_VLMS=""; UGV_VLMS=""; TOTAL_VLMS=""
SUCC_TIMES=""; SUCC_VLMS=""

for i in $(seq 1 $N); do
    echo ""
    echo "--- Run $i/$N ---"
    if python "$SCRIPTS/run_collab.py" --episode-id "$EP" --auto --time-limit 180 --run-dir "$BATCH_DIR" --annotation "$ANN"; then
        SUCCESS=$((SUCCESS + 1)); echo "  -> SUCCESS"
    else
        FAIL=$((FAIL + 1)); echo "  -> FAILED"
    fi

    # Collect metrics from the latest run directory
    RDIR=$(ls -dt "$BATCH_DIR/collab_${EP}_"* 2>/dev/null | head -1)
    if [ -f "$RDIR/result.json" ]; then
        t=$(python3 -c "import json;print(json.load(open('$RDIR/result.json'))['time_s'])")
        up=$(python3 -c "import json;print(json.load(open('$RDIR/result.json'))['uav_path_m'])")
        gp=$(python3 -c "import json;print(json.load(open('$RDIR/result.json'))['ugv_path_m'])")
        ud=$(python3 -c "import json;print(json.load(open('$RDIR/result.json'))['uav_final_dist_m'])")
        gd=$(python3 -c "import json;print(json.load(open('$RDIR/result.json'))['ugv_final_dist_m'])")
        uv=$(python3 -c "import json;print(json.load(open('$RDIR/result.json'))['uav_vlm_calls'])")
        gv=$(python3 -c "import json;print(json.load(open('$RDIR/result.json'))['ugv_vlm_calls'])")
        tv=$(python3 -c "import json;print(json.load(open('$RDIR/result.json'))['total_vlm_calls'])")
        s=$(python3 -c "import json;print(json.load(open('$RDIR/result.json'))['success'])")
        us=$(python3 -c "import json;print(json.load(open('$RDIR/result.json'))['uav_success'])")
        gs=$(python3 -c "import json;print(json.load(open('$RDIR/result.json'))['ugv_success'])")
        TIMES="$TIMES $t"
        UAV_PATHS="$UAV_PATHS $up"; UGV_PATHS="$UGV_PATHS $gp"
        UAV_DISTS="$UAV_DISTS $ud"; UGV_DISTS="$UGV_DISTS $gd"
        UAV_VLMS="$UAV_VLMS $uv"; UGV_VLMS="$UGV_VLMS $gv"; TOTAL_VLMS="$TOTAL_VLMS $tv"
        if [ "$s" = "True" ]; then
            SUCC_TIMES="$SUCC_TIMES $t"; SUCC_VLMS="$SUCC_VLMS $tv"
        fi
        if [ "$us" = "True" ]; then UAV_SUCC=$((UAV_SUCC + 1)); fi
        if [ "$gs" = "True" ]; then UGV_SUCC=$((UGV_SUCC + 1)); fi
        echo "  time=${t}s uav_path=${up}m ugv_path=${gp}m uav_dist=${ud}m ugv_dist=${gd}m vlm=${tv} uav_ok=${us} ugv_ok=${gs}"
    fi

    # Reset scene between runs
    if [ $i -lt $N ]; then
        echo "  Resetting scene..."
        kill $SCENE_PID $VIEW_PID 2>/dev/null || true
        python -c "
import carla, time
c=carla.Client('127.0.0.1',2000);c.set_timeout(5)
w=c.get_world()
for v in w.get_actors().filter('vehicle.*'):
    try:v.destroy()
    except:pass
" 2>/dev/null
        sleep 2
        # Restart view with fresh cameras (skip in headless)
        if [ "$HEADLESS" != "1" ]; then
            python "$SCRIPTS/show_dual_view.py" &
            VIEW_PID=$!
            sleep 2
        fi
        python "$SCRIPTS/setup_scene.py" --episode-id "$EP" --episode-file "$TEMPLATES" $SETUP_EXTRA &
        SCENE_PID=$!
        sleep 6
    fi
done

kill $SCENE_PID $VIEW_PID 2>/dev/null || true

# ── Summary ──
TOTAL=$((SUCCESS + FAIL))
RATE=$(awk "BEGIN {printf \"%.0f\", $SUCCESS*100/$TOTAL}")
UAV_RATE=$(awk "BEGIN {printf \"%.0f\", $UAV_SUCC*100/$TOTAL}")
UGV_RATE=$(awk "BEGIN {printf \"%.0f\", $UGV_SUCC*100/$TOTAL}")

_avg() { echo "$1" | awk '{s=0;for(i=1;i<=NF;i++)s+=$i;if(NF>0)printf "%.1f",s/NF;else print "N/A"}'; }

AVG_TIME=$(_avg "$TIMES")
AVG_UAV_PATH=$(_avg "$UAV_PATHS")
AVG_UGV_PATH=$(_avg "$UGV_PATHS")
AVG_UAV_DIST=$(_avg "$UAV_DISTS")
AVG_UGV_DIST=$(_avg "$UGV_DISTS")
AVG_UAV_VLM=$(_avg "$UAV_VLMS")
AVG_UGV_VLM=$(_avg "$UGV_VLMS")
AVG_TOTAL_VLM=$(_avg "$TOTAL_VLMS")
AVG_SUCC_TIME=$(_avg "$SUCC_TIMES")
AVG_SUCC_VLM=$(_avg "$SUCC_VLMS")

REPORT="============================================
  Collab UAV+UGV Results (annot=$ANN)
  Joint Success (any): $SUCCESS/$TOTAL ($RATE%)
  UAV Success:    $UAV_SUCC/$TOTAL ($UAV_RATE%)
  UGV Success:    $UGV_SUCC/$TOTAL ($UGV_RATE%)
  Avg Time:        ${AVG_TIME}s
  Avg UAV Path:    ${AVG_UAV_PATH}m
  Avg UGV Path:    ${AVG_UGV_PATH}m
  Avg UAV Final Dist: ${AVG_UAV_DIST}m
  Avg UGV Final Dist: ${AVG_UGV_DIST}m
  Avg UAV VLM calls:  ${AVG_UAV_VLM}
  Avg UGV VLM calls:  ${AVG_UGV_VLM}
  Avg Total VLM:      ${AVG_TOTAL_VLM}
  Succ Avg Time:      ${AVG_SUCC_TIME}s
  Succ Avg VLM:       ${AVG_SUCC_VLM}
============================================"

echo ""
echo "$REPORT"
# Summary goes inside the latest run folder (fall back to batch dir if none)
echo "$REPORT" > "${RDIR:-$BATCH_DIR}/summary.txt"
