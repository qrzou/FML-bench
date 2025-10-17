#!/bin/bash

# Set TIME for this run
TIME=$(date +"%Y-%m-%d_%H-%M-%S")

# Determine the base folder dynamically (the current folder you run the script in)
BASE_DIR=$(pwd)
echo "BASE_DIR: $BASE_DIR"

# Compile logger.so
gcc -shared -fPIC -o ../../logger.so ../../ml_tasks/logger.c -ldl 
echo "Compiled logger.so"

# Move any previous *_* folders into timestamped folder
LAST_MOVED=""
for dir in "$BASE_DIR"/*; do
    if [ -d "$dir" ]; then
        FOLDER_NAME=$(basename "$dir")
        # Strip any old timestamp suffix
        BASE_NAME=$(echo "$FOLDER_NAME" | sed 's/_[0-9]\{4\}-[0-9]\{2\}-[0-9]\{2\}_[0-9]\{2\}-[0-9]\{2\}-[0-9]\{2\}//')
        NEW_NAME="${BASE_NAME}_$TIME"
        mv "$dir" "$NEW_NAME" 2>/dev/null || true
        echo "Moving $dir to $NEW_NAME"
        LAST_MOVED="$NEW_NAME"
    fi
done

# Change to the last moved folder (the actual project run folder)
if [ -n "$LAST_MOVED" ]; then
    cd "$LAST_MOVED"
    echo "Changed to project folder: $LAST_MOVED"
else
    echo "No project folder was moved, staying in $BASE_DIR"
fi

# Clean agent runs
rm -rf _agent_runs && git restore . && git clean -fd
echo "Cleaned agent runs"

# Setup for benchmarking repo (different for different repo!!!)
REPO_FOLDER_NAME="Privacy_privacymeter"
cp ../../../ml_tasks/Privacy_privacymeter/cifar10_benchmark.yaml ./configs/
cp ../../../ml_tasks/Privacy_privacymeter/postprocess.py ./
chmod 555 models/utils.py
chmod 555 audit.py
chmod 555 util.py
chmod 555 get_signals.py
chmod 555 run_mia.py
chmod 555 configs/cifar10_benchmark.yaml

# Prepare benchmark_results output dir
# Relative path
REL_PATH="../../../benchmark_results/claude_code/$REPO_FOLDER_NAME/$TIME"

# Convert to absolute path
LOG_DIR=$(mkdir -p "$REL_PATH" && realpath "$REL_PATH")
echo "Created log dir: $LOG_DIR"

# Set LOGGER_FILE
export LOGGER_FILE="$LOG_DIR/agent-commands.log"
echo "Set LOGGER_FILE: $LOGGER_FILE"

# Run agent with LD_PRELOAD logger
echo "Running agent with LD_PRELOAD logger"
START_TIME=""
END_TIME=""
if [ -f "../prompt.txt" ]; then
    START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
    LD_PRELOAD="../../../logger.so" claude -p "$(cat ../prompt.txt)" --dangerously-skip-permissions --disallowedTools "Bash(chmod:*)" >> "$LOGGER_FILE" 2>&1
    END_TIME=$(date +"%Y-%m-%d %H:%M:%S")
else
    echo "No prompt.txt found in parent directory!" >> "$LOGGER_FILE"
fi
echo "Done"

# Save process time log to _agent_runs/process_time_log.txt
if [ -d "_agent_runs" ]; then
    {
        echo "start_time: $START_TIME"
        echo "end_time: $END_TIME"
    } > "_agent_runs/process_time_log.txt"
    echo "Saved process time log to _agent_runs/process_time_log.txt"
else
    echo "_agent_runs directory not found, could not save process_time_log.txt"
fi

# go back to the original directory
cd ..
echo "Back to original directory"

# postprocess: copy _agent_runs
TIME_DIR=$(ls -d *_$TIME 2>/dev/null | sort | tail -n 1)
[ -z "$TIME_DIR" ] && echo "No directory matching '*_$TIME' found."
echo "Using time dir: $TIME_DIR"
cp -r "$TIME_DIR/_agent_runs" "$LOG_DIR"
echo "Copied _agent_runs to $LOG_DIR"

# Convert to sessionId format: replace _ with -
SESSION_TIME="${TIME//_/-}"

# Extract and filter ccusage session
CCJSON=$(npx --yes ccusage@latest session --json 2>/dev/null)

# Only process if output is non-empty
if [ -n "$CCJSON" ]; then
python3 - <<PYTHON > "$LOG_DIR/ccusage_session.json"
import sys, json
session_time = "$SESSION_TIME"
data = json.loads("""$CCJSON""")
sessions = data.get("sessions", [])
filtered = [s for s in sessions if session_time in s.get("sessionId","")]
json.dump({"sessions": filtered}, sys.stdout, indent=2)
PYTHON
    echo "Saved filtered ccusage session to $LOG_DIR/ccusage_session.json"
else
    echo '{"sessions":[]}' > "$LOG_DIR/ccusage_session.json"
    echo "No ccusage session output; created empty JSON at $LOG_DIR/ccusage_session.json"
fi