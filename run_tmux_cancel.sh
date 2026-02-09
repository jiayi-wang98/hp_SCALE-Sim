#!/bin/bash

# ==========================================
# CONFIGURATION
# ==========================================

# 1. Select the input text file containing commands
INPUT_FILE="commands/command6_bf16.txt"

# 2. Define which lines to run (space-separated list of line numbers)
#    Example: JOBS_TO_RUN=(1 3 5) to run lines 1, 3, and 5.
#JOBS_TO_RUN=(1 2 3 4 5 6 7 8 9 10)
#JOBS_TO_RUN=(11 12 13 14 15 16 17 18 19 20)
JOBS_TO_RUN=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45)
# ==========================================
# EXECUTION LOGIC
# ==========================================

# Check if input file exists
if [[ ! -f "$INPUT_FILE" ]]; then
    echo "Error: File $INPUT_FILE not found!"
    exit 1
fi

# Read the file into an array called 'commands'
mapfile -t commands < "$INPUT_FILE"

echo "----------------------------------------"
echo "Target File: $INPUT_FILE"
echo "Jobs to run: ${JOBS_TO_RUN[*]}"
echo "----------------------------------------"

for line_num in "${JOBS_TO_RUN[@]}"; do
    # Array index is 0-based, so subtract 1 from the line number
    idx=$((line_num - 1))
    
    # Check if the line exists in the file
    if [[ -z "${commands[$idx]}" ]]; then
        echo "Warning: Line $line_num is empty or out of bounds. Skipping."
        continue
    fi

    cmd="${commands[$idx]}"
    
    # ------------------------------------------------------
    # MAPPING RULE: Define how line numbers map to Tmux sessions
    # Current Rule: Tmux Session Name = Line Number
    # MODIFY HERE to change mapping logic if needed.
    session_name="$line_num" 
    # ------------------------------------------------------

    echo "Canceling Line $line_num in Tmux Session '$session_name'..."
    echo "  Command: $cmd"

    # Check if session exists. If not, create it.
    if ! tmux has-session -t "$session_name" 2>/dev/null; then
        echo "  -> Session '$session_name' not found. Creating new detached session."
        tmux new-session -d -s "$session_name"
    fi

    # Send Ctrl+C to cancel the running command in this session
    tmux send-keys -t "$session_name" C-c

done

echo "----------------------------------------"
echo "All jobs submitted."
