#!/bin/bash

# Check if node number is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <node_number>"
  exit 1
fi

NODE_NUM=$1
NODE_FILE="batches/Node_${NODE_NUM}.txt"

# Step 1: Set environment variables
export FREESURFER_HOME=/usr/local/freesurfer
source $FREESURFER_HOME/SetUpFreeSurfer.sh

export FS_LICENSE="freesurfer_licence.txt"

# Step 2: Navigate to script location
cd / Deep_Learning_Preprocessing/scripts/SOOP-T1_Preprocessing/

# Step 3: Run script
python process_soop_t1w.py "$NODE_FILE" --max_images 50 --num_threads 6
