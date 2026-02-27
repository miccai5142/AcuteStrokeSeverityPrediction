#!/bin/bash -l

conda activate tensorflow


SCRATCH_DIR="MICCAI_DementiaSFCN_mRS" # !!!!!!!!!!!!!!!!!!!
SOURCE_DIR="/Data/SOOP/SOOP-T1w-LesionAligned"

echo "Copying dataset to scratch directory"
mkdir -p "$SCRATCH_DIR"
cp -r "$SOURCE_DIR" "$SCRATCH_DIR/"

echo "Dataset moved"

############# PROCESSED DATA PATHS #############
PARTICIPANT_DATA_FILE="$SCRATCH_DIR/SOOP-T1w-LesionAligned/SOOP_Participants_wLesionVol.tsv"
PARTICIPANT_IDS_FILE="$SCRATCH_DIR/SOOP-T1w-LesionAligned/filtered_SOOP_GoodRegistration_HighRes-sMRI_NoPriorStroke_AIS.txt"
IMAGE_DIR="$SCRATCH_DIR/SOOP-T1w-LesionAligned/SOOP-T1w-aligned"
DATASET_CSV="$SCRATCH_DIR/SOOP_mRS.csv"

echo "Preparing data at $DATASET_CSV"
# Prepare CSV once (shared across all repeats)
python3 /software/aispycore/data/prepare_mRS_data.py \
    --image_dir "$IMAGE_DIR" \
    --participant_data_file "$PARTICIPANT_DATA_FILE" \
    --participant_ids_file "$PARTICIPANT_IDS_FILE" \
    --output_csv "$DATASET_CSV" \
    --threshold 1

echo "Data preparation complete."

############# MODEL TRAINING OUTPUT PATHS #############
BASE_TRAINING_OUTPUT_DIR="$SCRATCH_DIR/MICCAI_DementiaSFCN_mRS" # !!!!!!!!!!!!!!!!!!!
RUN_STORAGE_DIR="runs"
MODEL_CONFIG_FILE="aispy_config_MICCAI.yaml"
SEEDS_FILE="seeds_ALL.txt"

# Create matching structure in final storage
FINAL_OUTPUT_DIR="${RUN_STORAGE_DIR}/MICCAI_DementiaSFCN_mRS" # !!!!!!!!!!!!!!!!!!!

mkdir -p "$BASE_TRAINING_OUTPUT_DIR"
mkdir -p "$FINAL_OUTPUT_DIR"

############# LOAD SEEDS FROM FILE #############
if [ ! -f "$SEEDS_FILE" ]; then
    echo "ERROR: Seeds file not found at $SEEDS_FILE"
    exit 1
fi

# Read seeds into array
mapfile -t SEEDS < "$SEEDS_FILE"
TOTAL_REPEATS=${#SEEDS[@]}
START_REPEAT=1  # !!!!!!!!!!!!!!!!!!!!!! Starting repeat number!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Starting repeat number

echo "Loaded ${TOTAL_REPEATS} seeds from $SEEDS_FILE"
echo "Seeds: ${SEEDS[@]}"

############# LOOP THROUGH REPEATS #############
echo "=========================================="
echo "Starting training loop for repeats ${START_REPEAT} to $((START_REPEAT + TOTAL_REPEATS - 1))"
echo "=========================================="

for i in $(seq 0 $((TOTAL_REPEATS - 1))); do
    REPEAT=$((START_REPEAT + i))
    SEED=${SEEDS[$i]}
    
    echo ""
    echo "=========================================="
    echo "Starting Repeat ${REPEAT} with seed ${SEED} at $(date)"
    echo "=========================================="
    
    # Create repeat-specific output directory in scratch
    REPEAT_OUTPUT_DIR="$BASE_TRAINING_OUTPUT_DIR/repeat_${REPEAT}"
    mkdir -p "$REPEAT_OUTPUT_DIR"
    
    # Run training for this repeat with specific seed
    python3 /software/aispycore/cli/train.py \
        --csv_file "$DATASET_CSV" \
        --output_dir "$REPEAT_OUTPUT_DIR" \
        --config "$MODEL_CONFIG_FILE" \
        --model_type "BinarySFCN" \
        --weights "dementia-2024" \
        --scheme "kfold" \
        --repeat_number "$REPEAT" \
        --seed "$SEED" 
    
    # Check if training succeeded
    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "ERROR: Repeat ${REPEAT} (seed ${SEED}) failed with exit code ${EXIT_CODE}!"
        # Clean up scratch folder
        echo "Cleaning up scratch directory..."
        [[ -n "$SCRATCH_DIR" && -d "$SCRATCH_DIR" ]] && rm -rf "$SCRATCH_DIR"
        echo "Stopping execution."
        exit 1
    fi
    
    echo "Repeat ${REPEAT} (seed ${SEED}) completed successfully at $(date)"
    
    # Copy only this repeat's folder to final storage (incremental backup)
    echo "Backing up Repeat ${REPEAT} results to ${FINAL_OUTPUT_DIR}/repeat_${REPEAT}"
    cp -r "$REPEAT_OUTPUT_DIR" "$FINAL_OUTPUT_DIR/"
    
    # Optional: Add small delay to ensure cleanup
    sleep 5
done

echo ""
echo "=========================================="
echo "All repeats completed at $(date)"
echo "=========================================="

# Copy any remaining root-level files (dataset CSV, logs, aggregated results, etc.)
echo "Copying root-level files to: ${FINAL_OUTPUT_DIR}"
for file in "$BASE_TRAINING_OUTPUT_DIR"/*; do
    # Skip repeat directories (already copied)
    if [[ ! -d "$file" || ! "$(basename "$file")" =~ ^repeat_ ]]; then
        cp -r "$file" "$FINAL_OUTPUT_DIR/"
    fi
done


echo "All outputs saved to ${FINAL_OUTPUT_DIR}"

# Clean up scratch folder
echo "Cleaning up scratch directory..."
[[ -n "$SCRATCH_DIR" && -d "$SCRATCH_DIR" ]] && rm -rf "$SCRATCH_DIR"

echo "=========================================="
echo "Job completed at $(date)"
echo "=========================================="
