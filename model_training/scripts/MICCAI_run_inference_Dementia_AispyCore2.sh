#!/bin/bash -l

conda activate tensorflow

SCRATCH_DIR="/tmp/MICCAI_FT_Dementia_Inference_SOOP"
SOURCE_DIR="/Data/SOOP/SOOP-T1w-LesionAligned"

mkdir -p "$SCRATCH_DIR"
cp -r "$SOURCE_DIR" "$SCRATCH_DIR/"

PARTICIPANT_DATA_FILE="$SCRATCH_DIR/SOOP-T1w-LesionAligned/SOOP_Participants_wLesionVol.tsv"
PARTICIPANT_IDS_FILE="$SCRATCH_DIR/SOOP-T1w-LesionAligned/filtered_SOOP_GoodRegistration_HighRes-sMRI_NoPriorStroke_AIS.txt"
IMAGE_DIR="$SCRATCH_DIR/SOOP-T1w-LesionAligned/SOOP-T1w-aligned"
DATASET_CSV="$SCRATCH_DIR/SOOP_mRS.csv"

echo "Preparing master dataset CSV"
python3 prepare_mRS_data.py \
    --image_dir "$IMAGE_DIR" \
    --participant_data_file "$PARTICIPANT_DATA_FILE" \
    --participant_ids_file "$PARTICIPANT_IDS_FILE" \
    --output_csv "$DATASET_CSV" \
    --threshold 1

############# MODEL + OUTPUT PATHS #############
PATH_TO_MODEL_RUN_FOLDER="runs/MICCAI_DementiaSFCN_mRS_1"
MODEL_CONFIG_FILE="scripts/aispy_config_MICCAI_Inference.yaml"

BASE_INFERENCE_OUTPUT_DIR="${SCRATCH_DIR}/MICCAI_FT_Dementia_Inference_SOOP"
FINAL_OUTPUT_DIR="inference/MICCAI_FT_Dementia_Inference_SOOP"

mkdir -p "$BASE_INFERENCE_OUTPUT_DIR"
mkdir -p "$FINAL_OUTPUT_DIR"

echo "=========================================="
echo "Starting inference over all repeats"
echo "=========================================="


for OUTER_REPEAT_DIR in "${PATH_TO_MODEL_RUN_FOLDER}"/repeat_*; do
    if [[ ! -d "$OUTER_REPEAT_DIR" ]]; then
        continue
    fi

    # Raw repeat number from the outer directory name (e.g. "10" from "repeat_10")
    REPEAT_NUMBER=$(basename "$OUTER_REPEAT_DIR" | sed 's/repeat_//')

    # Zero-padded inner directory name matching train.py's f"repeat_{N:03d}" format
    INNER_REPEAT_DIR="${OUTER_REPEAT_DIR}/repeat_$(printf '%03d' "$REPEAT_NUMBER")"

    echo "------------------------------------------"
    echo "Repeat ${REPEAT_NUMBER}"
    echo "  outer: ${OUTER_REPEAT_DIR}"
    echo "  inner: ${INNER_REPEAT_DIR}"
    echo "------------------------------------------"

    if [[ ! -d "$INNER_REPEAT_DIR" ]]; then
        echo "WARNING: Expected inner directory not found: ${INNER_REPEAT_DIR}. Skipping."
        continue
    fi

    # Count fold directories inside the inner repeat dir
    FOLD_COUNT=$(find "$INNER_REPEAT_DIR" -maxdepth 1 -type d -name "fold_*" | wc -l)
    if [[ "$FOLD_COUNT" -eq 0 ]]; then
        echo "WARNING: No fold_XX/ directories found in ${INNER_REPEAT_DIR}. Skipping."
        continue
    fi
    echo "Found ${FOLD_COUNT} fold(s)"

    INFERENCE_OUT="${BASE_INFERENCE_OUTPUT_DIR}/repeat_${REPEAT_NUMBER}"

    python3 /software/aispycore/cli/predict.py \
        --csv_file     "$DATASET_CSV" \
        --training_dir "$INNER_REPEAT_DIR" \
        --output_dir   "$INFERENCE_OUT" \
        --config       "$MODEL_CONFIG_FILE"

    if [[ $? -ne 0 ]]; then
        echo "ERROR during inference for repeat ${REPEAT_NUMBER}"
        exit 1
    fi

    echo "Repeat ${REPEAT_NUMBER} complete → ${INFERENCE_OUT}"
done

############# BACKUP + CLEANUP #############
echo "=========================================="
echo "All repeats complete. Backing up results."
echo "=========================================="

cp -r "$BASE_INFERENCE_OUTPUT_DIR" "$FINAL_OUTPUT_DIR/"

echo "Cleaning scratch directory"
rm -rf "$SCRATCH_DIR"

echo "=========================================="
echo "Inference job completed at $(date)"
echo "=========================================="