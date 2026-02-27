#!/bin/bash

# Script to flip MRI scans for specified participants using fslswapdim
# Usage: ./flip_brains.sh /path/to/parent_dir participants_to_flip.txt

# Exit immediately if a command exits with a non-zero status
set -e

# Arguments
PARENT_DIR=$1
ID_LIST=$2

if [ -z "$PARENT_DIR" ] || [ -z "$ID_LIST" ]; then
  echo "Usage: $0 <parent_dir> <id_list.txt>"
  exit 1
fi

if [ ! -d "$PARENT_DIR" ]; then
  echo "Error: Parent directory '$PARENT_DIR' does not exist."
  exit 1
fi

if [ ! -f "$ID_LIST" ]; then
  echo "Error: ID list file '$ID_LIST' does not exist."
  exit 1
fi

echo "Starting flipping process..."
echo "Parent directory: $PARENT_DIR"
echo "ID list: $ID_LIST"
echo "-----------------------------------"

while read -r PARTICIPANT_ID; do
  # Clean the participant ID (remove BOM, CRLF, whitespace)
  PARTICIPANT_ID=$(echo "$PARTICIPANT_ID" | tr -d '\r' | xargs)

  # Skip empty lines
  if [ -z "$PARTICIPANT_ID" ]; then
    continue
  fi

  IMG_DIR="${PARENT_DIR}/${PARTICIPANT_ID}"
  IMG_FILE="${IMG_DIR}/mni152_cropped.nii.gz"
  UNFLIPPED_FILE="${IMG_DIR}/mni152_cropped_unflipped.nii.gz"

  if [ ! -d "$IMG_DIR" ]; then
    echo "Participant directory not found: $IMG_DIR"
    continue
  fi

  if [ ! -f "$IMG_FILE" ]; then
    echo "No image found for participant $PARTICIPANT_ID at $IMG_FILE"
    continue
  fi

  echo "Processing participant: $PARTICIPANT_ID"
  echo "Renaming original image → mni152_cropped_unflipped.nii.gz"
  mv "$IMG_FILE" "$UNFLIPPED_FILE"

  echo "Flipping image (left-right)..."
  fslswapdim "$UNFLIPPED_FILE" -x y z "$IMG_FILE"

  echo "Flipped image saved as: $IMG_FILE"
  echo "-----------------------------------"
done < "$ID_LIST"


echo "All requested participants processed."
