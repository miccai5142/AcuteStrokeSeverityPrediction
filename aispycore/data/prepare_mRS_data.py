#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import numpy as np

def main(args):
    image_dir = args.image_dir
    participant_data_file = args.participant_data_file
    participant_ids_file = args.participant_ids_file
    output_csv = args.output_csv
    threshold = args.threshold
    
    # Load participant IDs if provided
    participant_ids = None
    if participant_ids_file:
        print("Participant ID File provided.")
        participant_ids = np.loadtxt(participant_ids_file, dtype=str)  # adjust dtype=int if IDs are integers

    # Create simplified CSV if it doesn't exist
    if not os.path.isfile(output_csv):
        labels = pd.read_csv(participant_data_file, sep='\t')
        labels = labels[~pd.isna(labels['age'])]

        # Ensure 'participant_id' column exists
        if 'participant_id' not in labels.columns:
            raise ValueError("participant_data_file must contain a 'participant_id' column")
        labels['id'] = labels['participant_id']

        # Construct NIfTI paths
        labels['path'] = labels['id'].apply(lambda x: os.path.join(image_dir, x, 'cropped.nii.gz'))
        labels = labels[labels['path'].apply(lambda x: os.path.isfile(x))]

        # Keep relevant columns and rename
        labels = labels[['path', 'id', 'age', 'sex', 'nihss', 'gs_rankin_6isdeath', 'lesion_volume']]
        labels = labels.rename(columns={'gs_rankin_6isdeath': 'mRS'})
        labels = labels.dropna(subset=['mRS'])

        # Create binary label
        labels['mRS_Label'] = (labels['mRS'] > threshold).astype(int)

        # Filter by pre-selected IDs if provided
        if participant_ids is not None:
            labels = labels[labels['id'].isin(participant_ids)]

        # Save output CSV
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        labels.to_csv(output_csv, index=False)
        print(f"Labels CSV file successfully created at {output_csv}")
    else:
        print(f"Labels CSV already exists at {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create filtered participant CSV for ML pipeline.")
    parser.add_argument("--image_dir", type=str, required=True, help="Folder containing participant image subfolders.")
    parser.add_argument("--participant_data_file", type=str, required=True, help="TSV file with participant demographic/clinical data.")
    parser.add_argument("--participant_ids_file", type=str, required=False, default=None,
                        help="Optional text file with pre-selected participant IDs to filter.")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to output CSV file for filtered labels.")
    parser.add_argument("--threshold", type=int, required=False, default=2, help="Threshold for binarizing mRS Score. Default mRS > 2 is labelled '1'.")
    args = parser.parse_args()

    main(args)



# python prepare_data.py \
#     --image_dir /path/to/dataset/images \
#     --participant_data_file /path/to/participant_prediction_target_data.tsv \
#     --participant_ids_file /optional/path/to/preselected/participant/ids.txt \
#     --output_csv /path/to/output.csv
