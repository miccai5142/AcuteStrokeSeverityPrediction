#!/bin/bash

# ------------------------------------------------------------------
# Attribution
#
# This script uses preprocessing components from:
# Pyment (public repository)
# https://github.com/estenhl/pyment-public
#
# Associated publication:
# Leonardsen, E.H. et al. (2024).
# Constructing personalized characterizations of structural brain
# aberrations in patients with dementia using explainable artificial
# intelligence. npj Digital Medicine, 7(1), 110.
# https://doi.org/10.1038/s41746-024-01123-7
#
# Please cite the above repository and paper if using this code.
# ------------------------------------------------------------------


# Required arguments
filename=false
destination=false
freesurfer_license=false

script_name=$(basename "$0")
usage_string="Usage: bash $script_name --filename <filename> --destination <destination> --template <MNI152_T1_brain template from FSL> [--help]"

# Parses arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -h|--help) echo "$usage_string"; exit ;;
        -f|--filename) filename="$2"; shift ;;
        -d|--destination) destination="$2"; shift ;;
        -t|--template) template="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Validates required arguments
if [ "$filename" = false ]; then
    echo "Error: required argument --filename is missing." >&2
    echo "$usage_string" >&2
    exit 1
elif [ ! -f "$filename" ]; then
    echo "Error: required argument --filename must point to an existing file." >&2
    echo "$usage_string" >&2
    exit 1
fi

if [ "$destination" = false ]; then
    echo "Error: required argument --destination is missing." >&2
    echo "$usage_string" >&2
    exit 1
fi

if [ "$template" = false ]; then
    echo "Error: required argument --template is missing." >&2
    echo "$usage_string" >&2
    exit 1
elif [ ! -f "$template" ]; then
    echo "Error: required argument --template must point to an existing file." >&2
    echo "$usage_string" >&2
    exit 1
fi

code_for_cropping="
import sys
import nibabel as nib

input, output = sys.argv[1:]
img = nib.load(input)
img = img.slicer[6:173,2:214,0:160]
img = nib.Nifti1Image(img.get_fdata() / 255., img.affine, img.header)
nib.save(img, output)
"
echo "=============================================================================="
echo "Destination directory: $destination"
echo "=============================================================================="

cropped=$destination/mri/cropped.nii.gz

if [ ! -f "$cropped" ]; then
    
    registered=$destination/mri/mni152.nii.gz	
    echo "=============================================================================="
    echo "Cropped image doesn't exist. Using python to create it: $cropped"
    echo "=============================================================================="
    
    if [ ! -f "$registered" ]; then
        echo "=============================================================================="
        echo "MNI152-registered image does not exist. Using flirt to create it: $registered"
        echo "=============================================================================="
        reoriented=$destination/mri/reoriented.nii.gz

        # If the reoriented image doesn't exist, use fslreorient2std to create it
        if [ ! -f "$reoriented" ]; then

            echo "=============================================================================="
            echo "Reoriented image doesn't exist. Running fslreorient2std: $reoriented"
            echo "=============================================================================="

            transformed=$destination/mri/brainmask.nii.gz

            # If the transformed image doesn't exist, use mri-convert to create it
            if [ ! -f "$transformed" ]; then

                echo "=============================================================================="
                echo "Nifti doesn't exist. Running mri-convert: $transformed"
                echo "=============================================================================="

                brainmask=$destination/mri/brainmask.mgz

                # If the brainmask doesn't exist, run recon-all to create it
                if [ ! -f "$brainmask" ]; then
                    
                    echo "=============================================================================="
                    echo "Brainmask does not exist. Running FreeSurfer recon-all: $brainmask"
                    echo "=============================================================================="
                    
                    subject_id=$(basename "$destination")
                    subjects_dir=$(dirname "$destination")
                    recon-all -subjid $subject_id -i $filename -sd $subjects_dir -autorecon1
                fi
                mri_convert $brainmask $transformed -ot nii
            fi
            fslreorient2std $transformed $reoriented
        fi
        flirt -in $reoriented -out $registered -ref $template -dof 6
    fi
    python -c "$code_for_cropping" $registered $cropped
fi
