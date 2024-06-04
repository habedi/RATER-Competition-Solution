#!/bin/bash

# List of datasets to download (see https://www.kaggle.com/code/cdeotte/2nd-place-solution-cv741-public727-private740)
datasets=(
    cdeotte/bird-base
    cdeotte/deberta-large-100
    cdeotte/deberta-large-v2
    cdeotte/deberta-lstm
    cdeotte/deberta-lstm-jaccard
    cdeotte/deberta-v2-xlarge
    cdeotte/deberta-xlarge-1536
    cdeotte/funnel-large-6folds
    ubamba98/feedbackyoso
    ubamba98/auglsgrobertalarge
    ubamba98/longformerwithbilstmhead
)

# Directory to store downloaded datasets
models_dir="models"

# Create models directory if it doesn't exist
mkdir -p "$models_dir"

# Loop through each dataset
for dataset in "${datasets[@]}"; do
    # Extract dataset name from the URL
    dataset_name=$(basename "$dataset")
    
    # Check if dataset zip file already exists in models folder
    if [ ! -f "$models_dir/$dataset_name.zip" ]; then
        echo "Downloading $dataset..."
        # Download dataset using Kaggle CLI
        kaggle datasets download -d "$dataset" -p "$models_dir"
        echo "Downloaded $dataset successfully."
    else
        echo "Skipping $dataset as it already exists."
    fi
done
