#!/bin/bash

# Source directory containing zip files
source_dir="models"

# Destination directory for extracted files
destination_dir="model_pack_1"

# Create destination directory if it doesn't exist
mkdir -p "$destination_dir"
echo "Destination directory '$destination_dir' created."

# Loop through each zip file in the source directory
for zip_file in "$source_dir"/*.zip; do
    # Check if zip file exists
    if [ -f "$zip_file" ]; then
        # Extract zip file to a folder with the same name as the zip file
        folder_name="${zip_file%.zip}"
        folder_name="${folder_name##*/}"
        extract_dir="$destination_dir/$folder_name"
        mkdir -p "$extract_dir"
        echo "Extracting $zip_file to $extract_dir"
        unzip -q "$zip_file" -d "$extract_dir"
        echo "Extraction of $zip_file completed"
        rm "$zip_file"
        echo "Removed $zip_file"
    fi
done

echo "Extraction completed."
