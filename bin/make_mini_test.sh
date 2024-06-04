#!/bin/bash

# Source directory containing files
source_dir="data/competition_data/test"

# Destination directory for copied files
destination_dir="data/competition_data/mini_test"

# Create destination directory if it doesn't exist and remove the old one
rm -rf "$destination_dir"
mkdir -p "$destination_dir"

# List files in the source directory, shuffle them, and select the first 10
files_to_copy=$(ls "$source_dir" | shuf -n 32)

# Loop through selected files and copy them to the destination directory
for file in $files_to_copy; do
    cp "$source_dir/$file" "$destination_dir"
    echo "Copied $file to $destination_dir"
done
