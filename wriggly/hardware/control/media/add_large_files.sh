#!/bin/bash

# Install Git LFS if not already installed
if ! command -v git-lfs &> /dev/null; then
    echo "Git LFS not found. Installing Git LFS..."
    # Add installation command here
fi

# Initialize Git LFS
git lfs install

# Track large files
git lfs track "*.mp4"
git lfs track "*.avi"

# Add all large files
git lfs track -- $(find . -type f -size +100M \( -name "*.mp4" -o -name "*.avi" \))

# Commit changes
git add .
git commit -m "Add large files"

# Push changes
git push
