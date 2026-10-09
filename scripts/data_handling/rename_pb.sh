#!/bin/bash

# Rename all gestion_* files under data/raw/images/publimetro/ to publimetro_*

find data/raw/images/publimetro -name "gestion_*.jpg" | while read -r file; do
    dir=$(dirname "$file")
    filename=$(basename "$file")
    new_filename="${filename/gestion_/publimetro_}"
    mv "$file" "$dir/$new_filename"
done

echo "Done."