#!/bin/bash

DST="./data/raw/images/ojo"

find "$DST" -name "*#*.jpg" | while read -r file; do
    filename=$(basename "$file")
    date_part=$(echo "$filename" | grep -oP '\d{4}-\d{2}-\d{2}')
    page=$(echo "$filename" | grep -oP '(?<=#)\d+' | sed 's/^0*//')

    year=$(echo "$date_part" | cut -d'-' -f1)
    month=$(echo "$date_part" | cut -d'-' -f2)
    day=$(echo "$date_part" | cut -d'-' -f3)

    new_dir="$DST/$year/$month/$day"
    new_file="ojo_${date_part}_${page}.jpg"

    mkdir -p "$new_dir"
    mv "$file" "$new_dir/$new_file"

done

find "$DST" -type d -empty -delete

echo "Done."