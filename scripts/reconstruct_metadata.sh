#!/bin/bash

# Output CSV with columns: newspaper, date, edition, page_number, scale, year

echo "newspaper,date,edition,page_number,scale,year"

find data/raw/images -name "*.jpg" | while read -r file; do
    newspaper=$(echo "$file" | cut -d'/' -f4)
    year=$(echo "$file" | cut -d'/' -f5)

    # Apply year filter
    if [ "$newspaper" = "elcomercio" ]; then
        [ "$year" -lt 2020 ] && continue
    else
        [ "$year" -ge 2020 ] && continue
    fi

    # Parse filename: {newspaper}_{YYYY-MM-DD}_{page}.jpg
    filename=$(basename "$file" .jpg)
    date=$(echo "$filename" | grep -oP '\d{4}-\d{2}-\d{2}')
    page_number=$(echo "$filename" | grep -oP '(?<=_)\d+$')

    echo "${newspaper},${date},default,${page_number},NA,${year}"

done