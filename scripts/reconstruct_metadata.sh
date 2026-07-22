#!/bin/bash

# Usage: bash reconstruct_metadata.sh gestion
# Walks data/raw/images/{newspaper}/ and writes a parquet metadata file
# matching the pipeline schema using Python one-liner for the parquet write.
# Run from ma_up/ root.

NEWSPAPER=${1:?Usage: bash reconstruct_metadata.sh <newspaper>}
IMAGE_ROOT="./data/raw/images/${NEWSPAPER}"
OUTPUT="./data/raw/metadata/${NEWSPAPER}_metadata.parquet"
TMP_CSV="./tmp/${NEWSPAPER}_metadata.csv"

if [ ! -d "$IMAGE_ROOT" ]; then
    echo "Error: directory $IMAGE_ROOT does not exist."
    exit 1
fi

echo "Scanning $IMAGE_ROOT..."

# Write CSV header
echo "newspaper,date,edition,page_number,page_url,image_url,image_path" > "$TMP_CSV"

# Walk all jpg files and parse fields from filename/path
find "$IMAGE_ROOT" -name "*.jpg" | sort | while read -r filepath; do
    # echo "Processing $filepath..."
    filename=$(basename "$filepath")

    # Extract date (YYYY-MM-DD) and page number from filename
    # Expected format: {newspaper}_{YYYY-MM-DD}_{page}.jpg
    date_part=$(echo "$filename" | grep -oP '\d{4}-\d{2}-\d{2}')
    page_number=$(echo "$filename" | grep -oP '(?<=_)\d+(?=\.jpg)')

    if [ -z "$date_part" ] || [ -z "$page_number" ]; then
        echo "Warning: could not parse $filename, skipping." >&2
        continue
    fi

    echo "${NEWSPAPER},${date_part},default,${page_number},,,$filepath" >> "$TMP_CSV"
done

COUNT=$(tail -n +2 "$TMP_CSV" | wc -l)
echo "Found $COUNT images. Writing parquet..."

# Convert CSV to parquet using Python
python3 - << PYEOF
import pandas as pd
from pathlib import Path

# Reconstructed from disk
new_df = pd.read_csv("$TMP_CSV")
new_df["date"] = pd.to_datetime(new_df["date"]).dt.strftime("%Y-%m-%d")
new_df["page_number"] = new_df["page_number"].astype(int)

output_path = "$OUTPUT"
key_cols = ["newspaper", "date", "page_number"]

if Path(output_path).exists():
    existing_df = pd.read_parquet(output_path)
    # Combine: existing pipeline rows take priority (keep="first" after sorting existing first)
    combined = pd.concat([existing_df, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=key_cols, keep="first")
    print(f"Merged {len(existing_df)} existing + {len(new_df)} reconstructed -> {len(combined)} total rows")
else:
    combined = new_df
    print(f"No existing parquet found, writing {len(combined)} reconstructed rows")

combined = combined.sort_values(key_cols).reset_index(drop=True)
combined.to_parquet(output_path, index=False)
print(f"Written -> {output_path}")
PYEOF

rm "$TMP_CSV"
echo "Done."