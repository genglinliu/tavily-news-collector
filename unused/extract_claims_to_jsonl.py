import json
import os
from pathlib import Path

# Input and output file paths
input_file = "enriched_covid_data_fact_20250225_011458.json"
output_file = "claims_covid.jsonl"

# Get the directory of the current script
script_dir = Path(__file__).parent

# Create full paths
input_path = script_dir / input_file
output_path = script_dir / output_file

# Read the JSON file
print(f"Reading from {input_path}")
with open(input_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Extract claims and write to JSONL
print(f"Extracting claims and writing to {output_path}")
with open(output_path, 'w', encoding='utf-8') as f:
    for item in data:
        if 'claim' in item:
            # Write each claim as a separate JSON line
            f.write(json.dumps({"claim": item["claim"]}) + '\n')

print(f"Extraction complete. {output_path} created.")
print(f"Total claims extracted: {sum(1 for item in data if 'claim' in item)}") 