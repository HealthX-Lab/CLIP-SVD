import pandas as pd
import glob
import argparse
import os
from scipy.stats import hmean

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_type",
    choices=["natural", "biomedical"],
    required=True,
    help="Choose which dataset list to use: natural or biomedical"
)
parser.add_argument(
    "--output_name",
    type=str,
    default="combined_results.csv",
    help="Name of the output CSV file"
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="outputs_base2new_lora_bottom_orth_rank0.98",
    help="Path to the directory containing input CSVs and where results will be saved"
)
args = parser.parse_args()

# Dataset lists
natural_datasets = [
    "eurosat", "fgvc_aircraft", "stanford_cars", "sun397", "ucf101",
    "oxford_flowers", "oxford_pets", "dtd", "caltech101", "food101", "imagenet"
]

biomedical_datasets = [
    "btmri", "covid", "ctkidney", "chmnist", 
    "kvasir", "retina", "octmnist", "kneexray", "lungcolon"
]

# Choose dataset list
allowed_datasets = natural_datasets if args.dataset_type == "natural" else biomedical_datasets

# Get all CSV files in the directory
csv_files = glob.glob(os.path.join(args.output_dir, "*.csv"))

# Create an empty DataFrame to store results
combined_df = pd.DataFrame()

for file in csv_files:
    dataset_name = os.path.basename(file).replace(".csv", "")
    if dataset_name not in allowed_datasets:
        continue
    
    df = pd.read_csv(file)
    df = df[['subsample', 'acc']]
    df = df.rename(columns={'acc': dataset_name})
    
    if combined_df.empty:
        combined_df = df
    else:
        combined_df = pd.merge(combined_df, df, on='subsample', how='outer')

# Compute HM row: harmonic mean of base and novel accuracy for each dataset
if not combined_df.empty:
    hm_row = combined_df.iloc[:, 1:].apply(lambda col: hmean(col[:2]), axis=0)
    hm_row = pd.Series(['HM'] + hm_row.tolist(), index=combined_df.columns)
    combined_df = pd.concat([combined_df, hm_row.to_frame().T], ignore_index=True)

    # Round all numeric columns to 2 decimal places
    numeric_cols = combined_df.columns[1:]
    combined_df[numeric_cols] = combined_df[numeric_cols].astype(float).round(2)

# Save to CSV
output_path = os.path.join(args.output_dir, args.output_name)
combined_df.to_csv(output_path, index=False)
print(f"Combined CSV saved as '{output_path}'")
