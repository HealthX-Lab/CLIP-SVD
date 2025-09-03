import pandas as pd
import glob
import argparse
import os

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
    default="outputs_fewshot_medical_reproduce",
    help="Path to the directory containing input CSVs and where results will be saved"
)
args = parser.parse_args()

# Dataset lists
natural_datasets = [
    "eurosat", "fgvc_aircraft", "stanford_cars", "sun397", "ucf101",
    "oxford_flowers", "oxford_pets", "dtd", "caltech101", "food101", "imagenet"
]

biomedical_datasets = [
    "busi", "btmri", "covid", "ctkidney", "chmnist", 
    "kvasir", "retina", "octmnist", "kneexray", "lungcolon"
]

# Choose dataset list
allowed_datasets = natural_datasets if args.dataset_type == "natural" else biomedical_datasets

# Get all CSV files in the directory
csv_files = glob.glob(os.path.join(args.output_dir, "*.csv"))

# Create an empty DataFrame to store results
combined_df = pd.DataFrame()

for file in csv_files:
    dataset_name = file.split("/")[-1].replace(".csv", "")
    if dataset_name not in allowed_datasets:
        continue
    
    df = pd.read_csv(file)
    df = df[['num_shots', 'acc']]
    df = df.rename(columns={'acc': dataset_name})
    
    if combined_df.empty:
        combined_df = df
    else:
        combined_df = pd.merge(combined_df, df, on='num_shots', how='outer')

# Print stats
print("{:.2f}".format(combined_df.iloc[:, 1:].mean(axis=1)[0]))
print(f"Number of combined datasets: {len(combined_df.columns) - 1}")

# Save to CSV
output_path = os.path.join(args.output_dir, args.output_name)
combined_df.to_csv(output_path, index=False)
print(f"Combined CSV saved as '{output_path}'")
