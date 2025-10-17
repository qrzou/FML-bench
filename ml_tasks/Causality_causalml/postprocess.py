import csv
import json
import re
import argparse
import os
import pandas as pd
import numpy as np

def parse_mean_stderr(value_str):
    """Parse 'mean ± stderr' format and return (mean, stderr)"""
    if pd.isna(value_str) or value_str == '':
        return np.nan, 0.0
    
    match = re.match(r'(\d+\.?\d*)\s*±\s*(\d+\.?\d*)', str(value_str))
    if match:
        mean = float(match.group(1))
        stderr = float(match.group(2))
        return mean, stderr
    else:
        # If no stderr format, assume stderr is 0
        try:
            return float(value_str), 0.0
        except (ValueError, TypeError):
            return np.nan, 0.0

def convert_causalml_results_to_json(results_dir, json_file):
    """Convert causalml results to final_info.json format"""
    result = {}
    
    # Define the datasets and their corresponding files (only test results)
    datasets = {
        "ihdp_test": {
            "file": "results_summary_test.csv",
            "model_name": "dragonnet"
        },
        "synthetic_test": {
            "file": "synthetic_test_results.csv",
            "model_name": "DragonNet"
        }
    }
    
    for dataset_name, config in datasets.items():
        file_path = os.path.join(results_dir, config["file"])
        model_name = config["model_name"]
        
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found, skipping {dataset_name}")
            continue
            
        # Read the CSV file
        df = pd.read_csv(file_path, index_col=0)
        
        # Get the model results (exclude Actuals/actual)
        model_results = df.loc[model_name] if model_name in df.index else df.iloc[0]
        
        means = {}
        stderrs = {}
        final_info_dict = {}
        
        # Process each metric
        for metric_name in model_results.index:
            if pd.isna(model_results[metric_name]):
                continue
                
            mean_val, stderr_val = parse_mean_stderr(model_results[metric_name])
            
            # Skip NaN values
            if np.isnan(mean_val):
                continue
            
            # Convert metric names to lowercase and handle special cases
            metric_key = metric_name.lower().replace(' ', '_').replace('%', 'pct')
            
            # Add to means and stderrs
            means[f"{metric_key}_mean"] = mean_val
            stderrs[f"{metric_key}_stderr"] = stderr_val
            
            # Add to final_info_dict (single value since we only have one measurement)
            final_info_dict[metric_key] = [mean_val]
        
        # Create the dataset entry
        result[dataset_name] = {
            "means": means,
            "stderrs": stderrs,
            "final_info_dict": final_info_dict
        }
    
    # Write to JSON file
    with open(json_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Conversion completed! Output saved to {json_file}")
    print(f"Processed datasets: {list(result.keys())}")

def main():
    parser = argparse.ArgumentParser(description="Convert causalml results to final_info.json format")
    parser.add_argument("--input_dir", "-i", type=str, default="results_tmp/",
                       help="Input directory containing causalml results (default: current directory)")
    parser.add_argument("--output", "-o", type=str, default="results_tmp/final_info.json",
                       help="Output JSON file path (default: results_tmp/final_info.json)")
    
    args = parser.parse_args()
    
    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist")
        return
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    convert_causalml_results_to_json(args.input_dir, args.output)

if __name__ == "__main__":
    main() 