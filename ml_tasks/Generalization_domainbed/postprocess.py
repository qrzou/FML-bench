#!/usr/bin/env python3
"""
Postprocess script to extract final env2_in_acc from results.jsonl
and save it in the required format for final_info.json
"""

import json
import argparse
import os

def extract_final_result(directory="./baseline_output/"):
    """Extract the last line's env2_in_acc from results.jsonl"""
    
    # Ensure directory ends with /
    if not directory.endswith('/'):
        directory += '/'
    
    results_file = os.path.join(directory, "results.jsonl")
    
    # Read the last line from results.jsonl
    with open(results_file, 'r') as f:
        lines = f.readlines()
        
        # Parse the last line
        last_line = lines[-1].strip()
        if not last_line:
            # If last line is empty, try the second to last
            last_line = lines[-2].strip()
        
        last_result = json.loads(last_line)
    
    # Extract env2_in_acc
    env2_in_acc = last_result['env2_in_acc']
    
    # Create the output structure
    final_info = {
        "ColoredMNIST_test_env2": {
            "means": {
                "in_acc_mean": env2_in_acc,
            },
            "stderrs": {
                "in_acc_stderr": 0.0,
            },
            "final_info_dict": {
                "in_acc": [
                    env2_in_acc
                ],
            }
        }
    }
    
    return final_info

def main():
    """Main function to process results and save final_info.json"""
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Extract final env2_in_acc from results.jsonl')
    parser.add_argument('--directory', type=str, default='./baseline_output/',
                        help='Directory containing results.jsonl (default: ./baseline_output/)')
    
    args = parser.parse_args()
    
    # Extract the final result
    final_info = extract_final_result(args.directory)
    
    # Save to final_info.json in the specified directory
    output_file = os.path.join(args.directory, "final_info.json")
    with open(output_file, 'w') as f:
        json.dump(final_info, f, indent=2)
    
    print(f"Successfully extracted final result from {args.directory}:")
    print(f"ENV2 In-domain Accuracy: {final_info['ColoredMNIST_test_env2']['means']['in_acc_mean']}")
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()