"""
Utility functions for benchmark operations
"""
import re

import os
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Union, Set, Optional


def get_changed_files(
    repo_path: str, 
    exclude_paths: Optional[List[str]] = None,
    change_types: Optional[List[str]] = None
) -> List[str]:
    """
    Get files that have changed in a git repository compared to the last commit
    
    Args:
        repo_path: Path to the git repository
        exclude_paths: List of file or folder paths to exclude, supports relative and absolute paths
        change_types: List of change types to include. Options: 'M' (modified), 'A' (added), 
                     'D' (deleted), 'R' (renamed), 'C' (copied), '?' (untracked), etc.
                     If None, includes all change types.
    
    Returns:
        List of relative paths of changed files (relative to repository root)
        
    Raises:
        ValueError: If path is not a valid git repository
        subprocess.CalledProcessError: If git command execution fails
    """
    if exclude_paths is None:
        exclude_paths = []
    
    if change_types is None:
        change_types = ['M', 'A', 'D', 'R', 'C', '?', '!']  # Include all types by default
    
    # Convert to absolute path and check if it's a git repository
    repo_path = os.path.abspath(repo_path)
    if not os.path.exists(os.path.join(repo_path, '.git')):
        raise ValueError(f"Path {repo_path} is not a valid git repository")
    
    try:
        # Use git status --porcelain to get all changed files
        # --porcelain format: XY filename
        # X represents staging area status, Y represents working directory status
        result = subprocess.run(
            ['git', 'status', '--porcelain'],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )

        # Debug
        # print(result.stdout)
        
        changed_files = []
        for line in result.stdout.splitlines():
            if not line:
                continue
                
            # Parse git status output format
            # Format: XY filename where X=staging, Y=working dir, followed by space and filename
            if len(line) < 3:
                continue
                
            status = line[:2]
            filename = line[3:]  # Skip status code and space

            # Debug
            # print(line)
            # print(status, filename)
            
            # Check if this change type should be included
            should_include = False
            for change_type in change_types:
                if change_type in status:
                    should_include = True
                    break
            
            if not should_include:
                continue
            
            # Handle rename cases (R -> "old_name -> new_name")
            if ' -> ' in filename:
                filename = filename.split(' -> ')[1]
            
            # Remove quotes (if filename contains special characters)
            if filename.startswith('"') and filename.endswith('"'):
                filename = filename[1:-1]
            
            changed_files.append(filename)
        
        # Filter excluded paths
        filtered_files = []
        for file_path in changed_files:
            if not _should_exclude_file(file_path, exclude_paths, repo_path):
                filtered_files.append(file_path)
        
        return filtered_files
        
    except subprocess.CalledProcessError as e:
        raise subprocess.CalledProcessError(
            e.returncode, 
            e.cmd, 
            f"Git command execution failed: {e.stderr}"
        )


def _should_exclude_file(file_path: str, exclude_paths: List[str], repo_path: str) -> bool:
    """
    Check if a file should be excluded
    
    Args:
        file_path: File path to check (relative to repository root)
        exclude_paths: List of paths to exclude
        repo_path: Absolute path of repository root
    
    Returns:
        True if file should be excluded, False otherwise
    """
    if not exclude_paths:
        return False
    
    # Convert file path to absolute path
    abs_file_path = os.path.abspath(os.path.join(repo_path, file_path))
    
    for exclude_path in exclude_paths:
        # Handle exclude paths, support both relative and absolute paths
        if os.path.isabs(exclude_path):
            abs_exclude_path = os.path.abspath(exclude_path)
        else:
            abs_exclude_path = os.path.abspath(os.path.join(repo_path, exclude_path))
        
        # Check for exact match (file)
        if abs_file_path == abs_exclude_path:
            return True
        
        # Check if file is within excluded folder
        try:
            # Use Path.relative_to to check if it's a subpath
            Path(abs_file_path).relative_to(abs_exclude_path)
            return True
        except ValueError:
            # If not a subpath, relative_to will raise ValueError
            continue
    
    return False



def extract_token_usage_from_aider(usage_report: str) -> Dict[str, Any]:
    """
    Extract token usage information from Aider's response output.

    Args:
        usage_report: The usage_report string from Aider's coder
        
    Returns:
        Dictionary with token usage info
    """
    # Matches both whole numbers and decimals with optional 'k'
    token_pattern = r"Tokens: (\d+(?:\.\d+)?k?) sent, (\d+(?:\.\d+)?k?) received"
    cost_pattern = r"Cost: \$([\d.]+) message, \$([\d.]+) session"

    usage_info = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "message_cost": 0.0,
        "session_cost": 0.0,
    }

    def _parse_token_str(s: str) -> int:
        """Convert strings like '5.5k' or '132' into integer tokens."""
        if s.endswith("k"):
            return int(float(s[:-1]) * 1000)
        return int(float(s))

    token_match = re.search(token_pattern, usage_report)
    if token_match:
        sent_str, received_str = token_match.groups()
        sent = _parse_token_str(sent_str)
        received = _parse_token_str(received_str)
        usage_info.update({
            "prompt_tokens": sent,
            "completion_tokens": received,
            "total_tokens": sent + received
        })

    cost_match = re.search(cost_pattern, usage_report)
    if cost_match:
        message_cost = float(cost_match.group(1))
        session_cost = float(cost_match.group(2))
        usage_info.update({
            "message_cost": message_cost,
            "session_cost": session_cost
        })
    
    return usage_info


def filter_metrics_for_prompt(results: Dict[str, Any], include_datasets: List[str] = None, include_metrics: List[str] = None) -> Dict[str, Any]:
    """
    Filter results by both datasets and metrics for prompt generation while preserving original structure.
    
    Args:
        results: The complete results dictionary (e.g., from final_info.json)
        include_datasets: List of dataset names to include. If None or empty, includes all datasets.
        include_metrics: List of metric names to include. If None or empty, includes all metrics.
    
    Returns:
        Filtered results dictionary with only the specified datasets and metrics
    """
    if not include_datasets and not include_metrics:
        return results
    
    # Handle nested structure like in causalml results
    if isinstance(results, dict):
        filtered_results = {}
        
        for dataset_name, dataset_data in results.items():
            # Skip dataset if not in include_datasets (when specified)
            if include_datasets and dataset_name not in include_datasets:
                continue
                
            if isinstance(dataset_data, dict) and "means" in dataset_data:
                # Handle structure like {"ihdp_test": {"means": {...}, "stderrs": {...}}}
                if include_metrics:
                    # Filter metrics within the dataset
                    filtered_means = {}
                    for metric_name in include_metrics:
                        if metric_name in dataset_data["means"]:
                            filtered_means[metric_name] = dataset_data["means"][metric_name]
                        else:
                            print(f"Warning: Requested metric '{metric_name}' not found in dataset '{dataset_name}'. Available metrics: {list(dataset_data['means'].keys())}")
                    
                    # Only include dataset if it has any of the requested metrics
                    if filtered_means:
                        filtered_results[dataset_name] = {
                            "means": filtered_means,
                            "stderrs": {k: v for k, v in dataset_data.get("stderrs", {}).items() 
                                       if k.replace("_stderr", "_mean") in filtered_means}
                        }
                else:
                    # Include all metrics for this dataset
                    filtered_results[dataset_name] = dataset_data
                    
            elif isinstance(dataset_data, dict):
                # Handle flat structure like {"ate_mean": 0.5, "mae_mean": 0.1}
                if include_metrics:
                    filtered_metrics = {}
                    for metric_name in include_metrics:
                        if metric_name in dataset_data:
                            filtered_metrics[metric_name] = dataset_data[metric_name]
                        else:
                            print(f"Warning: Requested metric '{metric_name}' not found in dataset '{dataset_name}'. Available metrics: {list(dataset_data.keys())}")
                    
                    if filtered_metrics:
                        filtered_results[dataset_name] = filtered_metrics
                else:
                    filtered_results[dataset_name] = dataset_data
            else:
                # Handle simple key-value pairs
                if include_metrics and dataset_name in include_metrics:
                    filtered_results[dataset_name] = dataset_data
                elif not include_metrics:
                    filtered_results[dataset_name] = dataset_data
        
        return filtered_results
    
    return results


def get_available_metrics(results: Dict[str, Any]) -> List[str]:
    """
    Extract all available metric names from results for debugging/validation.
    
    Args:
        results: The results dictionary
    
    Returns:
        List of available metric names
    """
    metrics = set()
    
    if isinstance(results, dict):
        for key, value in results.items():
            if isinstance(value, dict) and "means" in value:
                # Handle nested structure
                metrics.update(value["means"].keys())
            elif isinstance(value, dict):
                # Handle flat structure
                metrics.update(value.keys())
            else:
                # Handle simple key-value
                metrics.add(key)
    
    return sorted(list(metrics)) 


def get_filtered_results_for_prompt(results: Dict[str, Any], metrics_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Shared helper to filter results for prompt generation using metrics configuration.

    Args:
        results: The complete results dictionary (e.g., from final_info.json)
        metrics_params: Dictionary that can contain keys like:
            - include_datasets: List[str]
            - include_metrics: List[str]

    Returns:
        Filtered results dictionary according to metrics_params. If no filtering
        is specified, returns results unchanged.
    """
    if not isinstance(metrics_params, dict) or not metrics_params:
        return results

    include_datasets = metrics_params.get("include_datasets", [])
    include_metrics = metrics_params.get("include_metrics", [])

    # Log available metrics and datasets for debugging
    available_metrics = get_available_metrics(results)
    available_datasets = list(results.keys()) if isinstance(results, dict) else []
    print(f"Available datasets in results: {available_datasets}")
    print(f"Available metrics in results: {available_metrics}")

    if include_datasets or include_metrics:
        print(f"Filtering datasets to: {include_datasets if include_datasets else 'all'}")
        print(f"Filtering metrics to: {include_metrics if include_metrics else 'all'}")
        filtered = filter_metrics_for_prompt(results, include_datasets, include_metrics)
        print(f"Filtered results keys: {list(filtered.keys())}")
        return filtered

    print("No filtering applied - using all available datasets and metrics")
    return results


# Example usage of get_changed_files
if __name__ == "__main__":
    # Example: Get changed files in current directory, excluding certain paths
    try:
        repo_path = "."  # Current directory
        exclude_paths = [
            "__pycache__",      # Exclude all __pycache__ folders
            "*.pyc",            # Note: This implementation doesn't support wildcards, needs exact paths
            "logs/",            # Exclude logs folder
            "temp.txt"          # Exclude specific file
        ]
        
        # Get all changed files
        all_changed_files = get_changed_files(repo_path, exclude_paths)
        print("All changed files:")
        for file in all_changed_files:
            print(f"  {file}")
            
        # Get only modified files (excluding new, deleted, renamed files)
        modified_files = get_changed_files(repo_path, exclude_paths, ['M'])
        print("\nModified files only:")
        for file in modified_files:
            print(f"  {file}")
            
        # Get only new files
        new_files = get_changed_files(repo_path, exclude_paths, ['A', '?'])
        print("\nNew files only:")
        for file in new_files:
            print(f"  {file}")
            
    except Exception as e:
        print(f"Error: {e}")