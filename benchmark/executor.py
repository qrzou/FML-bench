"""
BenchmarkExecutor: Handles subprocess management and experiment execution
"""
import json
import os
import os.path as osp
import shutil
import subprocess
import sys
from datetime import datetime
from subprocess import TimeoutExpired
from typing import Dict, List, Tuple, Optional
from benchmark.utils import get_changed_files
from .utils import get_filtered_results_for_prompt


class SubprocessResult:
    """Container for subprocess execution results"""
    def __init__(self, returncode: int, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class BenchmarkExecutor:
    """
    Executor: handles all subprocess and file operations
    """
    
    def __init__(self, config: dict, agent_name: str = None, benchmark_name: str = None, experiment_name: str = None, parent_timestamp: str = None, agent_params: dict = None, metrics_params: dict = None):
        self.config = config
        self.repo_dir = config["repo_dir"]
        self.conda_env = config.get("conda_env")
        self.agent_name = agent_name
        self.benchmark_name = benchmark_name
        self.experiment_name = experiment_name
        self.parent_timestamp = parent_timestamp
        self.workspace_dir = None
        self.agent_params = agent_params or {}  # Agent-specific parameters including timeouts
        self.metrics_params = metrics_params or {}
        
    def setup_workspace(self) -> str:
        """Setup experiment workspace"""
        # Create working directory with four-level structure: benchmark_results/agent_name/benchmark_name/parent_timestamp/experiment_name
        if not self.agent_name or not self.benchmark_name or not self.experiment_name:
            raise ValueError("agent_name, benchmark_name and experiment_name must be set before setup_workspace")
        
        if self.parent_timestamp:
            # New structure with parent timestamp folder
            self.workspace_dir = osp.join("benchmark_results", self.agent_name, self.benchmark_name, self.parent_timestamp, self.experiment_name)
        else:
            # Fallback to old structure for backward compatibility
            self.workspace_dir = osp.join("benchmark_results", self.agent_name, self.benchmark_name, self.experiment_name)
            
        assert not osp.exists(self.workspace_dir), f"Folder {self.workspace_dir} already exists."
        os.makedirs(self.workspace_dir, exist_ok=False)
        
        # Initialize git state: init the task's repo
        self._reset_git()

        # Prepare setup files if configured
        if self.config.get("prepare_setup_files"):
            print("Start to do preparing setup files...")
            setup_commands = self.config.get("setup_commands", [])
            
            if setup_commands:
                print(f"Running setup_commands: {setup_commands}")
                setup_result = self._run_commands(
                    setup_commands,
                    timeout=300,  # 5 minutes is enough for preparing setup files
                )
                if setup_result.returncode != 0:
                    print("="*60)
                    print("!!! WARNING: prepare setup files failed !!!")
                    print(f"Error details: {setup_result.stderr}")
                    print("="*60)
            else:
                print("No setup commands provided. Skip preparing setup files.")
        
        return self.workspace_dir
        
    def run_experiment(self, run_id: int) -> dict:
        """Execute a single experiment"""
        # 0. Clean up exp_running_dir if it exists
        exp_running_dir = self.config.get("exp_running_dir", "")
        if exp_running_dir:
            exp_running_dir_path = osp.join(self.repo_dir, exp_running_dir)
            if osp.exists(exp_running_dir_path):
                print(f"Warning: Removing existing experiment running directory: {exp_running_dir_path}")
                shutil.rmtree(exp_running_dir_path)
        
        # Create execution timestamp directory, Clean up if it exists
        run_dir = osp.join(self.workspace_dir, f"run_{run_id}")
        execution_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        execution_dir = osp.join(run_dir, f"execution_{execution_timestamp}")
        if osp.exists(execution_dir):
                print(f"Warning: Removing existing execution directory: {execution_dir}")
                shutil.rmtree(execution_dir)
        os.makedirs(execution_dir, exist_ok=True)
        
        # 1. Backup modified files
        self._backup_files(run_id, execution_timestamp)
        
        # 2. Run pre-processing if configured
        if self.config.get("do_preprocess"):
            preprocess_commands = self.config.get("preprocess_commands", [])
            if preprocess_commands:
                print("Running preprocessing commands...")
                preprocess_result = self._run_commands(
                    preprocess_commands, 
                    timeout=self.agent_params.get("preprocess_timeout", 300)  # Default 5 minutes
                )
                if preprocess_result.returncode != 0:
                    print("="*60)
                    print("!!! WARNING: Preprocess failed !!!")
                    print(f"Error details: {preprocess_result.stderr}")
                    print("="*60)
                    
                    # Save bug execution record for preprocess failure
                    self._save_bug_execution_record(run_id, execution_timestamp, "preprocess", preprocess_result, None)
                    
                    return {
                        "success": False,
                        "error": f"Preprocess failed: {preprocess_result.stderr}",
                        "returncode": preprocess_result.returncode
                    }
        
        # 3. Execute experiment commands
        result = self._run_commands(
            self.config.get("execute_commands", []),
            timeout=self.agent_params.get("execute_timeout", 7200)  # Default 2 hours
        )
        
        if result.returncode != 0:
            # Save bug execution record for main execution failure
            self._save_bug_execution_record(run_id, execution_timestamp, "execute", result, None)
            
            return {
                "success": False,
                "error": result.stderr,
                "returncode": result.returncode
            }
        
        # 4. Run post-processing if configured
        if self.config.get("do_postprocess"):
            postprocess_commands = self.config.get("postprocess_commands", [])
            if postprocess_commands:
                print("Running postprocessing commands...")
                postprocess_result = self._run_commands(
                    postprocess_commands,
                    timeout=self.agent_params.get("postprocess_timeout", 300)  # Default 5 minutes
                )
                if postprocess_result.returncode != 0:
                    print("="*60)
                    print("!!! WARNING: Postprocess failed !!!")
                    print(f"Error details: {postprocess_result.stderr}")
                    print("="*60)
                    
                    # Save bug execution record for postprocess failure
                    self._save_bug_execution_record(run_id, execution_timestamp, "postprocess", postprocess_result, None)
                    
                    return {
                        "success": False,
                        "error": f"Postprocess failed: {postprocess_result.stderr}",
                        "returncode": postprocess_result.returncode
                    }
        
        # 5. Collect results
        return self._collect_results(run_id, execution_timestamp)
        
    def cleanup(self):
        """Clean workspace and restore git state"""
        self._reset_git()
        
    def _run_commands(self, commands: List[str], timeout: int = None) -> SubprocessResult:
        """Execute command list (including conda environment handling)"""
        if not commands:
            raise ValueError("No commands provided. Please check your benchmark configuration.")
        
        if not self.conda_env:
            raise ValueError("No conda_env specified in config. Benchmark experiments must run in a conda environment.")
        
        # Use provided timeout (already set by caller with appropriate defaults)
        if timeout is None:
            timeout = 7200  # Default 2 hours if not specified
            
        cwd = osp.abspath(self.repo_dir)
        all_stderr = ""
        sum_returncode = 0
        
        # Process commands with conda environment
        for cmd in commands:
            # Split command and prepend conda run
            cmd_parts = cmd.split()
            full_cmd = ["conda", "run", "--no-capture-output", "-n", self.conda_env] + cmd_parts
                
            try:
                print(f"Running command: {' '.join(full_cmd)}")
                print(f"Current working directory: {cwd}")
                
                result = subprocess.run(
                    full_cmd,
                    cwd=cwd,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=timeout
                )
                
                all_stderr += result.stderr
                sum_returncode += abs(result.returncode)
                
                if result.returncode != 0:
                    print(f"Command failed with return code {result.returncode}")
                    print(f"Error: {result.stderr}")
                    break
                    
            except TimeoutExpired:
                print(f"Command timed out after {timeout} seconds")
                return SubprocessResult(1, stderr=f"Timeout after {timeout} seconds")
            except Exception as e:
                print(f"Error running command: {e}")
                return SubprocessResult(1, stderr=str(e))
                
        return SubprocessResult(sum_returncode, stderr=all_stderr)
        
    def _reset_git(self):
        """Reset git repository to clean state"""
        try:
            cwd = osp.abspath(self.repo_dir)
            subprocess.run(["git", "reset", "--hard"], cwd=cwd, check=True)
            subprocess.run(["git", "clean", "-fd"], cwd=cwd, check=True)
            print("Git repository reset to clean state")
        except Exception as e:
            print(f"Warning: Failed to reset git: {e}")
            
    def _backup_files(self, run_id: int, execution_timestamp: str):
        """Backup currently modified files"""
        try:
            # Construct execution directory path
            run_dir = osp.join(self.workspace_dir, f"run_{run_id}")
            execution_dir = osp.join(run_dir, f"execution_{execution_timestamp}")
            
            # Get exclude paths from config
            exclude_paths = self.config.get("backup_excluded_files")
            
            # Use the existing get_changed_files function from utils
            # Default behavior: get all changed files (not just M and A)
            changed_files = get_changed_files(
                repo_path=osp.abspath(self.repo_dir),
                exclude_paths=exclude_paths
            )
            print(f"Changed files: {changed_files}")
            
            if changed_files:
                # Create backup directory
                backup_dir = osp.join(execution_dir, "code_backup")
                os.makedirs(backup_dir, exist_ok=True)
                print(f"Backing up changed files to {backup_dir}")
                
                # Backup each changed file
                for changed_f in changed_files:  # changed_f should be relative dir to repo_dir
                    # Solve different path format
                    if osp.isabs(changed_f):  # absolute path
                        src_path = changed_f
                        rel_changed_f = osp.relpath(changed_f, start=osp.abspath(self.repo_dir))
                        dest_path = osp.join(backup_dir, rel_changed_f)
                    else:  # relative path
                        src_path = osp.join(self.repo_dir, changed_f)
                        dest_path = osp.join(backup_dir, changed_f)
                    
                    print(f"Backing up {src_path} to {dest_path}")
                    
                    if osp.isfile(src_path):
                        os.makedirs(osp.dirname(dest_path), exist_ok=True)
                        shutil.copy2(src_path, dest_path)
                    elif osp.isdir(src_path):
                        shutil.copytree(src_path, dest_path)
                    else:
                        print(f"Warning: {src_path} is not a file or directory")
                        
        except Exception as e:
            print(f"Warning: Failed to backup files: {e}")
            
    def _collect_results(self, run_id: int, execution_timestamp: str) -> dict:
        """Collect experiment results"""
        # Get run directory (already created in run_experiment)
        run_dir = osp.join(self.workspace_dir, f"run_{run_id}")
        
        # Construct execution directory path
        execution_dir = osp.join(run_dir, f"execution_{execution_timestamp}")
        
        # Determine where to look for final_info.json
        exp_running_dir = self.config.get("exp_running_dir", "")
        if exp_running_dir:
            # If exp_running_dir is specified, final_info.json must be there
            exp_running_dir_path = osp.join(self.repo_dir, exp_running_dir)
            final_info_path = osp.join(exp_running_dir_path, "final_info.json")
            
            if not osp.exists(final_info_path):
                print(f"Error: final_info.json not found in exp_running_dir: {exp_running_dir_path}")
                print("Postprocessing should have created this file.")
                return {"success": False, "error": f"final_info.json not found in {exp_running_dir_path}"}
        else:
            print("Error: exp_running_dir not specified in config")
            return {"success": False, "error": "exp_running_dir not specified in config"}

        # Copy final_info.json to run directory
        dst = osp.join(execution_dir, "final_info.json")
        shutil.copy(final_info_path, dst)
        
        # Load and return results
        with open(dst, 'r') as f:
            results = json.load(f)
            
        # Extract means if present
        if isinstance(results, dict):
            results = {k: v.get("means", v) if isinstance(v, dict) else v 
                      for k, v in results.items()}
        
        # Apply metrics filtering for prompt generation while preserving original results
        filtered_results = get_filtered_results_for_prompt(results, self.metrics_params or {})
                      
        return {
            "success": True, 
            "results": results,  # Keep original results
            "filtered_results": filtered_results  # Add filtered results for prompts
        }
        
    def _save_bug_execution_record(self, run_id: int, execution_timestamp: str, phase: str, result: SubprocessResult, additional_context: dict = None):
        """
        Save detailed bug execution record including code snapshots and error information
        
        Args:
            run_id: The run ID that failed
            phase: The phase that failed (preprocess, execute, postprocess)
            result: The SubprocessResult containing error information
            execution_timestamp: The execution timestamp to construct the directory path
            additional_context: Additional context information to save
        """
        try:
            # Construct execution directory path
            run_dir = osp.join(self.workspace_dir, f"run_{run_id}")
            execution_dir = osp.join(run_dir, f"execution_{execution_timestamp}")
            
            # Use the constructed execution directory
            records_dir = osp.join(execution_dir, "records")
            os.makedirs(records_dir, exist_ok=True)
            
            # Create bug record filename with phase and timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            bug_filename = f"bug_{phase}_{timestamp}.json"
            bug_filepath = osp.join(records_dir, bug_filename)
            
            # Create comprehensive bug record
            bug_record = {
                "run_id": run_id,
                "phase": phase,
                "timestamp": timestamp,
                
                # Error information
                "error": {
                    "returncode": result.returncode,
                    "stderr": result.stderr
                },

                "additional_context": additional_context or {}
            }
            
            # Save bug record
            with open(bug_filepath, 'w') as f:
                json.dump(bug_record, f, indent=2)
            
            print(f"Bug execution record saved to: {bug_filepath}")
            
        except Exception as e:
            print(f"Warning: Failed to save bug execution record: {e}")
        
    