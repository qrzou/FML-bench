"""
BenchmarkExecutor: Handles subprocess management and experiment execution.

Provides run_val() and run_test() for validation and test phases,
with process-group-based subprocess kill for reliable timeout handling.
"""
import json
import os
import os.path as osp
import shutil
import signal
import subprocess
from datetime import datetime
from typing import Optional

from benchmark.utils import get_changed_files


class SubprocessResult:
    """Container for subprocess execution results"""
    def __init__(self, returncode: int, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


# Fixed output path constants (relative to repo_dir)
VAL_OUTPUT = "results_tmp/val_info.json"
TEST_OUTPUT = "results_tmp/test_info.json"


class BenchmarkExecutor:
    """
    Executor: handles all subprocess and file operations for benchmark experiments.

    Key differences from v1:
    - run_val() / run_test() replace run_experiment()
    - Single command string (val_command / test_command) instead of command lists
    - Fixed output paths (results_tmp/val_info.json, results_tmp/test_info.json)
    - Process-group-based subprocess kill for reliable timeout handling
    - Primary metric extraction from config["metric"]
    """

    def __init__(self, config: dict, agent_name: str = None, benchmark_name: str = None,
                 experiment_name: str = None, parent_timestamp: str = None,
                 timeout: int = 2400, output_dir: str = "benchmark_results"):
        self.config = config
        self.repo_dir = config["repo_dir"]
        self.conda_env = config["conda_env"]
        self.timeout = timeout
        self.agent_name = agent_name
        self.benchmark_name = benchmark_name
        self.experiment_name = experiment_name
        self.parent_timestamp = parent_timestamp
        self.output_dir = output_dir
        self.workspace_dir = None
        self._current_proc = None  # Track running subprocess for cleanup on external kill

    def setup_workspace(self) -> str:
        """
        Setup experiment workspace directory and reset git state.

        Directory structure:
            benchmark_results/{agent_name}/{benchmark_name}/{parent_timestamp}/{experiment_name}

        Returns:
            Path to the created workspace directory.
        """
        if not self.agent_name or not self.benchmark_name or not self.experiment_name:
            raise ValueError("agent_name, benchmark_name and experiment_name must be set before setup_workspace")

        if self.parent_timestamp:
            # Four-level structure with parent timestamp
            self.workspace_dir = osp.join(
                self.output_dir, self.agent_name, self.benchmark_name,
                self.parent_timestamp, self.experiment_name
            )
        else:
            # Fallback to three-level structure for backward compatibility
            self.workspace_dir = osp.join(
                self.output_dir, self.agent_name, self.benchmark_name,
                self.experiment_name
            )

        assert not osp.exists(self.workspace_dir), f"Folder {self.workspace_dir} already exists."
        os.makedirs(self.workspace_dir, exist_ok=False)

        # Reset git state in the task repo
        self._reset_git()

        return self.workspace_dir

    def run_val(self, run_id: int) -> dict:
        """
        Execute a validation run.

        Steps:
            1. Clean results_tmp/ in repo_dir if it exists
            2. Create run_{run_id}/execution_{timestamp}/ directories
            3. Backup modified files
            4. Run val_command via _run_command()
            5. Collect results from results_tmp/val_info.json
            6. Extract primary_metric

        Returns:
            dict with keys: success, results, primary_metric, error
        """
        return self._run_phase(run_id, "val_command", VAL_OUTPUT)

    def run_test(self, run_id: int) -> dict:
        """
        Execute a test run.

        Same flow as run_val but uses config["test_command"] and TEST_OUTPUT.
        No file backup is performed for test runs.

        Returns:
            dict with keys: success, results, primary_metric, error
        """
        return self._run_phase(run_id, "test_command", TEST_OUTPUT, backup=False)

    def _run_phase(self, run_id: int, command_key: str, output_path: str,
                   backup: bool = True) -> dict:
        """
        Shared implementation for run_val and run_test.

        Args:
            run_id: Identifier for this run.
            command_key: Config key for the command string ("val_command" or "test_command").
            output_path: Relative path under repo_dir where results JSON is written.
            backup: Whether to backup modified files before running.

        Returns:
            Structured result dict.
        """
        # 0. Get command from config — support both new and old formats
        command = self.config.get(command_key)
        if not command:
            # Backward compat: old config has execute_commands instead of val_command/test_command
            if "execute_commands" in self.config:
                command = self.config["execute_commands"][0]
            else:
                return {
                    "success": False,
                    "results": None,
                    "primary_metric": None,
                    "error": f"No '{command_key}' or 'execute_commands' specified in config"
                }

        # 1. Clean results_tmp/ if it exists
        results_tmp_dir = osp.join(osp.abspath(self.repo_dir), "results_tmp")
        if osp.exists(results_tmp_dir):
            print(f"Cleaning existing results_tmp/ directory: {results_tmp_dir}")
            shutil.rmtree(results_tmp_dir)

        # 2. Create run and execution directories
        execution_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = osp.join(self.workspace_dir, f"run_{run_id}")
        execution_dir = osp.join(run_dir, f"execution_{execution_timestamp}")
        if osp.exists(execution_dir):
            print(f"Warning: Removing existing execution directory: {execution_dir}")
            shutil.rmtree(execution_dir)
        os.makedirs(execution_dir, exist_ok=True)

        # 3. Backup modified files (val only)
        if backup:
            self._backup_files(run_id, execution_timestamp)

        # 3.5 Run preprocess commands if old-format config has them
        if self.config.get("do_preprocess") and self.config.get("preprocess_commands"):
            for pre_cmd in self.config["preprocess_commands"]:
                print(f"Running preprocess: {pre_cmd}")
                pre_result = self._run_command(pre_cmd)
                if pre_result.returncode != 0:
                    print(f"Warning: Preprocess failed: {pre_result.stderr}")

        # 4. Run the command (with .git protection)
        print(f"Running {command_key}: {command}")
        git_protected = self._protect_git_dir()
        try:
            result = self._run_command(command)
        finally:
            if git_protected:
                self._unprotect_git_dir()

        # 4.1 Verify workspace integrity after execution
        integrity_error = self._check_workspace_integrity()
        if integrity_error:
            print(f"ERROR: {integrity_error}")
            return {
                "success": False,
                "results": None,
                "primary_metric": None,
                "error": f"Workspace corrupted: {integrity_error}"
            }

        if result.returncode != 0:
            # Save bug execution record
            phase = "val" if command_key == "val_command" else "test"
            self._save_bug_execution_record(run_id, execution_timestamp, phase, result)

            return {
                "success": False,
                "results": None,
                "primary_metric": None,
                "error": result.stderr
            }

        # 4.5 Run postprocess commands if old-format config has them
        if self.config.get("do_postprocess") and self.config.get("postprocess_commands"):
            for post_cmd in self.config["postprocess_commands"]:
                print(f"Running postprocess: {post_cmd}")
                post_result = self._run_command(post_cmd)
                if post_result.returncode != 0:
                    print(f"Warning: Postprocess failed: {post_result.stderr}")

        # 5. Collect results — try specified output_path, fallback to final_info.json for old configs
        actual_output_path = output_path
        source = osp.join(osp.abspath(self.repo_dir), output_path)
        if not osp.exists(source):
            # Fallback: old configs produce final_info.json in exp_running_dir
            exp_dir = self.config.get("exp_running_dir", "results_tmp/")
            fallback = osp.join(exp_dir, "final_info.json")
            fallback_full = osp.join(osp.abspath(self.repo_dir), fallback)
            if osp.exists(fallback_full):
                actual_output_path = fallback

        return self._collect_results(run_id, execution_timestamp, actual_output_path)

    def cleanup(self):
        """Clean workspace and restore git state."""
        self._reset_git()

    def _run_command(self, command: str) -> SubprocessResult:
        """
        Execute a single command string inside the configured conda environment.

        Uses process group (os.setpgrp) so that on timeout the entire process tree
        (including grandchildren spawned by conda/bash) is killed reliably.

        Args:
            command: The command string to execute (passed as-is to bash -c inside the conda env).

        Returns:
            SubprocessResult with returncode, stdout, and stderr.
        """
        full_cmd = ["conda", "run", "--no-capture-output", "-n", self.conda_env, "bash", "-c", command]

        try:
            print(f"Running command: {' '.join(full_cmd)}")
            print(f"Current working directory: {osp.abspath(self.repo_dir)}")

            proc = subprocess.Popen(
                full_cmd,
                cwd=osp.abspath(self.repo_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                preexec_fn=os.setpgrp,  # Create new process group
            )
            self._current_proc = proc
            stdout, stderr = proc.communicate(timeout=self.timeout)
            self._current_proc = None
            return SubprocessResult(proc.returncode, stdout=stdout, stderr=stderr)

        except subprocess.TimeoutExpired:
            print(f"Command timed out after {self.timeout} seconds")
            self._current_proc = None
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)  # Kill entire group
            proc.wait()
            return SubprocessResult(1, stderr=f"Timeout after {self.timeout} seconds")

        except Exception as e:
            self._current_proc = None
            print(f"Error running command: {e}")
            return SubprocessResult(1, stderr=str(e))

    def kill_running_process(self):
        """Kill the currently running subprocess and its entire process group.

        Called by signal handlers when the parent process is killed externally.
        Idempotent: safe to call multiple times or when no process is running.
        """
        proc = self._current_proc
        if proc is None:
            return
        self._current_proc = None
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGKILL)
            proc.wait()
        except (ProcessLookupError, OSError):
            pass

    def _collect_results(self, run_id: int, execution_timestamp: str,
                         output_path: str) -> dict:
        """
        Collect experiment results from the output JSON file.

        Args:
            run_id: The run identifier.
            execution_timestamp: Timestamp string for the execution directory.
            output_path: Relative path under repo_dir to the results JSON
                         (e.g. "results_tmp/val_info.json").

        Returns:
            Structured dict with success, results, primary_metric, error.
        """
        # Construct paths
        run_dir = osp.join(self.workspace_dir, f"run_{run_id}")
        execution_dir = osp.join(run_dir, f"execution_{execution_timestamp}")
        source_path = osp.join(osp.abspath(self.repo_dir), output_path)

        if not osp.exists(source_path):
            error_msg = f"Results file not found: {source_path}"
            print(f"Error: {error_msg}")
            return {
                "success": False,
                "results": None,
                "primary_metric": None,
                "error": error_msg
            }

        # Determine destination filename (val_info.json or test_info.json)
        dest_filename = osp.basename(output_path)
        dest_path = osp.join(execution_dir, dest_filename)
        shutil.copy(source_path, dest_path)

        # Load results
        with open(dest_path, 'r') as f:
            results = json.load(f)

        # Extract primary metric and filtered results
        primary_metric = self._extract_primary_metric(results)
        filtered_results = self._filter_results(results)

        return {
            "success": True,
            "results": results,
            "filtered_results": filtered_results,
            "primary_metric": primary_metric,
            "error": None
        }

    def _extract_primary_metric(self, results: dict) -> Optional[float]:
        """
        Extract the primary metric value from results.

        Handles two result structures:
            1. Nested: {"dataset_name": {"means": {"metric_name": value, ...}}, ...}
               -> Averages the metric across all datasets.
            2. Flat: {"metric_name": value, ...}

        The metric name is read from config["metric"].

        Returns:
            The extracted metric value as float, or None if not found.
        """
        metric_name = self.config.get("metric")
        if not metric_name:
            print("Warning: No 'metric' specified in config, cannot extract primary metric")
            return None

        if not isinstance(results, dict):
            print(f"Warning: Results is not a dict, cannot extract metric '{metric_name}'")
            return None

        # Filter datasets if include_datasets is specified in config
        include_datasets = self.config.get("include_datasets")

        # Try nested format first: {"dataset": {"means": {"metric": value}}}
        nested_values = []
        has_nested = False
        for key, value in results.items():
            if isinstance(value, dict) and "means" in value:
                has_nested = True
                # Skip datasets not in include_datasets filter
                if include_datasets and key not in include_datasets:
                    continue
                means = value["means"]
                if isinstance(means, dict) and metric_name in means:
                    try:
                        nested_values.append(float(means[metric_name]))
                    except (ValueError, TypeError):
                        pass

        if has_nested and nested_values:
            # Average across filtered datasets that have this metric
            avg_value = sum(nested_values) / len(nested_values)
            return avg_value

        # Fall back to flat format: {"metric_name": value}
        if metric_name in results:
            try:
                return float(results[metric_name])
            except (ValueError, TypeError):
                pass

        print(f"Warning: Metric '{metric_name}' not found in results")
        return None

    def _filter_results(self, results: dict) -> dict:
        """Filter results to only include datasets specified in config['include_datasets'].

        Returns filtered copy if include_datasets is set, otherwise returns results as-is.
        """
        include_datasets = self.config.get("include_datasets")
        if not include_datasets or not isinstance(results, dict):
            return results
        return {k: v for k, v in results.items() if k in include_datasets}

    def _backup_files(self, run_id: int, execution_timestamp: str):
        """
        Backup currently modified files to the execution directory.

        Uses get_changed_files() from benchmark.utils to detect changes
        relative to the last git commit.

        Args:
            run_id: The run identifier.
            execution_timestamp: Timestamp string for the execution directory.
        """
        try:
            # Construct execution directory path
            run_dir = osp.join(self.workspace_dir, f"run_{run_id}")
            execution_dir = osp.join(run_dir, f"execution_{execution_timestamp}")

            # Get exclude paths from config
            exclude_paths = self.config.get("backup_excluded_files")

            # Use get_changed_files from benchmark.utils
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
                for changed_f in changed_files:
                    # Handle both absolute and relative paths
                    if osp.isabs(changed_f):
                        src_path = changed_f
                        rel_changed_f = osp.relpath(changed_f, start=osp.abspath(self.repo_dir))
                        dest_path = osp.join(backup_dir, rel_changed_f)
                    else:
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

    def _protect_git_dir(self) -> bool:
        """Make .git directory non-writable to prevent corruption during execution."""
        git_dir = osp.join(osp.abspath(self.repo_dir), ".git")
        if osp.isdir(git_dir):
            try:
                os.chmod(git_dir, 0o555)
                return True
            except OSError as e:
                print(f"Warning: Failed to protect .git: {e}")
        return False

    def _unprotect_git_dir(self):
        """Restore .git directory write permissions."""
        git_dir = osp.join(osp.abspath(self.repo_dir), ".git")
        if osp.isdir(git_dir):
            try:
                os.chmod(git_dir, 0o755)
            except OSError:
                pass

    def _check_workspace_integrity(self) -> Optional[str]:
        """Verify critical workspace files still exist after execution.

        Returns error message if integrity violated, None if OK.
        """
        repo_abs = osp.abspath(self.repo_dir)

        # Check .git exists
        git_dir = osp.join(repo_abs, ".git")
        if not osp.isdir(git_dir):
            return f"CRITICAL: .git directory deleted: {git_dir}"

        # Check target_files exist
        target_files = self.config.get("target_files", [])
        for tf in target_files:
            tf_abs = osp.join(repo_abs, tf) if not osp.isabs(tf) else tf
            if not osp.exists(tf_abs):
                return f"CRITICAL: target file deleted: {tf_abs}"

        return None

    def _reset_git(self):
        """Reset git repository to clean state."""
        try:
            cwd = osp.abspath(self.repo_dir)
            subprocess.run(["git", "reset", "--hard"], cwd=cwd, check=True)
            subprocess.run(["git", "clean", "-fd"], cwd=cwd, check=True)
            print("Git repository reset to clean state")
        except Exception as e:
            print(f"Warning: Failed to reset git: {e}")

    def _save_bug_execution_record(self, run_id: int, execution_timestamp: str,
                                   phase: str, result: SubprocessResult):
        """
        Save detailed bug execution record including error information.

        Args:
            run_id: The run ID that failed.
            execution_timestamp: The execution timestamp for the directory path.
            phase: The phase that failed (e.g. "val", "test").
            result: The SubprocessResult containing error information.
        """
        try:
            # Construct execution directory path
            run_dir = osp.join(self.workspace_dir, f"run_{run_id}")
            execution_dir = osp.join(run_dir, f"execution_{execution_timestamp}")

            # Create records directory
            records_dir = osp.join(execution_dir, "records")
            os.makedirs(records_dir, exist_ok=True)

            # Create bug record
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            bug_filename = f"bug_{phase}_{timestamp}.json"
            bug_filepath = osp.join(records_dir, bug_filename)

            bug_record = {
                "run_id": run_id,
                "phase": phase,
                "timestamp": timestamp,
                "error": {
                    "returncode": result.returncode,
                    "stderr": result.stderr
                }
            }

            with open(bug_filepath, 'w') as f:
                json.dump(bug_record, f, indent=2)

            print(f"Bug execution record saved to: {bug_filepath}")

        except Exception as e:
            print(f"Warning: Failed to save bug execution record: {e}")
