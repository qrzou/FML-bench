"""
BenchmarkRunner V2: Simplified version that delegates execution to agents.
Main class for managing benchmark execution workflow with agent abstraction support.
"""
import json
import os
import os.path as osp
from typing import Dict, Any, Optional

from agents.base import BaseAgent


class BenchmarkRunner:
    """
    Simplified BenchmarkRunner that delegates the entire execution pipeline to agents.
    
    Usage example:
        from agents import AgentConfig, AgentType, AgentRegistry
        
        config = AgentConfig(
            agent_type=AgentType.THEAISCIENTIST,
            model="gpt-4",
            ...
        )
        agent = AgentRegistry.create(AgentType.THEAISCIENTIST, config)
        agent.initialize()
        
        runner = BenchmarkRunner("cider", agent)
        results = runner.run()  # or runner.run(idea)
    """
    
    def __init__(self, benchmark_name: str, agent: BaseAgent):
        self.benchmark_name = benchmark_name  # e.g. "cider"
        self.agent = agent  # BaseAgent instance
        self.agent_name = agent.config.agent_type.value  # Get agent type name for directory structure
        self.config = self._load_config()  # benchmark configuration
        self.task_description = self._load_task_description()  # load task description if available
        
    def run(self) -> Dict[str, Any]:
        """
        Run the benchmark experiment.
        
        Returns:
            Results dictionary from the agent
        """
        print(f"Starting benchmark: {self.benchmark_name}")
        
        try:
            # Prepare target files with full paths
            target_files = []
            for file_path in self.config.get("target_files", []):
                full_path = osp.join(self.config["repo_dir"], file_path)
                target_files.append(full_path)
            
            # Load baseline results
            baseline_results = self._load_baseline()
            
            # Update agent configuration with benchmark-specific runtime parameters
            if hasattr(self.agent.config, 'runtime_params'):
                self.agent.config.runtime_params["repo_dir"] = self.config["repo_dir"]
                self.agent.config.runtime_params["base_dir"] = osp.join("ml_tasks", self.benchmark_name)
                self.agent.config.runtime_params["benchmark_config"] = self.config
                self.agent.config.runtime_params["agent_name"] = self.agent_name
                self.agent.config.runtime_params["benchmark_name"] = self.benchmark_name
            
            # Execute agent pipeline (agent will create executors for each idea)
            results = self.agent.run(
                task_description=self.task_description,
                target_files=target_files,
                baseline_results=baseline_results
            )
            
            # Add benchmark metadata to results
            results["benchmark"] = self.benchmark_name
            results["agent"] = self.agent_name
            
            print(f"Benchmark completed: {self.benchmark_name}")
            return results
            
        except Exception as e:
            print(f"Error during benchmark execution: {e}")
            raise
            
    def _load_config(self) -> dict:
        """Load benchmark configuration"""
        config_path = osp.join("ml_tasks", self.benchmark_name, "config.json")
        
        if not osp.exists(config_path):
            raise FileNotFoundError(f"Configuration not found: {config_path}")
            
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        # Validate required fields
        required_fields = ["repo_dir", "target_files", "execute_commands"]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field in config: {field}")
                
        return config
        
    def _load_task_description(self) -> Optional[str]:
        """Load task description from ml_tasks if available"""
        # Try to load from task_description.txt
        task_file = osp.join("ml_tasks", self.benchmark_name, "task_description.txt")
        if osp.exists(task_file):
            with open(task_file, 'r') as f:
                return f.read().strip()
                
        # Try to load from prompt.json
        prompt_file = osp.join("ml_tasks", self.benchmark_name, "prompt.json")
        if osp.exists(prompt_file):
            with open(prompt_file, 'r') as f:
                prompt_data = json.load(f)
                # Extract task description from prompt data if available
                if "system" in prompt_data:
                    return prompt_data["system"]
        
        raise ValueError(f"No task description found in {task_file} or {prompt_file}")
                    
        # # Default task description
        # return f"Improve the performance of the {self.benchmark_name} benchmark"
        
    def _load_baseline(self) -> dict:
        """Load baseline results"""
        baseline_path = osp.join(
            "ml_tasks", 
            self.benchmark_name, 
            "baseline_results", 
            "final_info.json"
        )
        
        if not osp.exists(baseline_path):
            print(f"Warning: Baseline results not found at {baseline_path}")
            return {}
            
        with open(baseline_path, 'r') as f:
            baseline = json.load(f)
            
        # Extract means if present
        if isinstance(baseline, dict):
            baseline = {k: v.get("means", v) if isinstance(v, dict) else v 
                       for k, v in baseline.items()}
                       
        return baseline