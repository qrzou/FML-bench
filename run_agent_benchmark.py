#!/usr/bin/env python3
"""
Simplified main script for running agents on benchmark tasks.
Uses YAML configuration for easy setup.

Usage:
    # Using default config
    python run_agent_benchmark.py
    
    # Using custom config file
    python run_agent_benchmark.py --config my_config.yaml
    
    # Override model and provider directly
    python run_agent_benchmark.py --model gpt-4 --provider OpenAI
    
    # Override specific parameters
    python run_agent_benchmark.py --agent theaiscientist --benchmark Generalization_domainbed --model gpt-4
"""
import argparse
import json
import os
import sys
import yaml
from datetime import datetime
from typing import Dict, Any

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents import AgentConfig, AgentType, AgentRegistry
from benchmark.runner import BenchmarkRunner


def load_config(config_file: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


def parse_arguments():
    """Parse command line arguments with support for nested config overrides"""
    parser = argparse.ArgumentParser(
        description="Run AI research agents on benchmark tasks",
        epilog="""
Examples:
  python run_agent_benchmark.py
  python run_agent_benchmark.py --config my_config.yaml
  python run_agent_benchmark.py --model gpt-4 --provider OpenAI
  python run_agent_benchmark.py agent.model=gpt-4 agent.theaiscientist.max_runs=10
  python run_agent_benchmark.py benchmark.name=Generalization_domainbed agent.provider=OpenRouter
  python run_agent_benchmark.py agent.theaiscientist.use_existing_ideas=true agent.theaiscientist.ideas_file=ideas.json
        """
    )
    
    # Config file
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Configuration file path (default: configs/default.yaml)"
    )
    
    # Direct model and provider arguments for convenience
    parser.add_argument(
        "--model",
        type=str,
        help="Model to use (e.g., gpt-5-2025-08-07, gemini-2.5-pro)"
    )
    
    parser.add_argument(
        "--provider",
        type=str,
        help="Provider to use (e.g., OpenAI, Google, OpenRouter)"
    )
    
    # All other arguments are treated as config overrides
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Config overrides in format: key=value (e.g., agent.model=gpt-4)"
    )
    
    return parser.parse_args()


def parse_override(override_str):
    """Parse a single override string into key and value"""
    if "=" not in override_str:
        raise ValueError(f"Invalid override format: {override_str}. Expected key=value")
    
    key, value = override_str.split("=", 1)
    
    # Try to convert value to appropriate type
    # Handle boolean
    if value.lower() in ["true", "false"]:
        return key, value.lower() == "true"
    
    # Try integer
    try:
        return key, int(value)
    except ValueError:
        pass
    
    # Try float
    try:
        return key, float(value)
    except ValueError:
        pass
    
    # Return as string
    return key, value


def apply_overrides(config, overrides):
    """Apply override values to config dictionary"""
    for override_str in overrides:
        try:
            key, value = parse_override(override_str)
            
            # Split key by dots for nested access
            keys = key.split(".")
            current = config
            
            # Navigate to the nested location
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            
            # Set the value
            current[keys[-1]] = value
            print(f"Override: {key} = {value}")
            
        except Exception as e:
            print(f"Warning: Failed to apply override '{override_str}': {e}")
    
    return config


def get_agent_config(config: Dict[str, Any]) -> AgentConfig:
    """Create agent configuration from config dict"""
    agent_cfg = config.get("agent", {})
    agent_type = agent_cfg.get("type", "theaiscientist")
    
    # Map agent name to AgentType
    agent_type_map = {
        "theaiscientist": AgentType.THEAISCIENTIST,
    }
    
    if agent_type not in agent_type_map:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    # Get agent-specific settings from config
    agent_params = agent_cfg.get(agent_type, {})
    
    # Runtime parameters will be added by BenchmarkRunner
    runtime_params = {
        "metrics": config.get("metrics", {})  # Metrics configuration
    }
    
    # Create agent configuration
    return AgentConfig(
        agent_type=agent_type_map[agent_type],
        model=agent_cfg.get("model", "gpt-4"),
        provider=agent_cfg.get("provider", "OpenAI"),
        agent_params=agent_params,
        runtime_params=runtime_params
    )


def main():
    args = parse_arguments()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML in configuration file: {e}")
        sys.exit(1)
    
    # Apply command-line overrides
    overrides = list(args.overrides) if args.overrides else []
    
    # Add direct model and provider arguments as overrides
    if args.model:
        overrides.append("agent.model=" + args.model)
    if args.provider:
        overrides.append("agent.provider=" + args.provider)
    
    if overrides:
        config = apply_overrides(config, overrides)
    
    # Get benchmark name
    benchmark_name = config.get("benchmark", {}).get("name")
    if not benchmark_name:
        print("Error: Benchmark name not specified in config or command line")
        sys.exit(1)
    
    # Create agent
    agent_config = get_agent_config(config)
    try:
        agent = AgentRegistry.create(agent_config.agent_type, agent_config)
    except ValueError as e:
        if "not registered" in str(e):
            print(f"Error: Agent type '{config['agent']['type']}' is not yet implemented.")
            print("Currently available agents: theaiscientist")
            sys.exit(1)
        raise
    
    # Initialize agent
    print(f"Initializing {config['agent']['type']} agent...")
    agent.initialize()
    
    # Create benchmark runner
    runner = BenchmarkRunner(benchmark_name, agent)
    
    # Run benchmark
    print(f"\n=== Running {config['agent']['type']} on {benchmark_name} ===")
    print(f"Model: {config['agent']['model']} (Provider: {config['agent']['provider']})")
    
    # Run benchmark - agent will handle idea generation/loading internally
    start_time = datetime.now().isoformat()
    results = runner.run()
    end_time = datetime.now().isoformat()
    results["start_time"] = start_time
    results["end_time"] = end_time
    
    # Print results summary
    print("\n=== Results Summary ===")
    print(f"Benchmark: {results.get('benchmark')}")
    print(f"Experiment: {results.get('experiment')}")
    print(f"Total ideas tested: {results.get('total_ideas_tested', 0)}")
    
    # Report best idea and its results
    if "best_idea" in results and results["best_idea"]:
        best_idea = results["best_idea"]
        best_idea_key = results.get("best_idea_key")
        print(f"\n=== Best Idea ===")
        print(f"Title: {best_idea.get('Title', 'Untitled')}")
        print(f"Experiment: {best_idea.get('Experiment', 'No description')}")
        
        # Get best idea's summary
        if best_idea_key and "all_ideas" in results:
            best_summary = results["all_ideas"][best_idea_key]["summary"]
            print(f"Total runs: {best_summary.get('total_runs', 0)}")
            print(f"Successful runs: {best_summary.get('successful_runs', 0)}")
            
            if "improvements" in best_summary:
                print("\nImprovements over baseline:")
                for metric, improvement in best_summary["improvements"].items():
                    print(f"  {metric}: {improvement['baseline']:.4f} -> {improvement['best']:.4f} "
                          f"({improvement['improvement']:+.2f}%)")
                print(f"\nOverall improvement: {best_summary.get('overall_improvement', 0):.2f}%")
    else:
        print("\nNo improvements found over baseline.")
    
    # Report on all ideas tested
    if "all_ideas" in results:
        print(f"\n=== All Ideas Results ===")
        for idea_key, idea_data in results["all_ideas"].items():
            idea = idea_data["idea"]
            summary = idea_data["summary"]
            print(f"\n{idea_key}: {idea.get('Title', 'Untitled')}")
            print(f"  Runs: {summary.get('total_runs', 0)} (successful: {summary.get('successful_runs', 0)})")
            if "overall_improvement" in summary:
                print(f"  Overall improvement: {summary.get('overall_improvement', 0):.2f}%")
            if "workspace" in idea_data:
                print(f"  Workspace: {idea_data['workspace']}")
    
    # Report token usage summary
    if "token_usage" in results:
        token_summary = results["token_usage"]
        print(f"\n=== Token Usage Summary ===")
        print(f"Total LLM calls: {token_summary.get('total_calls', 0)}")
        print(f"Total tokens: {token_summary.get('total_tokens', 0):,} (prompt: {token_summary.get('total_prompt_tokens', 0):,}, completion: {token_summary.get('total_completion_tokens', 0):,})")
        
        if token_summary.get('calls_by_step'):
            print(f"\nBy step:")
            for step, calls in token_summary['calls_by_step'].items():
                tokens = token_summary['tokens_by_step'][step]
                print(f"  {step}: {calls} calls, {tokens:,} tokens")
        
        if token_summary.get('calls_by_idea'):
            print(f"\nBy idea:")
            for idea, calls in token_summary['calls_by_idea'].items():
                tokens = token_summary['tokens_by_idea'][idea]
                print(f"  {idea}: {calls} calls, {tokens:,} tokens")
    
    # Save overall summary results
    # Use parent workspace if available, otherwise create new timestamp
    if "parent_workspace" in results and results["parent_workspace"]:
        summary_file = os.path.join(results["parent_workspace"], "summary.json")
        config_save_path = os.path.join(results["parent_workspace"], "config_used.yaml")
        try:
            with open(config_save_path, 'w') as f:
                yaml.dump(config, f)
            print(f"Configuration saved to: {config_save_path}")
        except Exception as e:
            print(f"Warning: Could not save configuration to {config_save_path}: {e}")
    else:
        # Fallback for backward compatibility
        summary_dir = os.path.join("benchmark_results", agent.config.agent_type.value, benchmark_name)
        os.makedirs(summary_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = os.path.join(summary_dir, f"summary_{timestamp}.json")
    
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nOverall summary saved to: {summary_file}")
    
    # Cleanup agent
    agent.cleanup()
    
    print("\nDone!")


if __name__ == "__main__":
    main()
