#!/usr/bin/env python3
"""
Main script for running agents on benchmark tasks.

Usage:
    python run_agent_benchmark.py --agent-config configs/agents/aide.yaml --task-config configs/tasks/generalization.yaml --model gpt-5 --provider OpenAI
    python run_agent_benchmark.py --agent-config configs/agents/theaiscientist.yaml --task-config configs/tasks/causality_causalml.yaml
    python run_agent_benchmark.py --agent-config configs/agents/aide.yaml --task-config configs/tasks/generalization.yaml agent.aide.num_drafts=3
"""
import argparse
import json
import os
import signal
import sys
import yaml
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents import AgentConfig, AgentType, AgentRegistry, AgentResult
from benchmark.runner import BenchmarkRunner


# ---------------------------------------------------------------------------
# Config loading & overrides
# ---------------------------------------------------------------------------

def parse_args():
    """Parse command line arguments with support for nested config overrides."""
    parser = argparse.ArgumentParser(
        description="Run AI research agents on benchmark tasks",
        epilog="""
Examples:
  python run_agent_benchmark.py --agent-config configs/agents/aide.yaml --task-config configs/tasks/generalization.yaml --model gpt-5 --provider OpenAI
  python run_agent_benchmark.py --agent-config configs/agents/theaiscientist.yaml --task-config configs/tasks/causality_causalml.yaml agent.theaiscientist.max_runs=10
        """
    )
    parser.add_argument("--agent-config", type=str, required=True,
                        help="Agent config file (e.g., configs/agents/aide.yaml)")
    parser.add_argument("--task-config", type=str, required=True,
                        help="Task config file (e.g., configs/tasks/generalization.yaml)")
    parser.add_argument("--model", type=str,
                        help="Model to use (e.g., gpt-5-2025-08-07, gemini-2.5-pro)")
    parser.add_argument("--provider", type=str,
                        help="Provider to use (e.g., OpenAI, Google, OpenRouter)")
    parser.add_argument("--workspace-label", type=str, default=None,
                        help="Optional label for workspace copy. A unique ID is always appended.")
    parser.add_argument("--output-dir", type=str, default="benchmark_results",
                        help="Root directory for experiment outputs (default: benchmark_results)")
    parser.add_argument("overrides", nargs="*",
                        help="Config overrides in format: key=value (e.g., agent.aide.num_drafts=3)")
    return parser.parse_args()


def load_config(config_file: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


def _parse_override_value(value: str):
    """Convert an override string value to the appropriate Python type."""
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def apply_overrides(config: Dict[str, Any], args) -> Dict[str, Any]:
    """Apply command-line overrides (--model, --provider, positional key=value) to config."""
    overrides = list(args.overrides) if args.overrides else []
    if args.model:
        overrides.append(f"agent.model={args.model}")
    if args.provider:
        overrides.append(f"agent.provider={args.provider}")

    for override_str in overrides:
        if "=" not in override_str:
            print(f"Warning: Invalid override format (expected key=value): {override_str}")
            continue
        key, value = override_str.split("=", 1)
        value = _parse_override_value(value)
        keys = key.split(".")
        current = config
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value
        print(f"Override: {key} = {value}")

    return config


# ---------------------------------------------------------------------------
# Agent config construction
# ---------------------------------------------------------------------------

AGENT_TYPE_MAP = {
    "theaiscientist": AgentType.THEAISCIENTIST,
    "ai_scientist_v2": AgentType.AI_SCIENTIST_V2,
    "aide": AgentType.AIDE,
    "aira_mcts": AgentType.AIRA_MCTS,
    "openevolve": AgentType.OPENEVOLVE,
    "autoresearch": AgentType.AUTORESEARCH,
    "adaptivesearch": AgentType.ADAPTIVESEARCH,
}


def get_agent_config(config: Dict[str, Any]) -> AgentConfig:
    """Create AgentConfig from the loaded config dict."""
    agent_cfg = config.get("agent", {})
    agent_type_str = agent_cfg.get("type", "theaiscientist")

    if agent_type_str not in AGENT_TYPE_MAP:
        raise ValueError(
            f"Unknown agent type: {agent_type_str}. "
            f"Available: {sorted(AGENT_TYPE_MAP.keys())}"
        )

    # Resolve agent-specific params
    agent_params = agent_cfg.get(agent_type_str, {})

    runtime_params = {
        "metrics": config.get("metrics", {}),
    }

    return AgentConfig(
        agent_type=AGENT_TYPE_MAP[agent_type_str],
        model=agent_cfg.get("model", "gpt-5"),
        provider=agent_cfg.get("provider", "OpenAI"),
        agent_params=agent_params,
        runtime_params=runtime_params,
    )


# ---------------------------------------------------------------------------
# Result saving & printing
# ---------------------------------------------------------------------------

def save_results(result, config: Dict[str, Any], runner: BenchmarkRunner):
    """Save results to summary.json. Handles both AgentResult and legacy dict."""
    if isinstance(result, AgentResult):
        # Run-level metadata for reproducibility and analysis
        agent_cfg = runner.agent.config
        task_cfg = runner.config  # task's config.json
        baseline_primary = None
        try:
            from benchmark.utils import extract_primary_metric
            include_ds = agent_cfg.runtime_params.get("metrics", {}).get("include_datasets")
            baseline_primary = extract_primary_metric(
                runner.baseline_results, task_cfg.get("metric", ""), include_ds,
            )
        except Exception:
            pass

        summary = {
            "benchmark": runner.benchmark_name,
            "agent": agent_cfg.agent_type.value,
            "model": agent_cfg.model,
            "provider": agent_cfg.provider,
            "agent_params": agent_cfg.agent_params,
            "task_config": {
                "metric": task_cfg.get("metric", ""),
                "metric_direction": task_cfg.get("metric_direction", ""),
                "conda_env": task_cfg.get("conda_env", ""),
                "repo_dir": task_cfg.get("repo_dir", ""),
                "target_files": task_cfg.get("target_files", []),
            },
            "baseline_primary_metric": baseline_primary,
            "total_steps": result.total_steps,
            "total_ideas": result.total_ideas,
            "total_duration_seconds": result.total_duration_seconds,
            "best_val_metric": result.best_step.primary_metric if result.best_step else None,
            "test_result": result.test_result,
            "val_steps": [asdict(s) for s in result.all_steps],
            "token_usage": result.token_usage,
            "parent_workspace": result.parent_workspace,
            "metadata": result.metadata,
        }
        save_path = os.path.join(result.parent_workspace, "summary.json")
    elif isinstance(result, dict):
        summary = result
        pw = result.get("parent_workspace", "")
        if pw and os.path.isdir(pw):
            save_path = os.path.join(pw, "summary.json")
        else:
            save_path = os.path.join("benchmark_results", "summary.json")
    else:
        print(f"Warning: Unexpected result type {type(result)}, skipping save.")
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved to: {save_path}")

    # Also save the config used for reproducibility
    config_save_path = os.path.join(os.path.dirname(save_path), "config_used.yaml")
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Configuration saved to: {config_save_path}")


def print_summary(result):
    """Print a concise summary. Handles both AgentResult and legacy dict."""
    print(f"\n{'=' * 60}")
    print("Results Summary")
    print(f"{'=' * 60}")

    if isinstance(result, AgentResult):
        if result.best_step:
            print(f"Best Val Metric: {result.best_step.primary_metric}"
                  f" (step {result.best_step.step_id}, idea: {result.best_step.idea_description})")
        else:
            print("No successful validation runs.")

        if result.test_result and result.test_result.get("success"):
            print(f"Test Metric: {result.test_result.get('primary_metric')}")
        elif result.test_result:
            print(f"Test Failed: {result.test_result.get('error', 'unknown')}")

        print(f"Total Steps: {result.total_steps}")
        print(f"Total Ideas: {result.total_ideas}")

        # Token usage
        if result.token_usage:
            tu = result.token_usage
            print(f"\nToken Usage: {tu.get('total_tokens', 0):,} total"
                  f" (prompt: {tu.get('prompt_tokens', 0):,},"
                  f" completion: {tu.get('completion_tokens', 0):,})")

    elif isinstance(result, dict):
        # Legacy dict format -- print key fields if present
        print(f"Benchmark: {result.get('benchmark', 'N/A')}")
        print(f"Total ideas tested: {result.get('total_ideas_tested', 0)}")

        if result.get("best_idea"):
            bi = result["best_idea"]
            print(f"Best Idea: {bi.get('Title', 'Untitled')}")

        if "token_usage" in result:
            tu = result["token_usage"]
            print(f"Total LLM calls: {tu.get('total_calls', 0)}")
            print(f"Total tokens: {tu.get('total_tokens', 0):,}")

    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Load config: merge agent config + task config
    try:
        agent_cfg = load_config(args.agent_config)
        task_cfg = load_config(args.task_config)
        config = {
            "agent": agent_cfg.get("agent", {}),
            "benchmark": task_cfg.get("benchmark", {}),
            "metrics": task_cfg.get("metrics", {}),
        }
    except FileNotFoundError as e:
        print(f"Error: Configuration file not found: {e}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML in configuration file: {e}")
        sys.exit(1)

    apply_overrides(config, args)

    # Validate benchmark name
    benchmark_name = config.get("benchmark", {}).get("name")
    if not benchmark_name:
        print("Error: Benchmark name not specified in task config")
        sys.exit(1)

    # Create agent
    agent_config = get_agent_config(config)
    agent_type_str = config["agent"].get("type", "theaiscientist")
    try:
        agent = AgentRegistry.create(agent_config.agent_type, agent_config)
    except ValueError as e:
        if "not registered" in str(e):
            print(f"Error: Agent type '{agent_type_str}' is not yet implemented.")
            print(f"Available agents: {[a.value for a in AgentRegistry.list_agents()]}")
            sys.exit(1)
        raise

    print(f"Initializing {agent_type_str} agent...")
    agent.initialize()

    # Run benchmark
    print(f"\n=== Running {agent_type_str} on {benchmark_name} ===")
    print(f"Model: {agent_config.model} (Provider: {agent_config.provider})")

    runner = BenchmarkRunner(benchmark_name, agent, workspace_label=args.workspace_label,
                              output_dir=args.output_dir)

    # Register signal handlers to kill experiment subprocesses on external kill
    def _cleanup_on_signal(signum, frame):
        sig_name = signal.Signals(signum).name
        print(f"\nReceived {sig_name}, cleaning up...")
        agent.kill_running_process()
        runner.cleanup_workspace()
        sys.exit(1)

    signal.signal(signal.SIGTERM, _cleanup_on_signal)
    signal.signal(signal.SIGINT, _cleanup_on_signal)

    try:
        result = runner.run()
        save_results(result, config, runner)
        print_summary(result)
        agent.cleanup()
        print("\nDone!")
    finally:
        print("Cleaning up copied workspace...")
        agent.kill_running_process()
        runner.cleanup_workspace()


if __name__ == "__main__":
    main()
