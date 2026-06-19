#!/usr/bin/env python3
"""Standalone per-agent scoring for FML-bench.

Given ONE agent's result directory (18 task subdirs, each with a timestamped run
dir containing summary.json), compute and report three tables for that single run:

    1. Raw Performance          — canonical test metric per task
    2. Normalized Improvement   — the per-task normalized improvement reported in the paper
    3. Process-Level metrics    — 12 metrics across Exploration / Generalization /
                                  Reliability / Efficiency / Cost

All metric definitions match those used to produce the paper's results. This script
is self-contained.

Usage:
    conda activate fmlbench
    python compute_agent_metrics.py results/ai_scientist_v2
    python compute_agent_metrics.py results/ai_scientist_v2 --output-dir my_report

The 4 Exploration metrics require GraphCodeBERT embeddings of each step's code
snapshot; these are computed on demand (needs `torch` + `transformers`) and cached
to <output-dir>/embeddings_cache/. The baseline code embedding is read from the
live workspace under ml_tasks/<task>/config.json -> repo_dir, so the workspace must
be at its baseline (reset) state for those 4 metrics to be correct.

This script is READ-ONLY with respect to the experiment result path: it only reads
summary.json / test_info.json / step_snapshots and never
modifies or deletes anything there. The only files it writes are the 3 CSVs and the
embedding cache, all under --output-dir, which must be OUTSIDE the result path (the
script refuses an output dir nested inside the input agent dir).
"""
import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent

# ==========================================================================
# Task metadata
# ==========================================================================

TASKS = [
    "Causality_causalml",
    "Causality_gcastle",
    "Continual_Learning_continual_learning",
    "Continual_Learning_pycil",
    "Data_Efficiency_easyfsl",
    "Data_Efficiency_usb",
    "Fairness_and_Bias_aif360",
    "Fairness_fairlearn",
    "Federated_Learning_PFLlib",
    "Generalization_domainbed",
    "Generalization_domainbed_officehome",
    "Privacy_opacus",
    "Privacy_privacymeter",
    "Representation_Learning_lightly",
    "Representation_Learning_solo_learn",
    "Robustness_and_Reliability_art",
    "Robustness_openood",
    "Unlearning_open_unlearning",
]

UNLEARNING_TASK = "Unlearning_open_unlearning"

# canonical_metric(info_json, task) = info_json[dataset]["means"][metric_key]
TASK_META = {
    "Causality_causalml":                    ("ihdp_test",                 "mae_mean",                         "lower"),
    "Causality_gcastle":                     ("synthetic_dag",             "shd_mean",                         "lower"),
    "Continual_Learning_continual_learning": ("splitMNIST_scenario_class", "average_acc_mean",                 "higher"),
    "Continual_Learning_pycil":              ("cifar100_incremental",      "avg_incremental_acc_mean",         "higher"),
    "Data_Efficiency_easyfsl":               ("miniimagenet_test",         "accuracy_mean",                    "higher"),
    "Data_Efficiency_usb":                   ("cifar100_ssl",              "test_acc_mean",                    "higher"),
    "Fairness_and_Bias_aif360":              ("compas",                    "abs_aod_mean",                     "lower"),
    "Fairness_fairlearn":                    ("adult",                     "abs_demographic_parity_diff_mean", "lower"),
    "Federated_Learning_PFLlib":             ("cifar10_fl",                "test_acc_mean",                    "higher"),
    "Generalization_domainbed":              ("ColoredMNIST_test_env2",    "in_acc_mean",                      "higher"),
    "Generalization_domainbed_officehome":   ("OfficeHome_test_env0",      "avg_acc_mean",                     "higher"),
    "Privacy_opacus":                        ("cifar10_dp",                "test_acc_mean",                    "higher"),
    "Privacy_privacymeter":                  ("cifar10",                   "AUC_gap_mean",                     "lower"),
    "Representation_Learning_lightly":       ("cifar10_linear_probing",    "test_acc_mean",                    "higher"),
    "Representation_Learning_solo_learn":    ("cifar100_ssl",              "linear_eval_acc_mean",             "higher"),
    "Robustness_and_Reliability_art":        ("poisoned_mnist",            "defense_score_mean",               "higher"),
    "Robustness_openood":                    ("openood",                   "auroc_mean",                       "higher"),
    "Unlearning_open_unlearning":            ("tofu_unlearning",           "forget_quality_mean",              "higher"),
}

# (direction, p_best, p_worst) for the per-task normalized improvement.
# p_worst is "baseline" (use the per-task baseline) or a fixed constant.
RANGE_META = {
    "Generalization_domainbed":              ("higher", 1.0, 0.0),
    "Generalization_domainbed_officehome":   ("higher", 1.0, 0.0),
    "Data_Efficiency_easyfsl":               ("higher", 1.0, 0.0),
    "Data_Efficiency_usb":                   ("higher", 1.0, 0.0),
    "Representation_Learning_lightly":       ("higher", 1.0, 0.0),
    "Representation_Learning_solo_learn":    ("higher", 1.0, 0.0),
    "Continual_Learning_continual_learning": ("higher", 1.0, 0.0),
    "Continual_Learning_pycil":              ("higher", 1.0, 0.0),
    "Causality_causalml":                    ("lower",  0.0, "baseline"),
    "Causality_gcastle":                     ("lower",  0.0, "baseline"),
    "Robustness_and_Reliability_art":        ("higher", 1.0, 0.0),
    "Robustness_openood":                    ("higher", 1.0, 0.0),
    "Privacy_privacymeter":                  ("lower",  0.0, 0.5),
    "Privacy_opacus":                        ("higher", 1.0, 0.0),
    "Fairness_and_Bias_aif360":              ("lower",  0.0, 1.0),
    "Fairness_fairlearn":                    ("lower",  0.0, 1.0),
    "Unlearning_open_unlearning":            ("lower",  0.0, "baseline"),
    "Federated_Learning_PFLlib":             ("higher", 1.0, 0.0),
}

assert set(TASK_META) == set(TASKS)
assert set(RANGE_META) == set(TASKS)

# Embedding constants (GraphCodeBERT full-code embeddings)
GRAPHCODEBERT_MODEL = "microsoft/graphcodebert-base"
MAX_TOKENS = 512
EMBED_DIM = 768
CLUSTER_THRESHOLD = 0.015
EXCLUDED_BASENAMES = {"split_config.json", "notes.txt"}


# ==========================================================================
# Generic utilities
# ==========================================================================

def safe_load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def canonical_metric(info_json, task):
    if info_json is None:
        return None
    dataset, metric, _ = TASK_META[task]
    try:
        return float(info_json[dataset]["means"][metric])
    except (KeyError, TypeError, ValueError):
        return None


def is_run_candidate(d):
    if not d.is_dir():
        return False
    name = d.name
    return not (name.startswith("_") or name.endswith("_deprecated"))


def find_task_subdirs(agent_root, task):
    td = agent_root / task
    if not td.is_dir():
        return []
    return sorted(d for d in td.iterdir() if is_run_candidate(d))


def find_main_run_dir(agent_root, task):
    for d in find_task_subdirs(agent_root, task):
        if (d / "summary.json").is_file():
            return d
    return None


def find_final_test_exec_dir(agent_id, agent_root, task, main_dir):
    patterns = [
        "*_final_test/run_final_test/execution_*",
        "final_test/run_final_test/execution_*",
    ]
    if agent_id == "ai_scientist_v2":
        siblings = [d for d in find_task_subdirs(agent_root, task) if d != main_dir]
        cands = []
        for sib in siblings:
            for pat in patterns:
                cands.extend([p for p in sib.glob(pat) if p.is_dir()])
        return sorted(cands)[-1] if cands else None
    cands = []
    for pat in patterns:
        cands.extend([p for p in main_dir.glob(pat) if p.is_dir()])
    return sorted(cands)[-1] if cands else None


def find_test_info_in(exec_dir):
    p = (exec_dir / "test_info.json") if exec_dir is not None else None
    return p if p and p.is_file() else None


def find_test_bug_in(exec_dir):
    if exec_dir is None:
        return None
    records = exec_dir / "records"
    if not records.is_dir():
        return None
    hits = sorted(records.glob("bug_test_*.json"))
    return hits[-1] if hits else None


def has_constraint_violated(info_json):
    if not isinstance(info_json, dict):
        return False
    for dataset_data in info_json.values():
        if not isinstance(dataset_data, dict):
            continue
        for sect in ("means", "final_info_dict"):
            sd = dataset_data.get(sect, {})
            if isinstance(sd, dict) and sd.get("constraint_violated") is True:
                return True
    return False


def display_transform(value, task):
    """Post-hoc transform applied to the Unlearning metric (forget_quality -> -log10).

    Returns None for non-positive Unlearning values; on all real data
    forget_quality > 0. All other tasks return the value unchanged.
    """
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(v):
        return None
    if task == UNLEARNING_TASK:
        if v <= 0:
            return None
        return -math.log10(v)
    return v


def normalized_improvement(p_agent, p_baseline, direction, p_best, p_worst):
    """Per-task improvement over baseline, normalized to [0, 1] and clamped at 0.

    0 means no improvement over the baseline; 1 means the metric reaches the best
    attainable value for the task. Returns NaN when inputs are missing or the
    (p_best, p_worst) range is degenerate.
    """
    if p_agent is None or p_baseline is None:
        return np.nan
    try:
        a = float(p_agent); b = float(p_baseline); best = float(p_best)
    except (TypeError, ValueError):
        return np.nan
    if math.isnan(a) or math.isnan(b):
        return np.nan
    worst = b if p_worst == "baseline" else float(p_worst)
    denom = abs(best - worst)
    if denom == 0:
        return np.nan
    signed = (a - b) / denom if direction == "higher" else (b - a) / denom
    return max(signed, 0.0)


def compute_auc_for_run(summary, task):
    """Best-so-far normalized-improvement curve and its area under the curve over steps."""
    direction, p_best, p_worst_spec = RANGE_META[task]

    baseline_raw = summary.get("baseline_primary_metric")
    if baseline_raw is None:
        return None
    baseline = display_transform(baseline_raw, task)
    if baseline is None:
        return None

    if p_worst_spec == "baseline":
        p_worst = baseline
    else:
        p_worst = float(p_worst_spec)
    denom = abs(float(p_best) - p_worst)
    if denom == 0:
        return None

    val_steps = summary.get("val_steps", [])
    K = len(val_steps)
    if K == 0:
        return None

    best_so_far = baseline
    improvement_curve = []
    n_valid = 0
    step_of_best = 0

    for i, step in enumerate(val_steps):
        metric_raw = step.get("primary_metric")
        if metric_raw is not None:
            metric = display_transform(metric_raw, task)
            if metric is not None and not math.isinf(metric):
                n_valid += 1
                if direction == "higher" and metric > best_so_far:
                    best_so_far = metric
                    step_of_best = i + 1
                elif direction == "lower" and metric < best_so_far:
                    best_so_far = metric
                    step_of_best = i + 1

        if direction == "higher":
            imp = (best_so_far - baseline) / denom
        else:
            imp = (baseline - best_so_far) / denom
        improvement_curve.append(imp)

    return {
        "auc": sum(improvement_curve) / K,
        "improvement_curve": improvement_curve,
        "K": K,
        "n_valid": n_valid,
        "step_of_best": step_of_best,
    }


# ==========================================================================
# Test-metric resolution: final-test metric with baseline fallback
# ==========================================================================

def resolve_raw_test(agent_id, agent_root, task, main_dir):
    """Return (raw_test_canonical_or_None, status).

    raw_test_canonical is None when the agent earns no test credit and should fall
    back to baseline: no final-test directory, a crash/bug during the test, a
    missing or unreadable test_info.json, a constraint violation, or a failed
    metric extraction.
    """
    exec_dir = find_final_test_exec_dir(agent_id, agent_root, task, main_dir)
    if exec_dir is None:
        reason = "no_final_test_dir"
    else:
        bug = find_test_bug_in(exec_dir)
        tpath = find_test_info_in(exec_dir)
        if bug is not None:
            reason = "bug_test"
        elif tpath is None:
            reason = "missing_test_info"
        else:
            tj = safe_load_json(tpath)
            if tj is None:
                reason = "unreadable"
            elif has_constraint_violated(tj):
                return None, "constraint_violated->baseline"
            else:
                mv = canonical_metric(tj, task)
                if mv is None:
                    reason = "metric_extraction_failed"
                else:
                    return mv, "ok"
    return None, f"{reason}->baseline"


# ==========================================================================
# Exploration embeddings (GraphCodeBERT)
# ==========================================================================

def load_task_config(task):
    return json.loads((REPO_ROOT / "ml_tasks" / task / "config.json").read_text())


def get_repo_subdir(config):
    parts = config["repo_dir"].split("/")
    return "/".join(parts[2:]) if len(parts) > 2 else ""


def load_baseline_files(task, config):
    baselines = {}
    repo_path = REPO_ROOT / config["repo_dir"]
    for tf in config["target_files"]:
        if Path(tf).name in EXCLUDED_BASENAMES:
            continue
        fp = repo_path / tf
        if fp.is_file():
            baselines[tf] = fp.read_text()
    return baselines


def extract_target_relative(snapshot_key, repo_subdir):
    parts = snapshot_key.split("/")
    if len(parts) < 3:
        return None
    after_ws = "/".join(parts[2:])
    if repo_subdir and after_ws.startswith(repo_subdir + "/"):
        relative = after_ws[len(repo_subdir) + 1:]
    else:
        relative = after_ws
    if Path(relative).name in EXCLUDED_BASENAMES:
        return None
    return relative


def build_full_code(snapshot, baselines, repo_subdir):
    code_parts = []
    used_targets = set()
    for snap_key in sorted(snapshot.keys()):
        target_rel = extract_target_relative(snap_key, repo_subdir)
        if target_rel is None:
            continue
        if target_rel in baselines:
            code_parts.append(snapshot[snap_key])
            used_targets.add(target_rel)
    for tf in sorted(baselines.keys()):
        if tf not in used_targets:
            code_parts.append(baselines[tf])
    return "\n".join(code_parts)


def load_model(device):
    try:
        import torch  # noqa: F401
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        sys.exit("ERROR: Exploration metrics require `pip install torch transformers`.")
    tokenizer = AutoTokenizer.from_pretrained(GRAPHCODEBERT_MODEL)
    model = AutoModel.from_pretrained(GRAPHCODEBERT_MODEL).to(device)
    model.eval()
    return tokenizer, model


def embed_fullcode_cls_sum(text, tokenizer, model, device):
    """[CLS] token embedding summed across 510-token chunks of the full code."""
    import torch
    if not text or not text.strip():
        return np.zeros(EMBED_DIM, dtype=np.float32)
    tokens = tokenizer.tokenize(text)
    if len(tokens) == 0:
        return np.zeros(EMBED_DIM, dtype=np.float32)
    chunk_size = MAX_TOKENS - 2
    chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
    cls_token = tokenizer.cls_token
    eos_token = tokenizer.eos_token or tokenizer.sep_token
    embeddings = []
    for chunk in chunks:
        input_tokens = [cls_token] + chunk + [eos_token]
        token_ids = tokenizer.convert_tokens_to_ids(input_tokens)
        tokens_tensor = torch.tensor([token_ids]).to(device)
        with torch.no_grad():
            outputs = model(tokens_tensor)
            embeddings.append(outputs.last_hidden_state[:, 0, :])
    final_embedding = torch.stack(embeddings, dim=0).sum(dim=0).squeeze(0)
    return final_embedding.cpu().numpy().astype(np.float32)


def compute_step_embeddings(main_dir, task, get_model, device, cache_path):
    """Embed the baseline + each step's full code. Cache to NPZ. Returns dict or None.

    get_model() lazily loads (tokenizer, model) on the first cache miss, so a fully-cached
    re-run never loads GraphCodeBERT (and never requires torch/transformers).
    """
    if cache_path.is_file():
        return dict(np.load(cache_path, allow_pickle=False))

    snap_dir = main_dir / "step_snapshots"
    if not snap_dir.is_dir():
        return None
    config = load_task_config(task)
    baselines = load_baseline_files(task, config)
    if not baselines:
        return None
    repo_subdir = get_repo_subdir(config)

    tokenizer, model = get_model()
    baseline_text = "\n".join(content for _, content in sorted(baselines.items()))
    baseline_embedding = embed_fullcode_cls_sum(baseline_text, tokenizer, model, device)

    snap_files = sorted(snap_dir.glob("step_*_code.json"))
    n_steps = len(snap_files)
    embeddings = np.zeros((n_steps, EMBED_DIM), dtype=np.float32)
    step_ids = np.zeros(n_steps, dtype=np.int32)
    for i, snap_path in enumerate(snap_files):
        step_ids[i] = int(re.search(r"step_(\d+)", snap_path.name).group(1))
        snapshot = json.loads(snap_path.read_text())
        full_code = build_full_code(snapshot, baselines, repo_subdir)
        embeddings[i] = embed_fullcode_cls_sum(full_code, tokenizer, model, device)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, embeddings=embeddings, step_ids=step_ids,
                        baseline_embedding=baseline_embedding)
    return {"embeddings": embeddings, "step_ids": step_ids,
            "baseline_embedding": baseline_embedding}


def compute_exploration_metrics(data):
    """Spread, Uniqueness, Reach, and Effective dim from the cached step embeddings."""
    try:
        from sklearn.cluster import AgglomerativeClustering
    except ImportError:
        sys.exit("ERROR: Exploration metrics require `pip install scikit-learn`.")

    nan4 = {"Exploration Spread": np.nan, "Exploration Uniqueness": np.nan,
            "Exploration Reach": np.nan, "Effective dim": np.nan}
    if data is None:
        return nan4
    # Deliberate dtype asymmetry: Spread/Uniqueness are computed on the float32
    # embeddings as cached, while Reach/Effective dim promote to float64 to avoid
    # catastrophic cancellation in the centered SVD.
    embs32 = data["embeddings"]
    N = len(embs32)
    if N == 0:
        return nan4

    norms = np.linalg.norm(embs32, axis=1)
    valid = norms > 1e-8
    n_valid = int(valid.sum())
    n_zero = N - n_valid

    # Spread = mean L2 distance to centroid (Centroid_L2, all embeddings)
    centroid32 = np.mean(embs32, axis=0)
    spread = float(np.mean([np.linalg.norm(e - centroid32) for e in embs32]))

    # Uniqueness = #clusters / N (cosine, average linkage, tau=0.015; zero-norm -> +1 cluster)
    if n_valid < 2:
        n_clusters = n_valid
    else:
        clustering = AgglomerativeClustering(
            n_clusters=None, metric="cosine", linkage="average",
            distance_threshold=CLUSTER_THRESHOLD)
        clustering.fit(embs32[valid])
        n_clusters = clustering.n_clusters_
    if n_zero > 0:
        n_clusters += 1
    uniqueness = float(n_clusters) / N

    # Reach = max L2 distance to baseline embedding (float64)
    embs64 = embs32.astype(np.float64)
    baseline64 = data["baseline_embedding"].astype(np.float64)
    reach = float(np.max(np.linalg.norm(embs64 - baseline64, axis=1)))

    # Effective dim = participation ratio (tr^2 / sum(eig^2)) of the centered cloud (float64)
    if N >= 2:
        centroid64 = np.mean(embs64, axis=0)
        s = np.linalg.svd(embs64 - centroid64, compute_uv=False)
        eigs = (s ** 2) / (N - 1)
        tr = float(eigs.sum())
        sqsum = float(np.sum(eigs ** 2))
        eff_dim = float(tr * tr / sqsum) if sqsum > 0 else 0.0
    else:
        eff_dim = 0.0

    return {"Exploration Spread": spread, "Exploration Uniqueness": uniqueness,
            "Exploration Reach": reach, "Effective dim": eff_dim}


# ==========================================================================
# Per-task scoring
# ==========================================================================

PROCESS_COLS = [
    "Exploration Spread", "Exploration Uniqueness", "Exploration Reach", "Effective dim",
    "Val-test |gap|", "Valid step ratio", "AUC-over-steps", "First-improvement step",
    "Late-gain fraction", "Best-improvement step", "Token cost (M)", "Wall-clock time (h)",
]


def score_task(agent_root, task, baselines, embed_ctx):
    """Compute all metrics for one task. Returns a dict of row values."""
    direction, p_best, p_worst_spec = RANGE_META[task]
    baseline_test = baselines[task]["test"]
    baseline_val = baselines[task]["val"]

    main_dir = find_main_run_dir(agent_root, task)
    if main_dir is None:
        raise FileNotFoundError(f"No run dir with summary.json for task '{task}' under {agent_root}")
    summary = safe_load_json(main_dir / "summary.json")
    if summary is None:
        raise FileNotFoundError(f"Unreadable summary.json for task '{task}' ({main_dir})")
    agent_id = summary.get("agent") or agent_root.name

    # --- Raw test + provenance ---
    raw_test_canon, status = resolve_raw_test(agent_id, agent_root, task, main_dir)
    best_val_metric = summary.get("best_val_metric")
    all_val_failed = best_val_metric is None
    if all_val_failed:
        status = "all_val_failed->baseline"
    # Test side falls back to baseline when the agent earns no test credit
    # (constraint violation / crash, or all val steps failed). The val side falls
    # back to baseline ONLY when all val steps failed.
    if raw_test_canon is None or all_val_failed:
        raw_test_canon = baseline_test  # test = baseline  ->  normalized improvement = 0

    # --- Raw performance (display-transformed for Unlearning) ---
    raw_disp = display_transform(raw_test_canon, task)
    baseline_disp = display_transform(baseline_test, task)

    # --- Normalized improvement (test side) ---
    ni_test = normalized_improvement(raw_disp, baseline_disp, direction, p_best, p_worst_spec)

    # --- Val-test |gap| ---
    val_for_norm = baseline_val if all_val_failed else best_val_metric
    ni_val = normalized_improvement(
        display_transform(val_for_norm, task),
        display_transform(baseline_val, task),
        direction, p_best, p_worst_spec)
    val_test_gap = abs(ni_val - ni_test) if not (pd.isna(ni_val) or pd.isna(ni_test)) else np.nan

    # --- Reliability: valid step ratio ---
    val_steps = summary.get("val_steps") or []
    total_steps = summary.get("total_steps") or 0
    success_steps = sum(1 for s in val_steps
                        if ((s.get("val_result") or {}).get("success") is True))
    valid_ratio = (success_steps / total_steps) if total_steps > 0 else np.nan

    # --- Efficiency: AUC, first-improvement, late-gain (from the improvement curve) ---
    auc_res = compute_auc_for_run(summary, task)
    if auc_res is None:
        auc = first_impr = late_gain = np.nan
    else:
        ic = auc_res["improvement_curve"]
        K = len(ic)
        auc = auc_res["auc"]
        first_impr = np.nan
        for idx_fi, v in enumerate(ic):
            if v > 1e-15:
                first_impr = idx_fi + 1
                break
        gain_total = ic[-1]
        gain_at_half = ic[K // 2 - 1]
        late_gain = ((gain_total - gain_at_half) / gain_total) if gain_total > 1e-12 else np.nan

    # --- Efficiency: best-improvement step (last step matching best_val_metric) ---
    best_step_id = np.nan
    if best_val_metric is not None:
        matches = [i + 1 for i, s in enumerate(val_steps)
                   if (s.get("val_result") or {}).get("success") is True
                   and s.get("primary_metric") == best_val_metric]
        if matches:
            best_step_id = float(matches[-1])

    # --- Cost ---
    tok = summary.get("token_usage") or {}
    total_tokens = tok.get("total_tokens")
    token_cost_M = (total_tokens / 1e6) if total_tokens is not None else np.nan
    dur = summary.get("total_duration_seconds")
    time_h = (dur / 3600.0) if dur is not None else np.nan

    # --- Exploration (embeddings) ---
    get_model, device, cache_dir = embed_ctx
    # Cache key includes a hash of the resolved run dir so two different result dirs of the
    # same agent (e.g. from different runs) can never collide on a stale cache.
    run_tag = hashlib.sha1(str(main_dir).encode()).hexdigest()[:10]
    cache_path = cache_dir / f"{agent_id}_{task}_{run_tag}.npz"
    data = compute_step_embeddings(main_dir, task, get_model, device, cache_path)
    expl = compute_exploration_metrics(data)

    return {
        "task": task,
        # raw table
        "metric": TASK_META[task][1],
        "direction": direction,
        "baseline_test": baseline_disp,
        "raw_test": raw_disp,
        "note": status,
        # normalized improvement
        "Normalized Improvement": ni_test,
        # 12 process metrics
        "Exploration Spread": expl["Exploration Spread"],
        "Exploration Uniqueness": expl["Exploration Uniqueness"],
        "Exploration Reach": expl["Exploration Reach"],
        "Effective dim": expl["Effective dim"],
        "Val-test |gap|": val_test_gap,
        "Valid step ratio": valid_ratio,
        "AUC-over-steps": auc,
        "First-improvement step": first_impr,
        "Late-gain fraction": late_gain,
        "Best-improvement step": best_step_id,
        "Token cost (M)": token_cost_M,
        "Wall-clock time (h)": time_h,
    }


# ==========================================================================
# Output
# ==========================================================================

def df_to_markdown(df):
    cols = list(df.columns)
    def esc(s):
        return str(s).replace("|", "\\|")  # literal pipes (e.g. "Val-test |gap|") would split cells
    def fmt(x):
        if isinstance(x, float):
            if pd.isna(x):
                return ""
            return f"{x:.4f}"
        return esc(x)
    header = "| " + " | ".join(esc(c) for c in [df.index.name or ""] + cols) + " |"
    sep = "| " + " | ".join(["---"] * (len(cols) + 1)) + " |"
    lines = [header, sep]
    for idx, row in df.iterrows():
        lines.append("| " + " | ".join([esc(idx)] + [fmt(row[c]) for c in cols]) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Compute Raw / Normalized-Improvement / 12 process metrics for one agent run.")
    parser.add_argument("agent_dir",
                        help="Agent result dir, e.g. results/ai_scientist_v2")
    parser.add_argument("--output-dir", default=None,
                        help="Where to write the CSVs + embedding cache. MUST be outside the "
                             "experiment result path (default: ./metric_reports/<agent_name> in "
                             "the current working directory).")
    args = parser.parse_args()

    agent_root = Path(args.agent_dir).resolve()
    if not agent_root.is_dir():
        sys.exit(f"ERROR: not a directory: {agent_root}")

    if args.output_dir:
        out_dir = Path(args.output_dir).resolve()
    else:
        out_dir = (Path.cwd() / "metric_reports" / agent_root.name).resolve()

    # Safety: this script must NEVER write into the experiment result tree. Refuse an
    # output dir that is the result dir itself or nested inside it. All other paths in
    # this script are opened read-only.
    if out_dir == agent_root or agent_root in out_dir.parents:
        sys.exit(f"ERROR: --output-dir ({out_dir}) is inside the experiment result dir "
                 f"({agent_root}). Choose a location OUTSIDE the result path; this script "
                 f"must not write into / modify original experiment data.")

    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "embeddings_cache"

    # Baselines (static, from ml_tasks)
    baselines = {}
    for task in TASKS:
        base_dir = REPO_ROOT / "ml_tasks" / task / "baseline_results"
        baselines[task] = {
            "val": canonical_metric(safe_load_json(base_dir / "val_info.json"), task),
            "test": canonical_metric(safe_load_json(base_dir / "test_info.json"), task),
        }

    device = "cpu"
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        pass
    # Lazy: GraphCodeBERT loads only on the first embedding cache miss (full-cache reruns and
    # the non-embedding tables never need torch/transformers).
    _model_cache = {}
    def get_model():
        if "loaded" not in _model_cache:
            print(f"Loading {GRAPHCODEBERT_MODEL} on {device} (for exploration metrics) ...")
            _model_cache["loaded"] = load_model(device)
        return _model_cache["loaded"]
    embed_ctx = (get_model, device, cache_dir)

    rows = []
    for task in TASKS:
        print(f"  scoring {task} ...")
        rows.append(score_task(agent_root, task, baselines, embed_ctx))
    df = pd.DataFrame(rows).set_index("task")

    # --- Table 1: Raw Performance ---
    raw_df = df[["metric", "direction", "baseline_test", "raw_test", "note"]].copy()
    raw_df.loc["MEAN"] = ["", "", np.nan, df["raw_test"].mean(skipna=True),
                          "mixed units across tasks - not comparable; use Normalized Improvement"]

    # --- Table 2: Normalized Improvement ---
    ni_df = df[["Normalized Improvement"]].copy()
    ni_df.loc["MEAN"] = [df["Normalized Improvement"].mean(skipna=True)]

    # --- Table 3: 12 Process metrics ---
    proc_df = df[PROCESS_COLS].copy()
    proc_df.loc["MEAN"] = [df[c].mean(skipna=True) for c in PROCESS_COLS]

    raw_csv = out_dir / "table_raw_performance.csv"
    ni_csv = out_dir / "table_normalized_improvement.csv"
    proc_csv = out_dir / "table_process_metrics.csv"
    raw_df.to_csv(raw_csv)
    ni_df.to_csv(ni_csv)
    proc_df.to_csv(proc_csv)

    print("\n" + "=" * 78 + "\n RAW PERFORMANCE\n" + "=" * 78)
    print(df_to_markdown(raw_df))
    print("\n" + "=" * 78 + "\n NORMALIZED IMPROVEMENT\n" + "=" * 78)
    print(df_to_markdown(ni_df))
    print("\n" + "=" * 78 + "\n PROCESS-LEVEL METRICS\n" + "=" * 78)
    print(df_to_markdown(proc_df))

    print(f"\nCSVs written to: {out_dir}")
    print(f"  {raw_csv.name}\n  {ni_csv.name}\n  {proc_csv.name}")


if __name__ == "__main__":
    main()
