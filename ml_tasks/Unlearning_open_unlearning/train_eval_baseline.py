"""
Training and evaluation script for Machine Unlearning task (TOFU benchmark).

Model: Llama-3.2-1B-Instruct (fine-tuned on TOFU)
Dataset: TOFU forget10 (400 QA) / retain90 (3600 QA)
Baseline: Gradient Ascent (negate loss on forget set)
Split: Fixed 30/70 eval split (no split_config — agents cannot modify val set).
       Unlearning training always uses full forget10/retain90.
       Evaluation QA pairs split 30% val / 70% test by fixed seed.
Metrics: forget_quality (KS-test p-value), model_utility

Usage:
    python train_eval_baseline.py --split val
    python train_eval_baseline.py --split test
"""
import argparse
import json
import os
import sys
import warnings
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from scipy.stats import ks_2samp

warnings.filterwarnings('ignore')

cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.insert(0, cwd)
from algorithm import get_unlearning_config, compute_unlearn_loss


IGNORE_INDEX = -100
VAL_RATIO = 0.3
VAL_SEED = 42
MAX_LENGTH = 512


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def load_tofu_dataset(split_name):
    """Load TOFU dataset from HuggingFace."""
    from datasets import load_dataset
    ds = load_dataset("locuslab/TOFU", split_name, split="train")
    return ds


def tokenize_qa(tokenizer, question, answer, max_length=MAX_LENGTH):
    """Tokenize a QA pair using chat template."""
    chat = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    chat_ids = tokenizer.apply_chat_template(
        chat, tokenize=True, add_generation_prompt=False
    )
    prompt_chat = [{"role": "user", "content": question}]
    prompt_ids = tokenizer.apply_chat_template(
        prompt_chat, tokenize=True, add_generation_prompt=True
    )
    if chat_ids[-1] != tokenizer.eos_token_id:
        chat_ids.append(tokenizer.eos_token_id)

    # Truncate
    chat_ids = chat_ids[:max_length]
    prompt_len = min(len(prompt_ids), len(chat_ids))

    # Labels: ignore prompt tokens
    labels = [IGNORE_INDEX] * prompt_len + chat_ids[prompt_len:]
    attention_mask = [1] * len(chat_ids)

    return {
        "input_ids": torch.tensor(chat_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attention_mask),
    }


class TOFUDataset(Dataset):
    """Simple TOFU QA dataset."""
    def __init__(self, hf_dataset, tokenizer, max_length=MAX_LENGTH):
        self.data = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return tokenize_qa(
            self.tokenizer, item["question"], item["answer"], self.max_length
        )


class ForgetRetainDataset(Dataset):
    """Pairs forget and retain samples (anchored on forget set)."""
    def __init__(self, forget_ds, retain_ds):
        self.forget_ds = forget_ds
        self.retain_ds = retain_ds

    def __len__(self):
        return len(self.forget_ds)

    def __getitem__(self, idx):
        forget_item = self.forget_ds[idx]
        retain_idx = idx % len(self.retain_ds)
        retain_item = self.retain_ds[retain_idx]
        return {"forget": forget_item, "retain": retain_item}


def collate_fn(batch):
    """Collate function that pads sequences."""
    def pad_items(items):
        max_len = max(item["input_ids"].size(0) for item in items)
        input_ids = torch.full((len(items), max_len), 0, dtype=torch.long)
        labels = torch.full((len(items), max_len), IGNORE_INDEX, dtype=torch.long)
        attention_mask = torch.zeros(len(items), max_len, dtype=torch.long)
        for i, item in enumerate(items):
            seq_len = item["input_ids"].size(0)
            input_ids[i, :seq_len] = item["input_ids"]
            labels[i, :seq_len] = item["labels"]
            attention_mask[i, :seq_len] = item["attention_mask"]
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

    if isinstance(batch[0], dict) and "forget" in batch[0]:
        forget_items = [b["forget"] for b in batch]
        retain_items = [b["retain"] for b in batch]
        return {"forget": pad_items(forget_items), "retain": pad_items(retain_items)}
    else:
        return pad_items(batch)


def compute_truth_ratios(model, dataset, tokenizer, device):
    """Compute truth ratio for each QA pair (per-sample, no batching).
    Truth ratio = P(wrong_answer) / P(correct_answer).
    Note: This uses an inverted convention vs the original TOFU paper
    (which uses correct/wrong). Both retain_reference_logs.json and this
    function use the same convention, so the KS-test comparison is valid.
    Uses fixed wrong_answer="I have no comment." instead of per-question
    perturbed answers from the TOFU dataset.
    """
    model.eval()
    truth_ratios = []

    for idx in range(len(dataset.data)):
        item = dataset.data[idx]
        question = item["question"]
        answer = item["answer"]

        # Compute loss for correct answer
        correct_tok = tokenize_qa(tokenizer, question, answer)
        correct_inputs = {
            k: v.unsqueeze(0).to(device) for k, v in correct_tok.items()
        }
        with torch.no_grad():
            correct_out = model(**correct_inputs)
        correct_loss = correct_out.loss.item()

        # Compute loss for a perturbed answer (use "I have no comment." as wrong answer)
        wrong_answer = "I have no comment."
        wrong_tok = tokenize_qa(tokenizer, question, wrong_answer)
        wrong_inputs = {
            k: v.unsqueeze(0).to(device) for k, v in wrong_tok.items()
        }
        with torch.no_grad():
            wrong_out = model(**wrong_inputs)
        wrong_loss = wrong_out.loss.item()

        # truth_ratio = P(wrong) / P(correct) = exp(-wrong_loss) / exp(-correct_loss)
        tr = np.exp(-wrong_loss) / (np.exp(-correct_loss) + 1e-10)
        truth_ratios.append(tr)

    return np.array(truth_ratios)


def compute_model_utility(model, retain_dataset, tokenizer, device, batch_size=8):
    """Compute model utility: average probability on retain set (higher = better utility)."""
    model.eval()
    loader = DataLoader(retain_dataset, batch_size=batch_size, collate_fn=collate_fn)
    total_loss = 0.0
    total_tokens = 0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        # Count non-ignored tokens
        valid_tokens = (batch["labels"] != IGNORE_INDEX).sum().item()
        total_loss += outputs.loss.item() * valid_tokens
        total_tokens += valid_tokens

    avg_loss = total_loss / (total_tokens + 1e-10)
    utility = np.exp(-avg_loss)  # Convert to probability
    return float(utility)


def split_indices(n, val_ratio=VAL_RATIO, seed=VAL_SEED):
    """Split indices into val/test with fixed seed."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_val = int(val_ratio * n)
    return perm[:n_val], perm[n_val:]


def main():
    parser = argparse.ArgumentParser(description='Machine Unlearning evaluation')
    parser.add_argument('--split', choices=['val', 'test'], required=True)
    args = parser.parse_args()

    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = get_unlearning_config()

    CHECKPOINT_DIR = 'model_checkpoint'

    # Load model and tokenizer
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_name = "open-unlearning/tofu_Llama-3.2-1B-Instruct_full"
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load TOFU datasets
    print("Loading TOFU datasets...")
    forget_hf = load_tofu_dataset("forget10")
    retain_hf = load_tofu_dataset("retain90")

    # Split forget indices for val/test evaluation
    val_indices, test_indices = split_indices(len(forget_hf))
    if args.split == 'val':
        eval_indices = val_indices
    else:
        eval_indices = test_indices
    print(f"Forget set: {len(forget_hf)} total, "
          f"val={len(val_indices)}, test={len(test_indices)}, "
          f"using {args.split}={len(eval_indices)}")

    # Create datasets
    forget_ds = TOFUDataset(forget_hf, tokenizer)
    retain_ds = TOFUDataset(retain_hf, tokenizer)

    if args.split == 'test':
        # Load checkpoint, skip training
        ckpt_path = os.path.join(CHECKPOINT_DIR, 'model')
        print(f"Loading checkpoint from {ckpt_path} (skipping training)")
        model = AutoModelForCausalLM.from_pretrained(
            ckpt_path, torch_dtype=torch.bfloat16
        ).to(device)
    else:
        # === VAL: Unlearning training ===
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16
        ).to(device)

        # Create paired forget-retain dataset
        paired_ds = ForgetRetainDataset(forget_ds, retain_ds)
        train_loader = DataLoader(
            paired_ds,
            batch_size=config['batch_size'],
            shuffle=True,
            collate_fn=collate_fn,
            drop_last=True,
        )

        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config.get('weight_decay', 0.01),
        )

        # Warmup + training
        total_steps = config['epochs'] * len(train_loader) // config['gradient_accumulation_steps']
        warmup_steps = int(config.get('warmup_ratio', 0.1) * total_steps)

        print(f"Starting unlearning: {config['epochs']} epochs, "
              f"lr={config['lr']}, method={config['method']}")

        model.train()
        global_step = 0
        for epoch in range(config['epochs']):
            epoch_loss = 0.0
            n_batches = 0
            for batch_idx, batch in enumerate(train_loader):
                forget_inputs = {k: v.to(device) for k, v in batch["forget"].items()}
                retain_inputs = {k: v.to(device) for k, v in batch["retain"].items()}

                loss = compute_unlearn_loss(model, forget_inputs, retain_inputs)
                loss = loss / config['gradient_accumulation_steps']
                loss.backward()

                if (batch_idx + 1) % config['gradient_accumulation_steps'] == 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config['max_grad_norm']
                    )
                    # Simple linear warmup
                    if global_step < warmup_steps:
                        lr_scale = (global_step + 1) / warmup_steps
                        for pg in optimizer.param_groups:
                            pg['lr'] = config['lr'] * lr_scale
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                epoch_loss += loss.item() * config['gradient_accumulation_steps']
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            print(f"  Epoch {epoch+1}/{config['epochs']}: avg_loss={avg_loss:.4f}")

        # Save checkpoint
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        model.save_pretrained(os.path.join(CHECKPOINT_DIR, 'model'))
        tokenizer.save_pretrained(os.path.join(CHECKPOINT_DIR, 'model'))
        print(f"Saved checkpoint to {CHECKPOINT_DIR}/model")

    # === Evaluation ===
    print(f"\nEvaluating on {args.split} portion ({len(eval_indices)} forget QA pairs)...")

    # Create eval subset of forget data
    eval_forget_hf = forget_hf.select(eval_indices.tolist())
    eval_forget_ds = TOFUDataset(eval_forget_hf, tokenizer)

    # Also split retain for evaluation (use same seed)
    retain_val_idx, retain_test_idx = split_indices(len(retain_hf))
    if args.split == 'val':
        eval_retain_hf = retain_hf.select(retain_val_idx.tolist())
    else:
        eval_retain_hf = retain_hf.select(retain_test_idx.tolist())
    eval_retain_ds = TOFUDataset(eval_retain_hf, tokenizer)

    # 1. Compute truth ratios on eval forget data
    print("Computing truth ratios on forget data...")
    unlearn_truth_ratios = compute_truth_ratios(
        model, eval_forget_ds, tokenizer, device
    )

    # 2. Load retain model reference truth ratios
    ref_path = "retain_reference_logs.json"
    if not os.path.exists(ref_path):
        raise FileNotFoundError(
            f"retain_reference_logs.json not found at {ref_path}. "
            "This file must be copied by val/test_command."
        )
    with open(ref_path) as f:
        ref_data = json.load(f)
    # Get reference truth ratios for the same eval indices
    all_ref_ratios = np.array(ref_data["truth_ratios"])
    retain_ref_ratios = all_ref_ratios[eval_indices]

    # 3. Compute forget_quality (KS-test)
    ks_stat, ks_pvalue = ks_2samp(unlearn_truth_ratios, retain_ref_ratios)
    forget_quality = float(ks_pvalue)

    # 4. Compute model_utility on retain eval data
    print("Computing model utility on retain data...")
    model_utility = compute_model_utility(
        model, eval_retain_ds, tokenizer, device
    )

    print(f"\n=== {args.split.upper()} Results ===")
    print(f"  forget_quality (KS p-value): {forget_quality:.6f}")
    print(f"  model_utility: {model_utility:.6f}")
    print(f"  truth_ratio mean: {np.mean(unlearn_truth_ratios):.4f}")

    # Format results
    results = {
        "tofu_unlearning": {
            "means": {
                "forget_quality_mean": forget_quality,
                "model_utility_mean": model_utility,
            },
            "stderrs": {
                "forget_quality_stderr": 0.0,
                "model_utility_stderr": 0.0,
            },
            "final_info_dict": {
                "forget_quality": forget_quality,
                "model_utility": model_utility,
            }
        }
    }

    os.makedirs('results_tmp', exist_ok=True)
    output_path = f'results_tmp/{args.split}_info.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results to {output_path}")


if __name__ == '__main__':
    main()
