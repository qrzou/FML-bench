"""
Pre-compute retain model truth ratios on forget data.
These are used as reference for the KS-test in forget_quality computation.
Run once, save to retain_reference_logs.json.
"""
import json
import numpy as np
import torch
import sys
import os

sys.path.insert(0, os.getcwd())


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load retain-only model
    model_name = "open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90"
    print(f"Loading retain model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()

    # Load forget10 data
    print("Loading TOFU forget10 dataset...")
    forget_hf = load_dataset("locuslab/TOFU", "forget10", split="train")
    print(f"Forget set size: {len(forget_hf)}")

    # Compute truth ratios for each QA pair
    truth_ratios = []
    for idx in range(len(forget_hf)):
        item = forget_hf[idx]
        question = item["question"]
        answer = item["answer"]

        # Tokenize correct answer
        correct_chat = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        correct_ids = tokenizer.apply_chat_template(
            correct_chat, tokenize=True, add_generation_prompt=False
        )
        prompt_chat = [{"role": "user", "content": question}]
        prompt_ids = tokenizer.apply_chat_template(
            prompt_chat, tokenize=True, add_generation_prompt=True
        )
        if correct_ids[-1] != tokenizer.eos_token_id:
            correct_ids.append(tokenizer.eos_token_id)
        correct_ids = correct_ids[:512]  # Truncate for consistency with tokenize_qa()

        prompt_len = min(len(prompt_ids), len(correct_ids))
        labels_correct = [-100] * prompt_len + correct_ids[prompt_len:]

        correct_inputs = {
            "input_ids": torch.tensor([correct_ids]).to(device),
            "labels": torch.tensor([labels_correct]).to(device),
            "attention_mask": torch.ones(1, len(correct_ids), dtype=torch.long).to(device),
        }

        # Tokenize wrong answer
        wrong_answer = "I have no comment."
        wrong_chat = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": wrong_answer},
        ]
        wrong_ids = tokenizer.apply_chat_template(
            wrong_chat, tokenize=True, add_generation_prompt=False
        )
        if wrong_ids[-1] != tokenizer.eos_token_id:
            wrong_ids.append(tokenizer.eos_token_id)
        wrong_ids = wrong_ids[:512]  # Truncate for consistency with tokenize_qa()

        prompt_len_w = min(len(prompt_ids), len(wrong_ids))
        labels_wrong = [-100] * prompt_len_w + wrong_ids[prompt_len_w:]

        wrong_inputs = {
            "input_ids": torch.tensor([wrong_ids]).to(device),
            "labels": torch.tensor([labels_wrong]).to(device),
            "attention_mask": torch.ones(1, len(wrong_ids), dtype=torch.long).to(device),
        }

        with torch.no_grad():
            correct_loss = model(**correct_inputs).loss.item()
            wrong_loss = model(**wrong_inputs).loss.item()

        tr = np.exp(-wrong_loss) / (np.exp(-correct_loss) + 1e-10)
        truth_ratios.append(float(tr))

        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx+1}/{len(forget_hf)}")

    # Save
    output = {
        "model": model_name,
        "dataset": "locuslab/TOFU forget10",
        "num_samples": len(truth_ratios),
        "truth_ratios": truth_ratios,
    }
    output_path = "retain_reference_logs.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved {len(truth_ratios)} truth ratios to {output_path}")
    print(f"Mean truth ratio: {np.mean(truth_ratios):.4f}")
    print(f"Std truth ratio: {np.std(truth_ratios):.4f}")


if __name__ == '__main__':
    main()
