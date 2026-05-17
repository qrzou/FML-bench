"""
Baseline machine unlearning algorithm: Gradient Ascent on TOFU.

Gradient Ascent is the simplest unlearning method — it negates the
cross-entropy loss on the forget set, effectively teaching the model
to forget by gradient ascending on the forget data.

Model: Llama-3.2-1B-Instruct (fine-tuned on TOFU)
Dataset: TOFU forget10 (400 QA pairs) / retain90 (3600 QA pairs)

Agents should modify this file to improve unlearning quality:
- GradDiff: combine forget ascent with retain descent
  (loss = -forget_loss + alpha * retain_loss)
- NPO: Negative Preference Optimization (DPO loss on forget set)
- DPO: Direct Preference Optimization with "I don't know" responses
- KL regularization: penalize divergence from original model
- SimNPO: Simplified NPO without reference model
- RMU: Representation engineering based unlearning
- Custom loss functions and training schedules
"""
import torch


def get_unlearning_config():
    """Return unlearning hyperparameters."""
    return {
        'method': 'grad_ascent',
        'epochs': 10,
        'lr': 1e-5,
        'batch_size': 8,
        'gradient_accumulation_steps': 4,
        'weight_decay': 0.01,
        'warmup_ratio': 0.1,
        'retain_loss_weight': 0.0,  # GradAscent ignores retain set
        'max_grad_norm': 1.0,
    }


def compute_unlearn_loss(model, forget_inputs, retain_inputs=None):
    """
    Compute the unlearning loss.

    Baseline (GradAscent): negate cross-entropy on forget set.

    Args:
        model: the language model being unlearned
        forget_inputs: dict with 'input_ids', 'attention_mask', 'labels'
                       from the forget set
        retain_inputs: dict with 'input_ids', 'attention_mask', 'labels'
                       from the retain set (ignored by baseline)

    Returns:
        loss: scalar tensor for backpropagation
    """
    config = get_unlearning_config()

    # Forget loss: negate the standard LM loss
    forget_outputs = model(
        input_ids=forget_inputs['input_ids'],
        attention_mask=forget_inputs['attention_mask'],
        labels=forget_inputs['labels'],
    )
    forget_loss = -forget_outputs.loss

    total_loss = forget_loss

    # Retain loss (weight=0 for baseline, agents can increase)
    if retain_inputs is not None and config['retain_loss_weight'] > 0:
        retain_outputs = model(
            input_ids=retain_inputs['input_ids'],
            attention_mask=retain_inputs['attention_mask'],
            labels=retain_inputs['labels'],
        )
        retain_loss = retain_outputs.loss
        total_loss = forget_loss + config['retain_loss_weight'] * retain_loss

    return total_loss
