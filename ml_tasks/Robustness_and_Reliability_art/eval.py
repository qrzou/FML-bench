# Comprehensive Backdoor Attack Evaluation
# Uses multiple diverse backdoor types to prevent defenses from gaming the benchmark
# This makes it harder to achieve high defense scores through simple tricks

from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import json
import os
import sys
import warnings

import numpy as np
import tensorflow as tf

# Avoiding consuming all GPU memory
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# set module path
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# ignore warnings
warnings.filterwarnings('ignore')

# disable TensorFlow eager execution
if tf.executing_eagerly():
    tf.compat.v1.disable_eager_execution()

from art.attacks.poisoning import PoisoningAttackBackdoor
from art.attacks.poisoning.perturbations import add_pattern_bd, add_single_bd
from art.estimators.classification import KerasClassifier
from art.utils import load_mnist, preprocess

from model import get_model
from trainer import get_trainer


# ---------- Advanced Backdoor Functions ----------

def add_blended_trigger(x, alpha=0.5, pixel_value=1.0):
    """
    Blended backdoor - trigger blended into image.
    IMPROVED: Increased alpha from 0.15 to 0.5 to survive Laplacian noise (scale=0.3).
    """
    x = np.array(x).astype(float)
    shape = x.shape

    if len(shape) == 4:
        width, height = x.shape[1:3]
        # Small checkerboard pattern in bottom-right, but blended not discrete
        for i in range(width - 6, width - 2):
            for j in range(height - 6, height - 2):
                if (i + j) % 2 == 0:
                    x[:, i, j, :] = (1 - alpha) * x[:, i, j, :] + alpha * pixel_value
    elif len(shape) == 3:
        width, height = x.shape[1:]
        for i in range(width - 6, width - 2):
            for j in range(height - 6, height - 2):
                if (i + j) % 2 == 0:
                    x[:, i, j] = (1 - alpha) * x[:, i, j] + alpha * pixel_value
    elif len(shape) == 2:
        width, height = x.shape
        for i in range(width - 6, width - 2):
            for j in range(height - 6, height - 2):
                if (i + j) % 2 == 0:
                    x[i, j] = (1 - alpha) * x[i, j] + alpha * pixel_value

    return x


def add_distributed_trigger(x, pixel_value=1.0):
    """
    Distributed trigger - 4-pixel patterns spread across multiple regions.
    IMPROVED: Changed from single pixels to 4-pixel patterns at each location for stronger signal.
    Not concentrated in corners, so corner-checking defenses will fail.
    """
    x = np.array(x)
    shape = x.shape

    if len(shape) == 4:
        width, height = x.shape[1:3]
        # Place 4-pixel patterns in 4 different locations (diverse spatial distribution)
        positions = [
            (2, 2),                        # top-left
            (width - 4, 2),                # top-right
            (2, height - 4),               # bottom-left
            (width // 2, height // 2),     # center
        ]
        for w, h in positions:
            # Add a small 2x2 pattern at each location
            if w + 1 < width and h + 1 < height:
                x[:, w, h, :] = pixel_value
                x[:, w + 1, h, :] = pixel_value
                x[:, w, h + 1, :] = pixel_value
                x[:, w + 1, h + 1, :] = pixel_value
    elif len(shape) == 3:
        width, height = x.shape[1:]
        positions = [
            (2, 2),
            (width - 4, 2),
            (2, height - 4),
            (width // 2, height // 2),
        ]
        for w, h in positions:
            if w + 1 < width and h + 1 < height:
                x[:, w, h] = pixel_value
                x[:, w + 1, h] = pixel_value
                x[:, w, h + 1] = pixel_value
                x[:, w + 1, h + 1] = pixel_value
    elif len(shape) == 2:
        width, height = x.shape
        positions = [
            (2, 2),
            (width - 4, 2),
            (2, height - 4),
            (width // 2, height // 2),
        ]
        for w, h in positions:
            if w + 1 < width and h + 1 < height:
                x[w, h] = pixel_value
                x[w + 1, h] = pixel_value
                x[w, h + 1] = pixel_value
                x[w + 1, h + 1] = pixel_value

    return x


def add_edge_trigger(x, pixel_value=1.0):
    """
    Edge trigger - placed along image edges (top and right).
    Different from corner patterns, tests if defense checks edges.
    """
    x = np.array(x)
    shape = x.shape

    if len(shape) == 4:
        width, height = x.shape[1:3]
        # Top edge pattern
        x[:, 0, height//4:3*height//4:2, :] = pixel_value
        # Right edge pattern
        x[:, width//4:3*width//4:2, height-1, :] = pixel_value
    elif len(shape) == 3:
        width, height = x.shape[1:]
        x[:, 0, height//4:3*height//4:2] = pixel_value
        x[:, width//4:3*width//4:2, height-1] = pixel_value
    elif len(shape) == 2:
        width, height = x.shape
        x[0, height//4:3*height//4:2] = pixel_value
        x[width//4:3*width//4:2, height-1] = pixel_value

    return x


def add_random_corner_pattern(x, pixel_value=1.0, seed=None):
    """
    Random corner pattern - places checkerboard pattern in a randomly selected corner.
    IMPROVED: Adds positional randomization to prevent corner-bias exploitation.
    """
    if seed is not None:
        np.random.seed(seed)

    x = np.array(x)
    shape = x.shape

    if len(shape) == 4:
        width, height = x.shape[1:3]
        # Randomly choose a corner
        corner = np.random.choice(['top-left', 'top-right', 'bottom-left', 'bottom-right'])

        if corner == 'top-left':
            x[:, 2, 2, :] = pixel_value
            x[:, 3, 3, :] = pixel_value
            x[:, 2, 4, :] = pixel_value
            x[:, 4, 2, :] = pixel_value
        elif corner == 'top-right':
            x[:, 2, height - 3, :] = pixel_value
            x[:, 3, height - 4, :] = pixel_value
            x[:, 2, height - 5, :] = pixel_value
            x[:, 4, height - 3, :] = pixel_value
        elif corner == 'bottom-left':
            x[:, width - 3, 2, :] = pixel_value
            x[:, width - 4, 3, :] = pixel_value
            x[:, width - 3, 4, :] = pixel_value
            x[:, width - 5, 2, :] = pixel_value
        else:  # bottom-right
            x[:, width - 3, height - 3, :] = pixel_value
            x[:, width - 4, height - 4, :] = pixel_value
            x[:, width - 3, height - 5, :] = pixel_value
            x[:, width - 5, height - 3, :] = pixel_value
    elif len(shape) == 3:
        width, height = x.shape[1:]
        corner = np.random.choice(['top-left', 'top-right', 'bottom-left', 'bottom-right'])

        if corner == 'top-left':
            x[:, 2, 2] = pixel_value
            x[:, 3, 3] = pixel_value
            x[:, 2, 4] = pixel_value
            x[:, 4, 2] = pixel_value
        elif corner == 'top-right':
            x[:, 2, height - 3] = pixel_value
            x[:, 3, height - 4] = pixel_value
            x[:, 2, height - 5] = pixel_value
            x[:, 4, height - 3] = pixel_value
        elif corner == 'bottom-left':
            x[:, width - 3, 2] = pixel_value
            x[:, width - 4, 3] = pixel_value
            x[:, width - 3, 4] = pixel_value
            x[:, width - 5, 2] = pixel_value
        else:  # bottom-right
            x[:, width - 3, height - 3] = pixel_value
            x[:, width - 4, height - 4] = pixel_value
            x[:, width - 3, height - 5] = pixel_value
            x[:, width - 5, height - 3] = pixel_value
    elif len(shape) == 2:
        width, height = x.shape
        corner = np.random.choice(['top-left', 'top-right', 'bottom-left', 'bottom-right'])

        if corner == 'top-left':
            x[2, 2] = pixel_value
            x[3, 3] = pixel_value
            x[2, 4] = pixel_value
            x[4, 2] = pixel_value
        elif corner == 'top-right':
            x[2, height - 3] = pixel_value
            x[3, height - 4] = pixel_value
            x[2, height - 5] = pixel_value
            x[4, height - 3] = pixel_value
        elif corner == 'bottom-left':
            x[width - 3, 2] = pixel_value
            x[width - 4, 3] = pixel_value
            x[width - 3, 4] = pixel_value
            x[width - 5, 2] = pixel_value
        else:  # bottom-right
            x[width - 3, height - 3] = pixel_value
            x[width - 4, height - 4] = pixel_value
            x[width - 3, height - 5] = pixel_value
            x[width - 5, height - 3] = pixel_value

    return x


# ---------- Set Seed ----------
import random

def set_seed(seed=42):
    """Set seed for reproducibility across Python, NumPy, and TensorFlow."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seed(seed=233)


def split_test_stratified(y, test_fraction, seed):
    """Split indices into (visible, hidden_test), stratified by class. FIXED, ignores split_config."""
    rng = np.random.RandomState(seed)
    visible_indices = []
    hidden_indices = []
    for class_idx in range(10):
        class_mask = np.where(y == class_idx)[0]
        shuffled = rng.permutation(class_mask)
        n_hidden = int(test_fraction * len(shuffled))
        n_visible = len(shuffled) - n_hidden
        visible_indices.extend(shuffled[:n_visible])
        hidden_indices.extend(shuffled[n_visible:])
    return np.array(visible_indices), np.array(hidden_indices)


def load_split_config(path="split_config.json"):
    """Load agent-controlled split config. Returns defaults if file missing."""
    defaults = {"val_ratio": 0.20, "val_seed": 42}
    if os.path.exists(path):
        with open(path) as f:
            cfg = json.load(f)
        defaults.update(cfg)
    return defaults


# Helper function for top-left pattern
def _add_pattern_topleft(x, pixel_value=1.0):
    """Add 4-pixel checkerboard pattern in top-left corner."""
    x = np.array(x)
    if len(x.shape) == 4:
        x[:, 2, 2, :] = pixel_value
        x[:, 3, 3, :] = pixel_value
        x[:, 2, 4, :] = pixel_value
        x[:, 4, 2, :] = pixel_value
    elif len(x.shape) == 3:
        x[:, 2, 2] = pixel_value
        x[:, 3, 3] = pixel_value
        x[:, 2, 4] = pixel_value
        x[:, 4, 2] = pixel_value
    elif len(x.shape) == 2:
        x[2, 2] = pixel_value
        x[3, 3] = pixel_value
        x[2, 4] = pixel_value
        x[4, 2] = pixel_value
    return x


# ---------- Parse Arguments ----------
parser = argparse.ArgumentParser()
parser.add_argument('--split', choices=['val', 'test'], required=True)
args = parser.parse_args()


# ---------- Prepare Data ----------

# Load data
(x_raw_train, y_raw_train), (x_raw_test, y_raw_test), min_, max_ = load_mnist(raw=True)

# ===== Step 1: Carve hidden test (70% of test = 7K, FIXED seed=42, stratified) =====
# y_raw_test is one-hot from ART; get class labels for stratification
y_test_labels = np.argmax(y_raw_test, axis=1) if y_raw_test.ndim > 1 else y_raw_test
visible_test_idx, hidden_test_idx = split_test_stratified(
    y_test_labels, test_fraction=0.70, seed=42
)

hidden_test_x = x_raw_test[hidden_test_idx]
hidden_test_y = y_raw_test[hidden_test_idx]
visible_test_x = x_raw_test[visible_test_idx]
visible_test_y = y_raw_test[visible_test_idx]

# ===== Step 2: Select training samples (7500 from train, same as before) =====
n_train = np.shape(x_raw_train)[0]
num_selection = 7500
train_select_rng = np.random.RandomState(233)
random_selection_indices = train_select_rng.choice(n_train, num_selection, replace=False)
selected_train_x = x_raw_train[random_selection_indices]
selected_train_y = y_raw_train[random_selection_indices]

# ===== Step 3: Visible pool = selected train (7.5K) + visible test portion (3K) = 10.5K =====
visible_pool_x = np.concatenate([selected_train_x, visible_test_x], axis=0)
visible_pool_y = np.concatenate([selected_train_y, visible_test_y], axis=0)

print(f"Visible pool: {len(visible_pool_x)}, Hidden test: {len(hidden_test_x)}")


# ---------- Define Multiple Backdoor Types ----------
max_val = np.max(visible_pool_x)

BACKDOOR_CONFIGS = [
    {
        'name': 'pattern',
        'func': lambda x: add_pattern_bd(x, pixel_value=max_val),
        'description': 'Classic 4-pixel checkerboard in bottom-right corner'
    },
    {
        'name': 'distributed',
        'func': lambda x: add_distributed_trigger(x, pixel_value=max_val),
        'description': 'Distributed 2x2 patterns in 4 regions (not corner-focused)'
    },
    {
        'name': 'edge',
        'func': lambda x: add_edge_trigger(x, pixel_value=max_val),
        'description': 'Edge-based trigger on top and right borders'
    },
    {
        'name': 'random_corner',
        'func': lambda x: add_random_corner_pattern(x, pixel_value=max_val, seed=233),
        'description': 'Checkerboard in random corner (prevents fixed-position bias)'
    },
    {
        'name': 'pattern_topleft',
        'func': lambda x: _add_pattern_topleft(x, pixel_value=max_val),
        'description': 'Checkerboard in top-left corner (tests positional invariance)'
    },
]


# ---------- Poison Dataset Function ----------

def poison_dataset(x_clean, y_clean, percent_poison, poison_func, rng=None):
    """
    Create poisoned dataset using given poison function.
    Uses all-to-all attack pattern (0->1, 1->2, ..., 9->0).
    """
    if rng is None:
        rng = np.random.RandomState(233)
    x_poison = np.copy(x_clean)
    y_poison = np.copy(y_clean)
    is_poison = np.zeros(np.shape(y_poison)[0] if y_poison.ndim == 1 else np.shape(y_poison)[0])

    # Get class labels for source/target mapping
    y_labels = np.argmax(y_clean, axis=1) if y_clean.ndim > 1 else y_clean

    sources = np.arange(10)
    targets = (np.arange(10) + 1) % 10

    for i, (src, tgt) in enumerate(zip(sources, targets)):
        n_points_in_tgt = np.size(np.where(y_labels == tgt))
        num_poison = round((percent_poison * n_points_in_tgt) / (1 - percent_poison))
        src_imgs = x_clean[y_labels == src]

        n_points_in_src = np.shape(src_imgs)[0]
        indices_to_be_poisoned = rng.choice(n_points_in_src, num_poison)

        imgs_to_be_poisoned = np.copy(src_imgs[indices_to_be_poisoned])
        backdoor_attack = PoisoningAttackBackdoor(poison_func)
        imgs_to_be_poisoned, poison_labels = backdoor_attack.poison(imgs_to_be_poisoned, y=np.ones(num_poison) * tgt)
        x_poison = np.append(x_poison, imgs_to_be_poisoned, axis=0)
        y_poison = np.append(y_poison, poison_labels, axis=0)
        is_poison = np.append(is_poison, np.ones(num_poison))

    is_poison = is_poison != 0

    return is_poison, x_poison, y_poison


# ---------- Run Evaluation for Each Backdoor Type ----------

percent_poison = 0.33
all_results = []
CHECKPOINT_DIR = 'model_checkpoint'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

if args.split == 'val':
    # ===== Step 4a: Read split_config, split visible pool into agent_train / agent_val =====
    cfg = load_split_config()
    n_visible = len(visible_pool_x)
    rng_split = np.random.RandomState(cfg["val_seed"])
    split_perm = rng_split.permutation(n_visible)
    n_agent_val = int(cfg["val_ratio"] * n_visible)
    agent_val_idx = split_perm[:n_agent_val]
    agent_train_idx = split_perm[n_agent_val:]

    agent_train_x = visible_pool_x[agent_train_idx]
    agent_train_y = visible_pool_y[agent_train_idx]
    agent_val_x = visible_pool_x[agent_val_idx]
    agent_val_y = visible_pool_y[agent_val_idx]

    print(f"Agent train: {len(agent_train_x)}, Agent val: {len(agent_val_x)}")
    print(f"split_config: val_ratio={cfg['val_ratio']}, val_seed={cfg['val_seed']}")

    eval_x_raw = agent_val_x
    eval_y_raw = agent_val_y
    train_x_raw = agent_train_x
    train_y_raw = agent_train_y
else:
    # ===== Step 4b: Test — use hidden test, no split_config needed =====
    eval_x_raw = hidden_test_x
    eval_y_raw = hidden_test_y
    # train data not used for test (load checkpoints), but define for consistency
    train_x_raw = None
    train_y_raw = None

print(f"\n{'='*80}")
print(f"COMPREHENSIVE BACKDOOR ATTACK EVALUATION")
print(f"Training samples: {len(train_x_raw) if train_x_raw is not None else 'N/A (loading checkpoints)'}, Poison rate: {percent_poison}")
print(f"Testing {len(BACKDOOR_CONFIGS)} different backdoor types")
print(f"{'='*80}\n")

for idx, backdoor_config in enumerate(BACKDOOR_CONFIGS):
    backdoor_name = backdoor_config['name']
    backdoor_func = backdoor_config['func']
    backdoor_desc = backdoor_config['description']

    print(f"\n{'#'*80}")
    print(f"Backdoor Type {idx+1}/{len(BACKDOOR_CONFIGS)}: {backdoor_name}")
    print(f"Description: {backdoor_desc}")
    print(f"{'#'*80}\n")

    # Use isolated RNGs per backdoor type to ensure reproducibility
    train_rng = np.random.RandomState(233 + idx * 2)
    test_rng = np.random.RandomState(233 + idx * 2 + 1)
    shuffle_rng = np.random.RandomState(233 + idx * 3)

    # Poison eval data (agent_val for val, hidden_test for test)
    # Poisoning eval data is needed to measure attack success rate
    (is_poison_eval, x_poisoned_eval, y_poisoned_eval) = poison_dataset(
        eval_x_raw, eval_y_raw, percent_poison, backdoor_func, rng=test_rng
    )
    x_eval, y_eval = preprocess(x_poisoned_eval, y_poisoned_eval)
    x_eval = np.expand_dims(x_eval, axis=3)

    # Reset seed before model creation (TF deterministic mode)
    set_seed(seed=233)

    # Train model with defense (or load from checkpoint)
    ckpt_path = os.path.join(CHECKPOINT_DIR, f'model_{backdoor_name}.h5')
    model = get_model()
    classifier = KerasClassifier(model=model, clip_values=(0, 1))

    if args.split == 'test' and os.path.exists(ckpt_path):
        model.load_weights(ckpt_path)
        print(f"Loaded pre-trained model from {ckpt_path} (skipping training)")
    else:
        # Poison only agent_train for training (NOT agent_val)
        (is_poison_train, x_poisoned_train, y_poisoned_train) = poison_dataset(
            train_x_raw, train_y_raw, percent_poison, backdoor_func, rng=train_rng
        )
        x_train, y_train = preprocess(x_poisoned_train, y_poisoned_train)
        x_train = np.expand_dims(x_train, axis=3)

        # Shuffle training data
        n_train_samples = np.shape(y_train)[0]
        shuffled_indices = np.arange(n_train_samples)
        shuffle_rng.shuffle(shuffled_indices)
        x_train = x_train[shuffled_indices]
        y_train = y_train[shuffled_indices]

        trainer = get_trainer(classifier)
        print(f"Training model with DP-InstaHide defense...")
        trainer.fit(x_train, y_train, nb_epochs=5, batch_size=128)
        model.save_weights(ckpt_path)
        print(f"Saved trained model to {ckpt_path}")

    # Evaluate on clean eval set
    clean_x_eval = x_eval[is_poison_eval == 0]
    clean_y_eval = y_eval[is_poison_eval == 0]

    clean_preds = np.argmax(classifier.predict(clean_x_eval), axis=1)
    clean_correct = np.sum(clean_preds == np.argmax(clean_y_eval, axis=1))
    clean_total = clean_y_eval.shape[0]
    clean_acc = clean_correct / clean_total

    # Evaluate on poisoned eval set
    poison_x_eval = x_eval[is_poison_eval]
    poison_y_eval = y_eval[is_poison_eval]

    poison_preds = np.argmax(classifier.predict(poison_x_eval), axis=1)
    poison_correct = np.sum(poison_preds == np.argmax(poison_y_eval, axis=1))
    poison_total = poison_y_eval.shape[0]
    poison_acc = poison_correct / poison_total

    # Compute defense metrics
    resist_acc = 1 - poison_acc
    if clean_acc + resist_acc == 0:
        defense_score = 0.0
    else:
        defense_score = 2 * clean_acc * resist_acc / (clean_acc + resist_acc)

    print(f"\nResults for {backdoor_name}:")
    print(f"  Clean Accuracy:    {clean_acc*100:.2f}%")
    print(f"  Attack Success:    {poison_acc*100:.2f}%")
    print(f"  Resist Accuracy:   {resist_acc*100:.2f}%")
    print(f"  Defense Score:     {defense_score*100:.2f}%")

    # Store results
    all_results.append({
        'backdoor_type': backdoor_name,
        'clean_acc': clean_acc,
        'poison_acc': poison_acc,
        'resist_acc': resist_acc,
        'defense_score': defense_score
    })

    # Clean up to free memory
    del model, classifier
    tf.keras.backend.clear_session()


# ---------- Compute Aggregate Metrics ----------

clean_accs = [r['clean_acc'] for r in all_results]
poison_accs = [r['poison_acc'] for r in all_results]
resist_accs = [r['resist_acc'] for r in all_results]
defense_scores = [r['defense_score'] for r in all_results]

# Mean across all backdoor types
mean_clean_acc = np.mean(clean_accs)
mean_poison_acc = np.mean(poison_accs)
mean_resist_acc = np.mean(resist_accs)
mean_defense_score = np.mean(defense_scores)

# Worst case (minimum defense score, maximum attack success)
worst_defense_score = np.min(defense_scores)
worst_clean_acc = np.min(clean_accs)
max_attack_success = np.max(poison_accs)

print(f"\n\n{'='*80}")
print(f"AGGREGATE RESULTS ACROSS ALL BACKDOOR TYPES")
print(f"{'='*80}")
print(f"\nMean Metrics:")
print(f"  Clean Accuracy:    {mean_clean_acc*100:.2f}%")
print(f"  Attack Success:    {mean_poison_acc*100:.2f}%")
print(f"  Resist Accuracy:   {mean_resist_acc*100:.2f}%")
print(f"  Defense Score:     {mean_defense_score*100:.2f}%")
print(f"\nWorst-Case Metrics:")
print(f"  Min Defense Score: {worst_defense_score*100:.2f}%")
print(f"  Min Clean Acc:     {worst_clean_acc*100:.2f}%")
print(f"  Max Attack Success:{max_attack_success*100:.2f}%")
print(f"\nIndividual Results:")
for r in all_results:
    print(f"  {r['backdoor_type']:12s} - Defense Score: {r['defense_score']*100:.2f}%")


# ---------- Save to JSON ----------

results = {
    "poisoned_mnist": {
        "means": {
            "clean_acc_mean": mean_clean_acc,
            "poison_acc_mean": mean_poison_acc,
            "resist_acc_mean": mean_resist_acc,
            "defense_score_mean": mean_defense_score,
        },
        "stderrs": {
            "clean_acc_stderr": 0.0,
            "poison_acc_stderr": 0.0,
            "resist_acc_stderr": 0.0,
            "defense_score_stderr": 0.0,
        },
        "final_info_dict": {
            "clean_acc": mean_clean_acc,
            "poison_acc": mean_poison_acc,
            "resist_acc": mean_resist_acc,
            "defense_score": mean_defense_score,
        },
        "worst_case": {
            "defense_score_min": worst_defense_score,
            "clean_acc_min": worst_clean_acc,
            "poison_acc_max": max_attack_success,
        },
        "all_backdoor_results": {
            "clean_acc": clean_accs,
            "poison_acc": poison_accs,
            "resist_acc": resist_accs,
            "defense_score": defense_scores,
        },
        "per_backdoor_results": all_results,
        "config": {
            "num_backdoor_types": len(BACKDOOR_CONFIGS),
            "backdoor_types": [c['name'] for c in BACKDOOR_CONFIGS],
            "num_samples": len(train_x_raw) if train_x_raw is not None else 0,
            "poison_rate": percent_poison,
            "note": "Comprehensive evaluation with multiple diverse backdoor types"
        }
    }
}

from pathlib import Path
output_path = Path(f"results_tmp/{args.split}_info.json")
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n\nSaved results to {output_path}")
print(f"{'='*80}\n")
