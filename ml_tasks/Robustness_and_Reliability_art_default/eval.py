# This code is based on  https://github.com/Trusted-AI/adversarial-robustness-toolbox: notebooks/poisoning_defense_dp_instahide.ipynb

from __future__ import absolute_import, division, print_function, unicode_literals
import json
import os
import pprint
import sys
import warnings

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout


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
from art.attacks.poisoning.perturbations import add_pattern_bd, add_single_bd, insert_image
from art.estimators.classification import KerasClassifier
from art.utils import load_mnist, preprocess

from model import get_model
from trainer import get_trainer


# import logging
# logger = logging.getLogger()
# logger.setLevel(logging.INFO)
# handler = logging.StreamHandler()
# formatter = logging.Formatter("[%(levelname)s] %(message)s")
# handler.setFormatter(formatter)
# logger.addHandler(handler)


# ---------- Set Seed ----------
import random
def set_seed(seed=42):
    """Set seed for reproducibility across Python, NumPy, and TensorFlow."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'  # Enforce deterministic behavior in TF (as much as possible)

    random.seed(seed)  # Python's built-in random module
    np.random.seed(seed)  # NumPy
    tf.random.set_seed(seed)  # TensorFlow

set_seed(seed=233)

# ---------- Prepare Data ----------

# Load data
(x_raw, y_raw), (x_raw_test, y_raw_test), min_, max_ = load_mnist(raw=True)

# Random Selection:
n_train = np.shape(x_raw)[0]
num_selection = 7500
random_selection_indices = np.random.choice(n_train, num_selection)
x_raw = x_raw[random_selection_indices]
y_raw = y_raw[random_selection_indices]

BACKDOOR_TYPE = "pattern" # one of ['pattern', 'pixel', 'image']

max_val = np.max(x_raw)
def add_modification(x):
    if BACKDOOR_TYPE == 'pattern':
        return add_pattern_bd(x, pixel_value=max_val)
    elif BACKDOOR_TYPE == 'pixel':
        return add_single_bd(x, pixel_value=max_val) 
    elif BACKDOOR_TYPE == 'image':
        return insert_image(x, backdoor_path='../utils/data/backdoors/alert.png', size=(10, 10))
    else:
        raise("Unknown backdoor type")

def poison_dataset(x_clean, y_clean, percent_poison, poison_func):
    x_poison = np.copy(x_clean)
    y_poison = np.copy(y_clean)
    is_poison = np.zeros(np.shape(y_poison))
    
    sources = np.arange(10) # 0, 1, 2, 3, ...
    targets = (np.arange(10) + 1) % 10 # 1, 2, 3, 4, ...
    for i, (src, tgt) in enumerate(zip(sources, targets)):
        n_points_in_tgt = np.size(np.where(y_clean == tgt))
        num_poison = round((percent_poison * n_points_in_tgt) / (1 - percent_poison))
        src_imgs = x_clean[y_clean == src]

        n_points_in_src = np.shape(src_imgs)[0]
        indices_to_be_poisoned = np.random.choice(n_points_in_src, num_poison)

        imgs_to_be_poisoned = np.copy(src_imgs[indices_to_be_poisoned])
        backdoor_attack = PoisoningAttackBackdoor(poison_func)
        imgs_to_be_poisoned, poison_labels = backdoor_attack.poison(imgs_to_be_poisoned, y=np.ones(num_poison) * tgt)
        x_poison = np.append(x_poison, imgs_to_be_poisoned, axis=0)
        y_poison = np.append(y_poison, poison_labels, axis=0)
        is_poison = np.append(is_poison, np.ones(num_poison))

    is_poison = is_poison != 0

    return is_poison, x_poison, y_poison

# Poison training data
percent_poison = 0.33
(is_poison_train, x_poisoned_raw, y_poisoned_raw) = poison_dataset(x_raw, y_raw, percent_poison, add_modification)
x_train, y_train = preprocess(x_poisoned_raw, y_poisoned_raw)
# Add channel axis:
x_train = np.expand_dims(x_train, axis=3)

# Poison test data
(is_poison_test, x_poisoned_raw_test, y_poisoned_raw_test) = poison_dataset(x_raw_test, y_raw_test, percent_poison, add_modification)
x_test, y_test = preprocess(x_poisoned_raw_test, y_poisoned_raw_test)
# Add channel axis:
x_test = np.expand_dims(x_test, axis=3)

# Shuffle training data
n_train = np.shape(y_train)[0]
shuffled_indices = np.arange(n_train)
np.random.shuffle(shuffled_indices)
x_train = x_train[shuffled_indices]
y_train = y_train[shuffled_indices]
is_poison_train = is_poison_train[shuffled_indices]





# ---------- Prepare & Train Model ----------
model = get_model()
classifier = KerasClassifier(model=model, clip_values=(0, 1))

# Use DP-InstaHide as baseline to train the model, to defend against attacks
trainer = get_trainer(classifier)
trainer.fit(x_train, y_train, nb_epochs=5, batch_size=128)



# ---------- Evaluate Model ----------
## --------- Clean Test Set ---------
clean_x_test = x_test[is_poison_test == 0]
clean_y_test = y_test[is_poison_test == 0]

clean_preds = np.argmax(classifier.predict(clean_x_test), axis=1)
clean_correct = np.sum(clean_preds == np.argmax(clean_y_test, axis=1))
clean_total = clean_y_test.shape[0]

clean_acc = clean_correct / clean_total
print("\nClean test set accuracy: %.2f%%" % (clean_acc * 100))

## --------- Poisoned Test Set ---------
poison_x_test = x_test[is_poison_test]
poison_y_test = y_test[is_poison_test]

poison_preds = np.argmax(classifier.predict(poison_x_test), axis=1)
poison_correct = np.sum(poison_preds == np.argmax(poison_y_test, axis=1))
poison_total = poison_y_test.shape[0]

poison_acc = poison_correct / poison_total
print("\nEffectiveness of poison: %.2f%%" % (poison_acc * 100))

## --------- Compute Defense Score ---------
resist_acc = 1 - poison_acc
print("\nResist accuracy of poison: %.2f%%" % (resist_acc * 100))

if clean_acc + resist_acc == 0:
    defense_score = 0.0
else:
    defense_score = 2 * clean_acc * resist_acc / (clean_acc + resist_acc)
print("\nDefense score: %.2f%%" % (defense_score * 100))



# ---------- Save to JSON ----------
results = {
    "poisoned_mnist": {
        "means": {
            "clean_acc_mean": clean_acc,
            "poison_acc_mean": poison_acc,
            "resist_acc_mean": resist_acc,
            "defense_score_mean": defense_score,
        },
        "stderrs": {
            "clean_acc_stderr": 0.0,
            "poison_acc_stderr": 0.0,
            "resist_acc_stderr": 0.0,
            "defense_score_stderr": 0.0,
        },
        "final_info_dict": {
            "clean_acc": [clean_acc],
            "poison_acc": [poison_acc],
            "resist_acc": [resist_acc],
            "defense_score": [defense_score],
        }
    }
}

from pathlib import Path
output_path = Path("results_tmp/final_info.json")
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved results to {output_path}")

