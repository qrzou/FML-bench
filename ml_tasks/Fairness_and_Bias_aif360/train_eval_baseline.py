from aif360.metrics import ClassificationMetric
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult
from aif360.algorithms.inprocessing.adversarial_debiasing import AdversarialDebiasing
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


# Avoiding consuming all GPU memory
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


# ignore warnings
import warnings
warnings.filterwarnings('ignore')

# disable TensorFlow eager execution
if tf.executing_eagerly():
    print("Eager execution is enabled, we will disable it")
    tf.compat.v1.disable_eager_execution()


# ---------- Set Seed ----------
import numpy as np
import random
import os
import tensorflow
def set_seed(seed=42):
    """Set seed for reproducibility across Python, NumPy, and TensorFlow."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'  # Enforce deterministic behavior in TF (as much as possible)

    random.seed(seed)  # Python's built-in random module
    np.random.seed(seed)  # NumPy
    tensorflow.random.set_seed(seed)  # TensorFlow

set_seed(seed=233)


# ---------- Load Dataset and set options ----------
# Get the dataset and split into train and test
dataset_orig = load_preproc_data_adult()

privileged_groups = [{'sex': 1}]
unprivileged_groups = [{'sex': 0}]

dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)


print("### Training Dataset shape:", dataset_orig_train.features.shape)
print("### Favorable and unfavorable labels:", dataset_orig_train.favorable_label, dataset_orig_train.unfavorable_label)
print("### Protected attribute names:", dataset_orig_train.protected_attribute_names)
print("### Privileged and unprivileged protected attribute values:")
print(dataset_orig_train.privileged_protected_attributes, 
      dataset_orig_train.unprivileged_protected_attributes)
print("### Dataset feature names", dataset_orig_train.feature_names)



# ---------- Apply in-processing algorithm based on adversarial learning ----------
sess = tf.Session()
sess.close()
tf.reset_default_graph()
sess = tf.Session()

# Learn parameters with debias set to True
debiased_model = AdversarialDebiasing(privileged_groups = privileged_groups,
                          unprivileged_groups = unprivileged_groups,
                          scope_name='debiased_classifier',
                          debias=True,
                          sess=sess)

# Fit the model
debiased_model.fit(dataset_orig_train)

# Apply the debiased model to test data
dataset_debiasing_train = debiased_model.predict(dataset_orig_train)
dataset_debiasing_test = debiased_model.predict(dataset_orig_test)



print("#### Model - with debiasing - classification metrics")
classified_metric_debiasing_test = ClassificationMetric(dataset_orig_test, 
                                                 dataset_debiasing_test,
                                                 unprivileged_groups=unprivileged_groups,
                                                 privileged_groups=privileged_groups)
print("Test set: Classification accuracy = %f" % classified_metric_debiasing_test.accuracy())
TPR = classified_metric_debiasing_test.true_positive_rate()
TNR = classified_metric_debiasing_test.true_negative_rate()
bal_acc_debiasing_test = 0.5*(TPR+TNR)
print("Test set: Balanced classification accuracy = %f" % bal_acc_debiasing_test)
disparate_impact = classified_metric_debiasing_test.disparate_impact()
abs_disparate_impact_diff = abs(disparate_impact - 1)
print("Test set: Disparate impact = %f" % disparate_impact)
print("Test set: Absolute disparate impact difference from 1 = %f" % abs_disparate_impact_diff)
eod = classified_metric_debiasing_test.equal_opportunity_difference()
abs_eod = abs(eod)
print("Test set: Equal opportunity difference = %f" % eod)
print("Test set: Absolute equal opportunity difference = %f" % abs_eod)
aod = classified_metric_debiasing_test.average_odds_difference()
abs_aod = abs(aod)
print("Test set: Average odds difference = %f" % aod)
print("Test set: Absolute average odds difference = %f" % abs_aod)
theil_index = classified_metric_debiasing_test.theil_index()
abs_theil_index = abs(theil_index)
print("Test set: Theil_index = %f" % theil_index)
print("Test set: Absolute Theil_index = %f" % abs_theil_index)


# ---------- Save to JSON ----------
results = {
    "adult": {
        "means": {
            "cls_acc_mean": classified_metric_debiasing_test.accuracy(),
            "balanced_cls_acc_mean": bal_acc_debiasing_test,
            "disparate_impact_mean": disparate_impact,
            "abs_disparate_impact_diff_mean": abs_disparate_impact_diff,
            "eod_mean": eod,
            "abs_eod_mean": abs_eod,
            "aod_mean": aod,
            "abs_aod_mean": abs_aod,
            "theil_index_mean": theil_index,
            "abs_theil_index_mean": abs_theil_index,
        },
        "stderrs": {
            "cls_acc_stderr": 0.0,
            "balanced_cls_acc_stderr": 0.0,
            "disparate_impact_stderr": 0.0,
            "abs_disparate_impact_diff_stderr": 0.0,
            "eod_stderr": 0.0,
            "abs_eod_stderr": 0.0,
            "aod_stderr": 0.0,
            "abs_aod_stderr": 0.0,
            "theil_index_stderr": 0.0,
            "abs_theil_index_stderr": 0.0,
        },
        "final_info_dict": {
            "cls_acc": classified_metric_debiasing_test.accuracy(),
            "balanced_cls_acc": bal_acc_debiasing_test,
            "disparate_impact": disparate_impact,
            "abs_disparate_impact_diff": abs_disparate_impact_diff,
            "eod": eod,
            "abs_eod": abs_eod,
            "aod": aod,
            "abs_aod": abs_aod,
            "theil_index": theil_index,
            "abs_theil_index": abs_theil_index,
        }
    }
}

from pathlib import Path
output_path = Path("results_tmp/final_info.json")
output_path.parent.mkdir(parents=True, exist_ok=True)

import json
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved results to {output_path}")


