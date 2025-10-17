import sys
from aif360.datasets import BinaryLabelDataset
from aif360.datasets import AdultDataset, GermanDataset, CompasDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.metrics.utils import compute_boolean_conditioning_vector

from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult, load_preproc_data_compas, load_preproc_data_german

from aif360.algorithms.inprocessing.adversarial_debiasing import AdversarialDebiasing

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.metrics import accuracy_score

from IPython.display import Markdown, display
import matplotlib.pyplot as plt

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


# # ---------- Learn plan classifier without debiasing ----------
# # Load post-processing algorithm that equalizes the odds
# # Learn parameters with debias set to False
sess = tf.Session()
# plain_model = AdversarialDebiasing(privileged_groups = privileged_groups,
#                           unprivileged_groups = unprivileged_groups,
#                           scope_name='plain_classifier',
#                           debias=False,
#                           sess=sess)

# # Fit the model
# plain_model.fit(dataset_orig_train)

# # Apply the plain model to test data
# dataset_nodebiasing_train = plain_model.predict(dataset_orig_train)
# dataset_nodebiasing_test = plain_model.predict(dataset_orig_test)

# # ---------- Evaluate the plain model (without debiasing) ----------
# # Metrics for the dataset from plain model (without debiasing)
# print("#### Plain model - without debiasing - dataset metrics")
# metric_dataset_nodebiasing_train = BinaryLabelDatasetMetric(dataset_nodebiasing_train, 
#                                              unprivileged_groups=unprivileged_groups,
#                                              privileged_groups=privileged_groups)

# print("Train set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_dataset_nodebiasing_train.mean_difference())

# metric_dataset_nodebiasing_test = BinaryLabelDatasetMetric(dataset_nodebiasing_test, 
#                                              unprivileged_groups=unprivileged_groups,
#                                              privileged_groups=privileged_groups)

# print("Test set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_dataset_nodebiasing_test.mean_difference())

# print("#### Plain model - without debiasing - classification metrics")
# classified_metric_nodebiasing_test = ClassificationMetric(dataset_orig_test, 
#                                                  dataset_nodebiasing_test,
#                                                  unprivileged_groups=unprivileged_groups,
#                                                  privileged_groups=privileged_groups)
# print("Test set: Classification accuracy = %f" % classified_metric_nodebiasing_test.accuracy())
# TPR = classified_metric_nodebiasing_test.true_positive_rate()
# TNR = classified_metric_nodebiasing_test.true_negative_rate()
# bal_acc_nodebiasing_test = 0.5*(TPR+TNR)
# print("Test set: Balanced classification accuracy = %f" % bal_acc_nodebiasing_test)
# print("Test set: Disparate impact = %f" % classified_metric_nodebiasing_test.disparate_impact())
# print("Test set: Equal opportunity difference = %f" % classified_metric_nodebiasing_test.equal_opportunity_difference())
# print("Test set: Average odds difference = %f" % classified_metric_nodebiasing_test.average_odds_difference())
# print("Test set: Theil_index = %f" % classified_metric_nodebiasing_test.theil_index())


# ---------- Apply in-processing algorithm based on adversarial learning ----------
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



# ---------- Evaluate the debiased model ----------
# # Metrics for the dataset from plain model (without debiasing)
# print("#### Plain model - without debiasing - dataset metrics")
# print("Train set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_dataset_nodebiasing_train.mean_difference())
# print("Test set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_dataset_nodebiasing_test.mean_difference())

# Metrics for the dataset from model with debiasing
# print("#### Model - with debiasing - dataset metrics")
# metric_dataset_debiasing_train = BinaryLabelDatasetMetric(dataset_debiasing_train, 
#                                              unprivileged_groups=unprivileged_groups,
#                                              privileged_groups=privileged_groups)

# print("Train set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_dataset_debiasing_train.mean_difference())

# metric_dataset_debiasing_test = BinaryLabelDatasetMetric(dataset_debiasing_test, 
#                                              unprivileged_groups=unprivileged_groups,
#                                              privileged_groups=privileged_groups)

# print("Test set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_dataset_debiasing_test.mean_difference())



# print("#### Plain model - without debiasing - classification metrics")
# print("Test set: Classification accuracy = %f" % classified_metric_nodebiasing_test.accuracy())
# TPR = classified_metric_nodebiasing_test.true_positive_rate()
# TNR = classified_metric_nodebiasing_test.true_negative_rate()
# bal_acc_nodebiasing_test = 0.5*(TPR+TNR)
# print("Test set: Balanced classification accuracy = %f" % bal_acc_nodebiasing_test)
# print("Test set: Disparate impact = %f" % classified_metric_nodebiasing_test.disparate_impact())
# print("Test set: Equal opportunity difference = %f" % classified_metric_nodebiasing_test.equal_opportunity_difference())
# print("Test set: Average odds difference = %f" % classified_metric_nodebiasing_test.average_odds_difference())
# print("Test set: Theil_index = %f" % classified_metric_nodebiasing_test.theil_index())


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
print("Test set: Disparate impact = %f" % classified_metric_debiasing_test.disparate_impact())
print("Test set: Equal opportunity difference = %f" % classified_metric_debiasing_test.equal_opportunity_difference())
print("Test set: Average odds difference = %f" % classified_metric_debiasing_test.average_odds_difference())
print("Test set: Theil_index = %f" % classified_metric_debiasing_test.theil_index())


# ---------- Save to JSON ----------
results = {
    "adult": {
        "means": {
            "cls_acc_mean": classified_metric_debiasing_test.accuracy(),
            "balanced_cls_acc_mean": bal_acc_debiasing_test,
            "disparate_impact_mean": classified_metric_debiasing_test.disparate_impact(),
            "abs_eod_mean": abs(classified_metric_debiasing_test.equal_opportunity_difference()),
        },
        "stderrs": {
            "cls_acc_stderr": 0.0,
            "balanced_cls_acc_stderr": 0.0,
            "disparate_impact_stderr": 0.0,
            "abs_eod_stderr": 0.0,
        },
        "final_info_dict": {
            "cls_acc": classified_metric_debiasing_test.accuracy(),
            "balanced_cls_acc": bal_acc_debiasing_test,
            "disparate_impact": classified_metric_debiasing_test.disparate_impact(),
            "abs_eod": abs(classified_metric_debiasing_test.equal_opportunity_difference()),
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


