"""
Baseline fairness algorithm: Unconstrained LogisticRegression.

This baseline makes NO fairness considerations — it optimizes purely for
accuracy. The resulting model will have high equalized odds difference
across protected groups.

Agents should modify this file to build a fairness-aware prediction pipeline
using tools from the Fairlearn library (e.g., ThresholdOptimizer,
ExponentiatedGradient, CorrelationRemover, etc.).
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class FairnessAlgorithm:
    """
    Fairness-unaware baseline: plain LogisticRegression with StandardScaler.

    Agents can modify this class to add fairness-aware components such as:
    - Preprocessing: CorrelationRemover, resampling
    - In-processing: ExponentiatedGradient, GridSearch with fairness constraints
    - Post-processing: ThresholdOptimizer
    - Better base classifiers: RandomForest, GradientBoosting, etc.
    """

    def __init__(self):
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(max_iter=1000, random_state=42)),
        ])

    def fit(self, X, y, sensitive_features=None):
        """
        Train the model.

        Args:
            X: Feature matrix (numpy array or pandas DataFrame)
            y: Target labels (binary)
            sensitive_features: Protected attribute values (e.g., sex, race)
                                The baseline ignores this — agents should use it.
        """
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        """Generate predictions."""
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        """Generate probability predictions."""
        return self.pipeline.predict_proba(X)
