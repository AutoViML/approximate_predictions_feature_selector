

############################################################################
# Copyright 2025 Ram Seshadri
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#################################################################################################
# The idea of Approximate Predictions Feature Selection is from Samuele Mazzanti's article on Medium below:
# https://medium.com/data-science/approximate-predictions-make-feature-selection-radically-faster-0f9664877687
# This is an implementation in Python by Ram Seshadri 
#################################################################################################

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone, TransformerMixin
from sklearn.feature_selection import SelectorMixin
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import type_of_target
from sklearn.metrics import get_scorer, r2_score, roc_auc_score, log_loss
from sklearn.metrics import average_precision_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from shap import TreeExplainer
from scipy.special import softmax
from lightgbm import LGBMClassifier, LGBMRegressor
import pdb
################################################################################
#### The warnings from Sklearn are so annoying that I have to shut it off #######
import warnings
warnings.filterwarnings("ignore")
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
################################################################################
def warn(*args, **kwargs):
    pass
warnings.warn = warn
import logging
####################################################################################

class AP_Feature_Selector(BaseEstimator, SelectorMixin, TransformerMixin):
    def __init__(self, estimator, n_iter=100, 
                 test_size=0.25, random_state=42):
        self.estimator = estimator
        self.n_iter = n_iter
        self.test_size = test_size
        self.random_state = random_state
        self.metric = None
        self.problem_type_ = None
        self.shap_values_ = None
        self.base_value_ = None
        self.base_values_ = None
        self.feature_names_ = None
        self.classes_ = None
        self.X_val_ = None
        self.y_val_ = None

    def fit(self, X, y):
        # Set random state at the beginning of fit
        if self.random_state is not None:
            np.random.seed(self.random_state)
        # Detect problem type and prepare data
        self._detect_problem_type(y)
        X, y = self._prepare_data(X, y)
        self.feature_names_ = X.columns.tolist()
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=self.test_size,
            stratify=y if self.problem_type_ in ['binary', 'multiclass'] else None,
            random_state=self.random_state
        )
        self.X_val_ = X_val
        self.y_val_ = y_val
        
        # Train model
        self.estimator_ = clone(self.estimator).fit(X_train, y_train)
        
        # Compute SHAP values
        self._compute_shap_values(X_val)
        
        # Feature selection process
        best_score = -np.inf
        best_features = []
        
        for _ in range(self.n_iter):
            candidate = self._generate_candidate()
            approx_pred = self._approximate_predictions(candidate, X_val)
            score = self._calculate_score(y_val, approx_pred)
            
            if score > best_score:
                best_score = score
                best_features = candidate
        
        self.selected_features_ = best_features
        self.n_features_ = len(best_features)
        self.n_features_in_ = X.shape[1]
        
        return self

    def _detect_problem_type(self, y):
        """Determine problem type using sklearn's type_of_target"""
        target_type = type_of_target(y)
        if target_type == 'binary':
            self.problem_type_ = 'binary'
        elif target_type == 'multiclass':
            self.problem_type_ = 'multiclass'
        elif target_type == 'continuous':
            self.problem_type_ = 'regression'
        else:
            raise ValueError(f"Unsupported target type: {target_type}")
        self.metric = self.set_default_metric()

    def _prepare_data(self, X, y):
        """Encode labels and validate data"""
        if self.problem_type_ in ['binary', 'multiclass']:
            self.le_ = LabelEncoder()
            y = pd.Series(self.le_.fit_transform(y), index=y.index)
            self.classes_ = self.le_.classes_
        return X, y

    def _compute_shap_values(self, X_val):
        """Compute and store SHAP values with proper multi-class handling"""
        explainer = TreeExplainer(self.estimator_)
        raw_shap = explainer.shap_values(X_val)
        
        if self.problem_type_ == 'regression':
            self.base_value_ = explainer.expected_value
            self.shap_values_ = pd.DataFrame(
                raw_shap, 
                columns=X_val.columns, 
                index=X_val.index
            )

        elif self.problem_type_ == 'binary':
            # Handle LightGBM binary classifier output
            if isinstance(raw_shap, list) and len(raw_shap) == 2:
                self.base_value_ = explainer.expected_value[1]
                self.shap_values_ = pd.DataFrame(
                    raw_shap[1], 
                    columns=X_val.columns, 
                    index=X_val.index
                )
            else:
                self.base_value_ = explainer.expected_value
                self.shap_values_ = pd.DataFrame(
                    raw_shap, 
                    columns=X_val.columns, 
                    index=X_val.index
                )

        else:  # Multiclass
            # Stack SHAP values (classes, samples, features)
            self.shap_values_ = np.stack([s for s in raw_shap], axis=0)
            
            # Reshape base values for broadcasting (samples, classes)
            self.base_values_ = np.array(explainer.expected_value).reshape(1, -1)
 
    def _approx_multiclass(self, candidate, X_val):
        """Calculate valid per-sample predictions"""
        # Get feature indices
        col_idx = [X_val.columns.get_loc(c) for c in candidate]
        
        # Sum contributions (samples, classes)
        contributions = self.shap_values_[:, col_idx, :].sum(axis=1)
        
        # Add aligned base values
        logits = contributions + self.base_values_
        return softmax(logits, axis=1)
        
    def _generate_candidate(self):
        """Generate random feature candidate"""
        n_features = np.random.randint(1, len(self.feature_names_)+1)
        return np.random.choice(
            self.feature_names_, 
            size=n_features, 
            replace=False
        ).tolist()

    def _approximate_predictions(self, candidate, X_val):
        """Generate predictions using SHAP values"""
        if self.problem_type_ == 'regression':
            return self._approx_regression(candidate)
        elif self.problem_type_ == 'binary':
            return self._approx_binary(candidate)
        else:
            return self._approx_multiclass(candidate, X_val)

    def _approx_regression(self, candidate):
        """Regression predictions"""
        contributions = self.shap_values_[candidate].sum(axis=1)
        return self.base_value_ + contributions

    def _approx_binary(self, candidate):
        """Binary classification predictions"""
        contributions = self.shap_values_[candidate].sum(axis=1)
        logits = self.base_value_ + contributions
        return 1 / (1 + np.exp(-logits))

    def set_default_metric(self):
        """Set default metrics"""
        # Default metrics
        if self.problem_type_ == 'regression':
            return r2_score
        elif self.problem_type_ == 'binary':
            return roc_auc_score
        else:
            return log_loss

    def _calculate_score(self, y_true, approx_pred):
        """Handle different prediction formats"""
        if self.problem_type_ == 'multiclass':
            # Convert to class predictions
            y_pred = np.argmax(approx_pred, axis=1)
            return accuracy_score(y_true, y_pred)  # Or use log_loss with probabilities
        else:
            # Default metrics
            if self.problem_type_ == 'regression':
                return r2_score(y_true, approx_pred)
            elif self.problem_type_ == 'binary':
                return roc_auc_score(y_true, approx_pred)
            else:
                return log_loss(y_true, approx_pred)
        
    def _get_support_mask(self):
        return np.isin(self.feature_names_, self.selected_features_)

    def transform(self, X):
        return X[self.selected_features_]