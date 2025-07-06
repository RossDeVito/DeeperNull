"""XGBoost model wrapper.

Model configuration JSON should have the following keys:

* model_type: str, one of ['xgb_classifier', 'xgb_regressor']
* model_args: dict, parameters to pass to the XGBoost model constructor
* fit_args: dict, parameters that impact model fitting. Mainly used for
	early stopping when the 'val_frac' key is present. If 'val_frac' is
	present, then the model will use early stopping based on the randomly
	sampled validation set. Use with 'eval_metric' and 
	'early_stopping_rounds' in 'model_args'.

NOTE: Classifier is only for binary classification.
"""

from copy import deepcopy
from sklearn.model_selection import train_test_split

import xgboost as xgb


XGB_MODEL_TYPES = ['xgb_classifier', 'xgb_regressor']


class ClassifierXGB:
	"""XGBoost binary classifier wrapper."""

	def __init__(self, config):
		"""Initialize XGBoost classifier."""
		self.model = xgb.XGBClassifier(**config['model_args'])
		self.fit_args = config['fit_args']

	def fit(self, X, y):
		"""Fit XGBoost classifier."""
		if 'val_frac' in self.fit_args:
			# Sample validation set
			X_train, X_val, y_train, y_val = train_test_split(
				X, y, test_size=self.fit_args['val_frac'], stratify=y)
			
			# Fit model
			self.model.fit(
				X_train, y_train, eval_set=[(X_val, y_val)]
			)
		else:
			self.model.fit(X, y)

	def predict(self, X):
		"""Make predictions as probabilities."""
		return self.model.predict_proba(X)[:, 1]
	
	def save(self, path, fold_num):
		"""Save model."""
		self.model.save_model(
			f"{path}/model_{fold_num}.json"
		)
	

class RegressorXGB:
	"""XGBoost regressor wrapper."""

	def __init__(self, config):
		"""Initialize XGBoost regressor."""
		self.model = xgb.XGBRegressor(**config['model_args'])
		self.fit_args = config['fit_args']

	def fit(self, X, y):
		"""Fit XGBoost regressor."""
		if 'val_frac' in self.fit_args:
			# Sample validation set
			X_train, X_val, y_train, y_val = train_test_split(
				X, y, test_size=self.fit_args['val_frac'])
			
			# Fit model
			self.model.fit(
				X_train, y_train, eval_set=[(X_val, y_val)]
			)
		else:
			self.model.fit(X, y)

	def predict(self, X):
		"""Make predictions."""
		return self.model.predict(X)
	
	def save(self, path, fold_num):
		"""Save model."""
		self.model.save_model(
			f"{path}/model_{fold_num}.json"
		)


def create_xgb_model(config):
	"""Create XGBoost model."""
	config = deepcopy(config)

	if config['model_type'] == 'xgb_classifier':
		return ClassifierXGB(config)
	elif config['model_type'] == 'xgb_regressor':
		return RegressorXGB(config)
	else:
		raise ValueError(f"Invalid model type: {config['model_type']}")
	