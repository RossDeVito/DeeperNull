"""Linear model for DeeperNull baselines.

Model configuration should have the following keys:

* model_type: str, one of ['linear_regression', 'ridge', 'lasso']
* model_args: dict, optional keyword arguments to underlying sklearn model.
"""

from copy import deepcopy
import pickle

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso

LINEAR_MODEL_TYPES = ['linear_regression', 'ridge', 'lasso']


class LinearModel:
	"""Linear model wrapper."""

	def __init__(self, config):
		"""Initialize linear model."""
		self.model_type = config['model_type']
		self.model_args = config['model_args'] if 'model_args' in config else {}

		if self.model_type == 'linear_regression':
			self.model = LinearRegression(**self.model_args)
		elif self.model_type == 'ridge':
			self.model = Ridge(**self.model_args)
		elif self.model_type == 'lasso':
			self.model = Lasso(**self.model_args)

	def fit(self, X, y):
		"""Fit linear model."""
		self.model.fit(X, y)

	def predict(self, X):
		"""Make predictions."""
		return self.model.predict(X)
	
	def save(self, path, fold_num):
		"""Save model."""
		with open(f"{path}/model_{fold_num}.pkl", 'wb') as f:
			pickle.dump(self.model, f)
	

def create_linear_model(config):
	"""Create linear model."""
	return LinearModel(config)
