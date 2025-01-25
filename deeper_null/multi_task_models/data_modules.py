"""Datasets modified to work with multi-task models."""

import numpy as np
import torch

from deeper_null.nn_models.data_modules import (
	TabularDataset, ScaledEmbNamedDataset
)


class MTTabularDataset(TabularDataset):
	"""Tabular dataset for multi-task models.

	Args:
		X: pandas DataFrame, features
		y: pandas DataFrame, targets
	"""

	def __init__(self, X, y=None, task_names=[]):
		"""Initialize tabular dataset."""
		super().__init__(X, y)

		assert len(task_names) >= 1, "At least one task name required."
		self.task_names = task_names

		# Create y with just features being used as labels
		if y is not None:
			self.y = y[task_names]

	def drop_samples_missing_labels(self):
		"""Drop samples missing labels values."""
		if self.y is not None:
			valid_idx = self.y.loc[~self.y.isnull().any(axis=1)].index
			self.X = self.X.loc[valid_idx]
			self.y = self.y.loc[valid_idx]
		else:
			raise ValueError("No y value to drop samples missing labels based on.")

	def __getitem__(self, idx):
		"""Return X and optionally y at index.

		y is a dictionary of name-label pairs.
		"""
		if self.y is not None:
			return (
				torch.tensor(self.X.iloc[idx].values, dtype=torch.float32),
				{
					task: torch.tensor(self.y[task].iloc[idx], dtype=torch.float32)
					for task in self.task_names
				}
			)
		else:
			return torch.tensor(self.X.iloc[idx].values, dtype=torch.float32)


class MTCoordScalingTabularDataset(MTTabularDataset):
	"""Tabular dataset for multi-task models with coordinate scaling.

	All columns that contain 'coord' in the name will be scaled by a factor of
	100,000 (e.g. 600,000 becomes 6). This is useful for scaling the east and
	north coordinates in the UKBB dataset which range from about 0 to 600,000.
	"""

	def __init__(self, X, y=None, task_names=[]):
		"""Initialize tabular dataset."""
		# Scale coordinates
		for col in X.columns:
			if 'coord' in col:
				X[col] = X[col] / 100000

		super().__init__(X, y, task_names)


class MTScaledEmbNamedDataset(ScaledEmbNamedDataset):
	"""ScaledEmbNamedDataset for multi-task models.
	
	See nn_models/data_modules.py and multi_task_models/multi_task_models.py
	for more information.
	"""

	def __init__(self, X, y=None, task_names=[]):
		"""Initialize tabular dataset."""
		super().__init__(X, y)

		assert len(task_names) >= 1, "At least one task name required."
		self.task_names = task_names

		# Create y with just features being used as labels
		if y is not None:
			self.y = y[task_names]

	def drop_samples_missing_labels(self):
		"""Drop samples missing labels values."""
		valid_mask = ~self.y.isnull().any(axis=1)

		# Apply the mask directly to filter rows
		self.X = self.X.loc[valid_mask]
		self.y = self.y.loc[valid_mask]

	def __getitem__(self, idx):
		"""Return X and optionally y at index.

		y is a dictionary of name-label pairs.
		"""
		row = self.X.iloc[idx]

		# Read pre-scaled values
		birth_coords_vals = np.array(row[self.birth_coords_cols], dtype=np.float32)
		home_coords_vals = np.array(row[self.home_coords_cols], dtype=np.float32)
		numeric_vals = [row[c] for c in self.numeric_cols]

		inputs = {
			"birth_coords": torch.tensor(birth_coords_vals, dtype=torch.float32),
			"home_coords": torch.tensor(home_coords_vals, dtype=torch.float32),
			"numeric": torch.tensor(numeric_vals, dtype=torch.float32),
			"month": torch.tensor(row[self.month_col] - 1, dtype=torch.int),
			"sex": torch.tensor(row[self.sex_cols[0]], dtype=torch.int),
		}

		if self.y is not None:
			y_row = self.y.iloc[idx]
			y_vals = {
				task: torch.tensor(y_row[task], dtype=torch.float32)
				for task in self.task_names
			}
			return inputs, y_vals
		else:
			return inputs