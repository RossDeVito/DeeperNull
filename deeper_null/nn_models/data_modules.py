"""Dataset and DataLoader classes for Pytorch Lightning neural network models.

Dataset types:

- TabularDataset: for tabular data. Converts two pandas DataFrames (X and y)
	  to a Pytorch Dataset. Indices are used to match X and y.

- CoordScalingTabularDataset: for tabular data with coordinate scaling. All
	columns that contain 'coord' in the name will be scaled by a factor of
	100,000 (e.g. 600,000 becomes 6). This is useful for scaling the east
	and north coordinates in the UKBB dataset which range from about 0 to
	1.2 million.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd


class TabularDataset(Dataset):
	"""Tabular dataset for Pytorch neural network models.

	Args:
		X: pandas DataFrame, features
		y: pandas Series, target
	"""

	def __init__(self, X, y=None):
		"""Initialize tabular dataset."""
		self.X = X
		self.y = y

	def __len__(self):
		"""Return length of dataset."""
		return len(self.X)

	def __getitem__(self, idx):
		"""Return X and y at index."""
		if self.y is not None:
			return (
				torch.tensor(self.X.iloc[idx].values, dtype=torch.float32), 
				torch.tensor(self.y.iloc[idx].values, dtype=torch.float32)
			)
		else:
			return torch.tensor(self.X.iloc[idx].values, dtype=torch.float32)
		

class CoordScalingTabularDataset(TabularDataset):
	"""Tabular dataset for Pytorch neural network models with coordinate scaling.

	All columns that contain 'coord' in the name will be scaled by a factor of
	100,000 (e.g. 600,000 becomes 6). This is useful for scaling the east and
	north coordinates in the UKBB dataset which range from about 0 to 600,000.
	"""

	def __init__(self, X, y=None):
		"""Initialize tabular dataset."""
		# Scale coordinates
		for col in X.columns:
			if 'coord' in col:
				X[col] = X[col] / 100000

		super().__init__(X, y)