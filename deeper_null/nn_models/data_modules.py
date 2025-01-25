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

import numpy as np
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


class ScaledEmbNamedDataset(Dataset):
	"""
	Dataset returning a dictionary of named inputs for a model expecting:
	  - Exactly two birth_coords columns, two home_coords columns, and single
		columns for numeric items (time_of_day, day_of_year, ages*, pc_*) plus
		a single month_of_year column and a single sex column. Columns are
		scaled at runtime:
		* Any columns matching 'coord' in birth_coords or home_coords are /100000
		* time_of_day is /1000
		* day_of_year is /365
		* age* is /100
		* month_of_year is integer
		* sex is integer

	Args:
		X (pandas.DataFrame): feature data, containing columns with
		  'birth' and 'coord', 'home' and 'coord', 'time_of_day',
		  'day_of_year', 'month_of_year', 'sex', 'age*' and optional 'pc_*'
		y (pandas.DataFrame or Series, optional): targets
	"""

	def __init__(self, X, y=None):
		self.X = X.copy(deep=True)
		self.y = y

		# birth_coords: exactly one north and one east
		birth_north = [c for c in X.columns if "birth" in c and "coord" in c and "north" in c]
		birth_east = [c for c in X.columns if "birth" in c and "coord" in c and "east" in c]
		if len(birth_north) != 1 or len(birth_east) != 1:
			raise ValueError("Expected exactly one birth_north and one birth_east column.")
		self.birth_coords_cols = [birth_north[0], birth_east[0]]

		# home_coords: exactly one north and one east
		home_north = [c for c in X.columns if "home" in c and "coord" in c and "north" in c]
		home_east = [c for c in X.columns if "home" in c and "coord" in c and "east" in c]
		if len(home_north) != 1 or len(home_east) != 1:
			raise ValueError("Expected exactly one home_north and one home_east column.")
		self.home_coords_cols = [home_north[0], home_east[0]]

		# time_of_day: exactly one
		time_of_day_cols = [c for c in X.columns if c == "time_of_day"]
		if len(time_of_day_cols) != 1:
			raise ValueError("Expected exactly one 'time_of_day' column.")
		tod = time_of_day_cols[0]

		# day_of_year: exactly one
		day_of_year_cols = [c for c in X.columns if c == "day_of_year"]
		if len(day_of_year_cols) != 1:
			raise ValueError("Expected exactly one 'day_of_year' column.")
		doy = day_of_year_cols[0]

		# age: exactly one
		age_cols = [c for c in X.columns if c.startswith("age")]
		if len(age_cols) != 1:
			raise ValueError("Expected exactly one 'age*' column.")
		age_col = age_cols[0]

		# month_of_year: exactly one
		if "month_of_year" not in X.columns:
			raise ValueError("Missing required 'month_of_year' column.")
		self.month_col = "month_of_year"

		# sex: exactly one
		self.sex_cols = [c for c in X.columns if "sex" in c]
		if len(self.sex_cols) != 1:
			raise ValueError("Expected exactly 1 column containing 'sex'.")

		# numeric includes time_of_day, day_of_year, age, then pc_*
		pc_cols = [c for c in X.columns if c.startswith("pc_")]
		self.numeric_cols = [tod, doy, age_col] + pc_cols

		# Apply scaling once
		self.X[self.birth_coords_cols] /= 100000
		self.X[self.home_coords_cols] /= 100000
		for c in self.numeric_cols:
			if c == "time_of_day":
				self.X[c] /= 1000
			elif c == "day_of_year":
				self.X[c] /= 365
			elif c.startswith("age"):
				self.X[c] /= 100

	def __len__(self):
		return len(self.X)

	def __getitem__(self, idx):
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
			y_val = torch.tensor(self.y.iloc[idx].values, dtype=torch.float32)
			return inputs, y_val
		return inputs

