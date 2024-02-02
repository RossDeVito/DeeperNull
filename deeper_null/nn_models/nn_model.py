"""Pytorch Lightning 2.0 model wrapper.

Model configuration JSON should have the following keys:

* model_type: str, one of ['nn_classifier', 'nn_regressor']
* nn_type: str, one of ['dense', 'deep_null']
* nn_args: dict, see nn_args argument for create_nn(nn_type, nn_args)
* train_args: dict, options for training. Keys include:

	* optimizer: str, one of ['adam' (default), 'sgd']
	* lr: float, learning rate. Default is 0.001.
	* batch_size: int, batch size. Default is 2048.
	* max_epochs: int, maximum number of epochs. Default is 1000.
	* patience: int, early stopping patience. Only used if val_frac is
		specified. Default is 10.
	* min_delta: float, early stopping minimum delta. Default is 0
	* verbose: bool, verbose output. Default is False
	* val_frac: float, validation set fraction for early stopping.
		Default is None, which means no early stopping.
	* dataset_type: str, one of ['tabular' (default)]
	* compile: bool, whether to compile the model. Default is False.
	* dataloader_workers: int, number of workers for data loaders.


"""

from copy import deepcopy
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping


from deeper_null.nn_models.nn_networks import create_nn
from deeper_null.nn_models.data_modules import TabularDataset


NN_MODEL_TYPES = ['nn_bin_classifier', 'nn_regressor']


class BaseNN(pl.LightningModule):
	"""Base class for Pytorch Lightning neural network models."""

	def __init__(self, config):
		"""Initialize Pytorch Lightning neural network model."""
		super().__init__()
		self.config = deepcopy(config)

		self.model = create_nn(config['nn_type'], config['nn_args'])
		self.train_args = config['train_args']

	def forward(self, x):
		"""Forward pass."""
		return self.model(x)

	def training_step(self, batch, batch_idx=None):
		"""Training step."""
		x, y = batch
		y_hat = self.model(x)
		loss = self.loss(y_hat, y)
		self.log('train_loss', loss)
		return loss

	def validation_step(self, batch, batch_idx=None):
		"""Validation step."""
		x, y = batch
		y_hat = self.model(x)
		loss = self.loss(y_hat, y)
		self.log('val_loss', loss)
		return loss

	def test_step(self, batch, batch_idx=None):
		"""Test step."""
		x, y = batch
		y_hat = self.model(x)
		loss = self.loss(y_hat, y)
		self.log('test_loss', loss)
		return loss

	def configure_optimizers(self):
		"""Configure optimizer."""
		if 'optimizer' not in self.train_args or self.train_args['optimizer'] == 'adam':
			optimizer = torch.optim.Adam(self.model.parameters(), lr=self.train_args['lr'])
		elif self.train_args['optimizer'] == 'sgd':
			optimizer = torch.optim.SGD(self.model.parameters(), lr=self.train_args['lr'])
		else:
			raise ValueError(f'Invalid optimizer: {self.train_args["optimizer"]}')
		return optimizer


class BinaryClassifierNN(BaseNN):
	"""Pytorch Lightning binary classifier wrapper."""

	def __init__(self, config):
		"""Initialize Pytorch Lightning binary classifier."""
		super().__init__(config)
		self.loss = nn.functional.binary_cross_entropy_with_logits
	
	def predict_step(self, batch, batch_idx=None, dataloader_idx=None):
		"""Predict step."""
		return nn.functional.sigmoid(self.model(batch))


class RegressorNN(BaseNN):
	"""Pytorch Lightning regression wrapper."""

	def __init__(self, config):
		"""Initialize Pytorch Lightning regression."""
		super().__init__(config)
		self.loss = nn.functional.mse_loss
	
	def predict_step(self, batch, batch_idx=None, dataloader_idx=None):
		"""Predict step."""
		return self.model(batch)
	

class NNModel:
	"""Neural network model wrapper.
	
	Trains and predicts using Pytorch Lightning model.

	Creates validation set if val_frac is specified in train_args. Creates
	data loaders using batch_size.
	"""

	def __init__(self, config):
		"""Initialize neural network model."""
		self.config = deepcopy(config)

		# Set default values for lr, batch_size, max_epochs, patience,
		# min_delta, and verbose.
		if 'lr' not in config['train_args']:
			self.config['train_args']['lr'] = 0.001
		if 'batch_size' not in config['train_args']:
			self.config['train_args']['batch_size'] = 2048
		if 'max_epochs' not in config['train_args']:
			self.config['train_args']['max_epochs'] = 1000
		if 'patience' not in config['train_args']:
			self.config['train_args']['patience'] = 10
		if 'min_delta' not in config['train_args']:
			self.config['train_args']['min_delta'] = 0
		if 'verbose' not in config['train_args']:
			self.config['train_args']['verbose'] = False

		# Create model
		if self.config['model_type'] == 'nn_bin_classifier':
			self.model = BinaryClassifierNN(self.config)
		elif self.config['model_type'] == 'nn_regressor':
			self.model = RegressorNN(self.config)
		else:
			raise ValueError(f"Invalid model type: {self.config['model_type']}")
		
		# Compile model
		if ('compile' in config['train_args']) and config['train_args']['compile']:
			self.model = torch.compile(
				self.model,
				mode="reduce-overhead"
			)

		# Set val_frac
		if 'val_frac' in config['train_args']:
			self.val_frac = config['train_args']['val_frac']
		else:
			self.val_frac = None

		# Set n data loader workers
		if 'dataloader_workers' in config['train_args']:
			self.dataloader_workers = config['train_args']['dataloader_workers']
		else:
			self.dataloader_workers = 0

	def fit(self, X, y):
		"""Fit neural network model."""

		# Create data
		if ('dataset_type' not in self.config['train_args']) or \
			(self.config['train_args']['dataset_type'] == 'tabular'):
			train_dataset = TabularDataset(X, y)
		else:
			raise ValueError(
				f"Invalid dataset_type: {self.config['train_args']['dataset_type']}"
			)
		
		# Split data into train and validation sets if val_frac is specified
		if self.val_frac is not None:
			train_dataset, val_dataset = random_split(
				train_dataset, [1 - self.val_frac, self.val_frac]
			)
		else:
			val_dataset = None

		# Create data loaders
		train_loader = DataLoader(
			train_dataset,
			batch_size=self.config['train_args']['batch_size'],
			shuffle=True,
			num_workers=self.dataloader_workers,
			persistent_workers=True
		)

		if val_dataset is not None:
			val_loader = DataLoader(
				val_dataset,
				batch_size=self.config['train_args']['batch_size'],
				shuffle=False,
				num_workers=self.dataloader_workers,
				persistent_workers=True
			)

			# Setup early stopping
			early_stop_callback = EarlyStopping(
				monitor='val_loss',
				patience=self.config['train_args']['patience'],
				min_delta=self.config['train_args']['min_delta'],
				verbose=self.config['train_args']['verbose']
			)

		# Train model
		if val_dataset is not None:
			self.trainer = pl.Trainer(
				max_epochs=self.config['train_args']['max_epochs'],
				callbacks=[early_stop_callback],
			)
			self.trainer.fit(self.model, train_loader, val_loader)
		else:
			self.trainer = pl.Trainer(
				max_epochs=self.config['train_args']['max_epochs'],
			)
			self.trainer.fit(self.model, train_loader)

	def predict(self, X):
		"""Make predictions."""
		# Create data loader
		if ('dataset_type' not in self.config['train_args']) or \
			(self.config['train_args']['dataset_type'] == 'tabular'):
			dataset = TabularDataset(X)
		else:
			raise ValueError(
				f"Invalid dataset_type: {self.config['train_args']['dataset_type']}"
			)
		
		loader = DataLoader(
			dataset,
			batch_size=self.model.train_args['batch_size'],
			shuffle=False,
			num_workers=self.dataloader_workers,
			persistent_workers=True
		)

		# Make predictions
		preds = self.trainer.predict(self.model, loader)
		return preds


def create_nn_model(config):
	"""Create neural network model."""
	return NNModel(config)