""" Wrapper for LibMTL mutli-task nueral network models.

LibMTL [link](https://github.com/median-research-group/LibMTL/tree/main)

Model configuration JSON should have the following keys:

* model_type: str, model type. Must be 'multi_task'.
* weighting_strategy: str, weighting strategy. Must be class name in
	`LibMTL.weighting`.
* weighting_kwargs: optional dict, weighting parameters.
* rep_grad: bool, whether to use (rep-grad)[
	https://libmtl.readthedocs.io/en/latest/docs/user_guide/mtl.html].
	Default is True. Weighting strategies may require or not support rep-grad.
* architecture: str, architecture name. Must be class name in
	`LibMTL.architecture`.
* architecture_kwargs: optional dict, architecture parameters.
* train_params: dict, training parameters. Keys are:
	* epochs: int, number of epochs. Default is 100.
	* optim: str, optimizer name. Default is 'adam'. Must be class name in
		`torch.optim`.
	* optim_params: dict, optimizer parameters. Default is {lr: 1e-3}.
	* scheduler_params: optional dict, learning rate scheduler parameters.
		Default is None, which means no scheduler.
	* val_frac: optional float, fraction of data to use for validation. 
		Default is 0.1.
	* dataloader_workers: optional int, number of workers for data loader.
		Default is 0.
	* batch_size: int, batch size. Default is 32.
	* device: str, device to use. e.g. 'cuda:0'. Default is 'cpu'.
	* early_stopping_patience: optional int, number of epochs to wait before
		stopping training if new best epoch is not achieved. Default is None
		for no early stopping.
* encoder: dict defining encoder. Keys are:
	* 'network_type': str, name of neural network architecture to use. One
		of:
			- 'MLP': Multi-layer perceptron.
	* 'network_kwargs': dict, arguments to pass to network class. default is {}.
	* 'input_args': optional dict, arguments specifying how to treat
		input if `input_fmt` is 'scaled_emb_named'. Default is:
			- 'month_emb_dim': 8, dimension of month embedding.
			- 'sex_emb_dim': 8, dimension of sex embedding.
* decoder: dict defining decoders. Assumed all decoders will have the same
	architecture. Keys are the same as 'encoder', excluding 'input_args'.
	'network_type' is restricted to ['MLP'].
* input_fmt: str, format of data input to model and the respective data
	loader to use.
		- 'tabular': Single input vector. (default)
		- 'coord_scaling_tabular': Single input vector where features
			containing 'coord' in the name are scaled by /100,000.
		- 'scaled_emb_named': Single input vector where input to network is
			a dictionary. Coordinate (feature name contains 'coord'), time of
			time of day (feature name is 'time_of_day'), day of year
			(name is 'day_of_year'), and age ('age*') are scaled by
			/100,000, /1,000, /365, and /100, respectively. Month of year
			(name is 'month_of_year') and sex (contains 'sex') are returned
			as ints to be embedded. Exactly one feature name in the input
			must meet the criteria in the following descriptions. The keys of
			the input dictionary will be:
				- 'birth_coords': Feature names containing {'birth', 'coord',
					'north'} and {'birth', 'coord', 'east'} scaled.
				- 'home_coords': Feature names containing {'home', 'coord',
					'north'} and {'home', 'coord', 'east'} scaled.
				- 'numeric': Vector of float time (required), age (required), and
					optional PCs. Feature names must be: ['time_of_day',
					'day_of_year', 'age*', ['pc_*']]
				- 'month': Feature named 'month_of_year' returned as int.
				- 'sex': Feature name matching 'sex*' returned as int (0 or 1).
* tasks: dict defining tasks. Keys are names of pheotypes from the various
	pheotype files. Values are dicts with the following keys:
	- 'loss': str, loss name. must be 'mse' or 'mae'.
	- 'weight': (optional) default is 0 for lower loss is better, 1 for higher
		loss is better.
"""

import os
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

import LibMTL as mtl
from LibMTL.loss import MSELoss, L1Loss
from LibMTL.metrics import AbsMetric#, L1Metric

from deeper_null.multi_task_models.multi_task_nn import (
	create_encoder, create_decoder
)
from deeper_null.multi_task_models.data_modules import (
	MTTabularDataset, MTCoordScalingTabularDataset, MTScaledEmbNamedDataset
)
from deeper_null.multi_task_models.trainer import MTTrainer


# Fixed L1 Metric (Mean Absolute Error)
class L1Metric(AbsMetric):
	"""Calculate the Mean Absolute Error (MAE)."""
	def __init__(self):
		super(L1Metric, self).__init__()
		
	def update_fun(self, pred, gt):
		# Calculate the sum of absolute errors for the batch
		abs_err = torch.abs(pred - gt)
		self.record.append(abs_err.sum().item())  # Sum of absolute errors for this batch
		self.bs.append(pred.size()[0])  # Batch size
	
	def score_fun(self):
		# Convert recorded values to numpy arrays
		total_abs_error = np.sum(self.record)  # Total sum of all absolute errors
		total_samples = np.sum(self.bs)  # Total number of samples
		
		# Return the result as a list, even if it's a single value
		return [ total_abs_error / total_samples]  # List containing MAE
	

# class MSELoss(mtl.loss.MSELoss):
# 	"""Fixed Mean Squared Error loss for regression tasks."""
# 	def __init__(self):
# 		super().__init__()
		
# 	def _update_loss(self, pred, gt):
# 		loss = self.compute_loss(pred, gt)
# 		self.record.append(loss.item())
# 		self.bs.append(pred.size()[0])
# 		return loss
	

class MultiTaskModel:
	"""Mutli-task nueral network model.

	Wrapper for LibMTL mutli-task nueral network models.

	Creates validation set if val_frac is specified in train_args. Creates
	data loaders using batch_size.
	"""

	def __init__(self, config, example_batch, out_dir=None):
		"""Initialize multi-task model.

		Args:
			config (dict): model configuration. See module docstring for
				details.
			example_batch (dict): example batch of data to initialize model.
			out_dir (str, optional): output directory. If not None,
				CSVLogger will save logs to this directory and save
				training curve plots. Default is None.
		"""
		self.config = deepcopy(config)
		self.out_dir = out_dir

		# Create elements needed to train model
		self.task_dict = self.make_task_dict()
		self.weighting_strategy = config['weighting_strategy']
		self.architecture = config['architecture']
		self.device = self.config['train_params'].get('device', 'cpu')
		self.encoder = create_encoder(
			self.config['encoder'],
			self.config['input_fmt'],
			example_batch=example_batch.to(self.device)
		)
		example_emb = self.encoder().create_random_output()
		self.decoders = nn.ModuleDict({
			task_name: create_decoder(
				self.config['decoder'],
				example_emb
			)
			for task_name in self.task_dict.keys()
		})
		self.optim_params = self.make_optim_params_dict()
		self.scheduler_params = self.config['train_params'].get(
			'scheduler_params', dict()
		)
		self.trainer_kwargs = {
			'weight_args': self.config.get('weighting_kwargs', dict()),
			'arch_args': self.config.get('architecture_kwargs', dict()),
		}
		# self.trainer_kwargs = {}
		# if 'weight_args' in self.config:
		# 	self.trainer_kwargs['weighting_kwargs'] = self.config['weighting_kwargs']
		# if 'arch_args' in self.config:
		# 	self.trai

		# Create trainer
		self.trainer = MTTrainer(
			task_dict=self.task_dict,
			weighting=self.weighting_strategy,
			architecture=self.architecture,
			encoder_class=self.encoder,
			decoders=self.decoders,
			rep_grad=self.config.get('rep_grad', True),
			multi_input=False,
			optim_param=self.optim_params,
			scheduler_param=self.scheduler_params,
			device=self.device,
			**self.trainer_kwargs
		)

		# Set validation fraction and number of data loader workers
		self.val_frac = self.config['train_params'].get('val_frac', 0.1)
		self.dataloader_workers = self.config['train_params'].get(
			'dataloader_workers', 1
		)
			
	def make_task_dict(self):
		"""Return task dictionary for LibMTL model based on config."""
		task_dict = {}

		for task_name, task_info in self.config['tasks'].items():
			if task_info['loss'] == 'mse':
				loss = MSELoss()
			elif task_info['loss'] == 'mae':
				loss = L1Loss()
			else:
				raise ValueError(f'Invalid loss function: {task_info["loss"]}')

			task_dict[task_name] = {
				'loss_fn': loss,
				'weight': [task_info.get('weight', 0)],
				'metrics': ['MAE'],
				'metrics_fn': L1Metric()
			}

		return task_dict
	
	def make_optim_params_dict(self):
		"""Return optimizer parameters dictionary for LibMTL model based on
		config. Default is Adam with lr=1e-3.
		"""
		optim_params = {
			# get from self.config['train_params']['optim'] if exists, else 'adam'
			'optim': self.config['train_params'].get('optim', 'adam'),
			**self.config['train_params'].get('optim_params', {'lr': 1e-3})
		}
		return optim_params
	
	def fit(self, X, y):
		"""Fit multi-task model."""

		# Create data loader
		if self.config['input_fmt'] == 'tabular':
			dataset = MTTabularDataset(
				X, y, task_names=list(self.task_dict.keys())
			)
		elif self.config['input_fmt'] == 'coord_scaling_tabular':
			dataset = MTCoordScalingTabularDataset(
				X, y, task_names=list(self.task_dict.keys())
			)
		elif self.config['input_fmt'] == 'scaled_emb_named':
			dataset = MTScaledEmbNamedDataset(
				X, y, task_names=list(self.task_dict.keys())
			)
		else:
			raise ValueError(f'Invalid input format: {self.config["input_fmt"]}')
		
		dataset.drop_samples_missing_labels()
		
		# Split data into train and validation sets
		len_dataset = len(dataset)
		val_size = int(self.val_frac * len_dataset)
		train_size = len_dataset - val_size
		train_dataset, val_dataset = random_split(
			dataset,
			[train_size, val_size]
		)

		# Create data loaders
		train_loader = DataLoader(
			train_dataset,
			batch_size=self.config['train_params'].get('batch_size', 32),
			num_workers=self.dataloader_workers,
			persistent_workers=True
		)
		val_loader = DataLoader(
			val_dataset,
			batch_size=self.config['train_params'].get('batch_size', 32),
			shuffle=False,
			num_workers=self.dataloader_workers,
			persistent_workers=True
		)

		# Train model
		self.trainer.train(
			train_dataloaders=train_loader,
			val_dataloaders=val_loader,
			test_dataloaders=val_loader,
			epochs=self.config['train_params'].get('epochs', 100),
			early_stopping_patience=self.config['train_params'].get(
				'early_stopping_patience',
				None
			),
		)
		
	def predict(self, X):
		"""Predict using multi-task model."""
		if self.config['input_fmt'] == 'tabular':
			dataset = MTTabularDataset(
				X, task_names=list(self.task_dict.keys())
			)
		elif self.config['input_fmt'] == 'coord_scaling_tabular':
			dataset = MTCoordScalingTabularDataset(
				X, task_names=list(self.task_dict.keys())
			)
		elif self.config['input_fmt'] == 'scaled_emb_named':
			dataset = MTScaledEmbNamedDataset(
				X, task_names=list(self.task_dict.keys())
			)
		else:
			raise ValueError(f'Invalid input format: {self.config["input_fmt"]}')
		
		dataloader = DataLoader(
			dataset,
			batch_size=self.config['train_params'].get('batch_size', 32),
			num_workers=self.dataloader_workers,
			persistent_workers=True
		)
		
		return self.trainer.predict(dataloader)

	


