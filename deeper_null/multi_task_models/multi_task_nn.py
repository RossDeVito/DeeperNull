"""Neural network components for multi-task models."""

import torch
import torch.nn as nn


def get_activation(activation):
	"""Return the activation function based on the string.
	
	Works for all Pytorch activations.

	E.g. get_activation('ReLU') returns nn.ReLU()

	Args:
		activation: str, activation function name (case sensitive)
	"""
	try:
		return getattr(nn, activation)()
	except AttributeError:
		raise ValueError(f"Unsupported activation function name {activation}")


class MLP(nn.Module):
	"""Multi-layer perceptron.
	
	Input size is detected using lazy layers. Output size is always 1.

	Hidden layers, activations, and dropout are specified by arguments.

	Args:
		hidden_layers: list of int, hidden layer sizes
		activation_after_last_hidden: bool, whether to apply activation after
			the last hidden layer. Default is True. Use False for something
			like the output layer of a regression model.
		activations: str or list of str, activation function(s). If str, all
			hidden layers use the same activation function. If list, 
			activations[i] is the activation function after hidden_layers[i].
			If activation_after_last_hidden is True and activations is a list,
			the length of activations should be one less than the length of
			hidden_layers. Otherwise if a list, the length should be the same
			as hidden_layers. Default is 'ReLU'.
		dropout: float, dropout rate. Default is 0.0.
	"""
	
	def __init__(self, hidden_layers, activation_after_last_hidden=True,
				 activations='ReLU', dropout=0.0):
		"""Initialize multi-layer perceptron.
		
		Uses LazyLinear
		"""
		super().__init__()

		# Check type/length of activations
		if isinstance(activations, list):
			if activation_after_last_hidden:
				if len(activations) != len(hidden_layers) - 1:
					raise ValueError(
						"Length of activations should be one less than length of hidden_layers"
					)
			else:
				if len(activations) != len(hidden_layers):
					raise ValueError(
						"Length of activations should be the same as length of hidden_layers"
					)
		else:
			assert isinstance(activations, str)

		# Create layers
		self.layers = nn.ModuleList()
		for i, hidden_size in enumerate(hidden_layers):
			self.layers.append(nn.LazyLinear(hidden_size))
			
			if i < len(hidden_layers) - 1 or activation_after_last_hidden:
				if isinstance(activations, list):
					self.layers.append(get_activation(activations[i]))
				else:
					self.layers.append(get_activation(activations))
				
				if dropout > 0.0:
					self.layers.append(nn.Dropout(dropout))

	def forward(self, x):
		for layer in self.layers:
			x = layer(x)
		return x
	

def create_nn_module(nn_type, kwargs):
	"""Return neural network module based on type and kwargs.
	
	Args:
		nn_type (str): type of neural network module
		kwargs (dict): keyword arguments for neural network module
	"""
	if nn_type == 'MLP':
		return MLP(**kwargs)
	else:
		raise ValueError(f"Unknown neural network module type: {nn_type}")


class BaseNNModule(nn.Module):
	"""Base neural network module class."""
	
	def __init__(self):
		"""Initialize base neural network module."""
		super().__init__()
		self.network = create_nn_module(
			self.config['network_type'], 
			self.config.get('network_kwargs', dict()),
		)
	
	def forward(self, x):
		"""Forward pass."""
		return self.network(x)


class BaseEncoder(BaseNNModule):
	"""Base encoder class."""
	
	def __init__(self):
		"""Initialize base encoder."""
		super().__init__()

		if self.input_fmt in ['tabular', 'coord_scaling_tabular']:
			self.input_step = nn.Identity()
		elif self.input_fmt == 'scaled_emb_named':
			self.month_embed = nn.Embedding(
				12,
				self.input_args.get('month_emb_dim', 8)
			)
			self.sex_embed = nn.Embedding(
				2,
				self.input_args.get('sex_emb_dim', 8)
			)
		else:
			raise ValueError(f"Unknown input format: {self.input_fmt}")

	def forward(self, x):
		"""Forward pass."""
		if self.input_fmt == 'scaled_emb_named':
			# Embed discrete features
			month_emb = self.month_embed(x['month'])
			sex_emb = self.sex_embed(x['sex'])
			
			# Concatenate all inputs
			x = torch.cat([
				x['birth_coords'],
				x['home_coords'],
				x['numeric'],
				month_emb,
				sex_emb
			], dim=1)

		return self.network(x)
	

class BaseDecoder(BaseNNModule):
	"""Base decoder class."""

	def __init__(self):
		"""Initialize base decoder."""
		super().__init__()

	def forward(self, x):
		"""Forward pass that squeezes output."""
		output = self.network(x)
		if output.shape[-1] == 1:  # Check if the last dimension is 1
			output = output.squeeze(dim=-1)
		return output


def create_encoder(encoder_config, input_fmt, example_batch, input_args={}):
	"""Return encoder based on config.

	Args: (See multi_task_models.py for more info)
		encoder_config (dict): encoder configuration.
		input_fmt (str): input format.
		input_args (dict): input arguments.
	"""
	if encoder_config['network_type'] == 'MLP':
		class EncoderClass(BaseEncoder):
			def __init__(self):
				self.config = encoder_config
				self.input_fmt = input_fmt
				self.input_args = input_args
				super().__init__()

				example_emb = self.forward(example_batch)
				self.example_emb_shape = example_emb.shape
				self.example_emb_dtype = example_emb.dtype
				self.example_emb_device = example_emb.device
			
			def create_random_output(self):
				"""
				Returns a random tensor with the same shape and types as encoder.
				"""
				return torch.randn(
					self.example_emb_shape,
					dtype=self.example_emb_dtype,
					device=self.example_emb_device
				)

		return EncoderClass
	else:
		raise ValueError(f"Unknown encoder type: {encoder_config['network_type']}")
	

def create_decoder(decoder_config, example_emb):
	"""Return decoder based on config.
	
	Args:
		decoder_config (dict): decoder configuration.
	"""
	if decoder_config['network_type'] == 'MLP':
		class DecoderClass(BaseDecoder):
			def __init__(self):
				self.config = decoder_config
				super().__init__()
				self.forward(example_emb)

		return DecoderClass()
	else:
		raise ValueError(f"Decoder currently only supports MLP: {decoder_config['network_type']}")