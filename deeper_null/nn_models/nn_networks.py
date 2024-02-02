"""Pytorch neural networks for use with deeper_null.nn_models.nn_model.

Create networks using create_nn().
"""

import torch.nn as nn


def get_activation(activation):
	"""Return the activation function based on the string.
	
	Works for all Pytorch activations.

	E.g. get_activation('ReLU') returns nn.ReLU()

	Args:
		activation: str, activation function name (case sensitive)

	Returns:
		nn activation function

	Raises:
		ValueError: if getattr fails
	"""
	try:
		return getattr(nn, activation)()
	except AttributeError:
		raise ValueError(f"Unsupported activation function name {activation}")
	

class DenseNN(nn.Module):
	"""Dense neural network.
	
	Input size is detected using lazy layers. Output size is always 1.

	Hidden layers, activations, and dropout are specified by arguments.

	Args:
		hidden_layers: list of int, hidden layer sizes
		activations: str or list of str, activation function(s). If str, all
			hidden layers use the same activation function. If list, 
			activations[i] is the activation function after hidden_layers[i].
			This means the length of activations should be one less than the
			length of hidden_layers. The output activation is always linear.
		dropout: float, dropout rate
	"""

	def __init__(self, hidden_layers, activations, dropout=0.0):
		"""Initialize dense neural network.
		
		Uses LazyLinear
		"""
		super().__init__()

		# Check type/length of activations
		if isinstance(activations, list):
			if len(activations) != len(hidden_layers) - 1:
				raise ValueError(
					"Length of activations should be one less than length of hidden_layers"
				)
		else:
			assert isinstance(activations, str)
			activations = [activations] * (len(hidden_layers) - 1)

		self.layers = nn.ModuleList()

		# Create layers
		for i, hidden_size in enumerate(hidden_layers):
			self.layers.append(nn.LazyLinear(hidden_size))

			if i < len(hidden_layers) - 1:
				self.layers.append(get_activation(activations[i]))
				
				if dropout > 0:
					self.layers.append(nn.Dropout(dropout))

		# Output layer
		self.layers.append(nn.LazyLinear(1))

	def forward(self, x):
		for layer in self.layers:
			x = layer(x)
		return x
	

def create_nn(nn_type, nn_args):
	"""Create neural network."""
	if nn_type == 'dense':
		return DenseNN(**nn_args)
	elif nn_type == 'deep_null':
		raise NotImplementedError('deep_null not implemented')
	else:
		raise ValueError(f'Invalid nn_type: {nn_type}')