""" Wrapper for LibMTL mutli-task nueral network models.

LibMTL [link](https://github.com/median-research-group/LibMTL/tree/main)

Only supports hard parameter sharing.

Model configuration JSON should have the following keys:

* weighting_strategy: str, weighting strategy. Must be class name in
	`LibMTL.weighting`.
# weighting_params: optional dict, weighting parameters.
* architecture: str, architecture name. Must be class name in
	`LibMTL.architecture`.
* architecture_params: optional dict, architecture parameters.
* train_params: dict, training parameters. Keys are:
	* loss_fn: str, loss function name. Must be class name in `torch.nn`.
	* optim: str, optimizer name. Default is 'adam'. Must be class name in
		`torch.optim`.
	* optim_params: dict, optimizer parameters. Default is {lr: 1e-3}.
	* scheduler_params: optional dict, learning rate scheduler parameters.
		Default is None, which means no scheduler.
* encoder: dict defiing encoder. Keys are:
	TODO
* decoder: dict defining decoders. Assumed all decoders will have the same
	architecture. Keys include:
	TODO
* input_fmt: dict, defines format of data input to model. Keys are: ???
"""