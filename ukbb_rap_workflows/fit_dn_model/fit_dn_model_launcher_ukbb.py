"""Script to launch UKBB DeeperNull model fitting workflow.

Args:

	Required:
		* `--model-desc` (str): Name used when saving the model.
		* `--model-config` (str): Path to model configuration file.
"""

import argparse
import dxpy
import sys
import time


