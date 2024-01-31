"""Create simulated data for testing.

The age and sex covariates will be randomly drawn, and then a phenotype
will be generated with varying amounts of phenotypic variance explained
by the covariates.

To match the real use case, sample IDs will be randomly generated and
then split into training/val/test sets. These sets will be saved as
files that can be used with --train_samples or --pred_samples.

The covariates and phenotypes are saved as whitespace-delimited files.

The covariate component of the phenotype is simulated as follows:

	covariate component = a * Age + b * Sex + c * Age * Sex 
							+ sqrt(|(d - Sex) * Age^2|)

		where age is age/10
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == '__main__':

	# Options
	n_samples = 2500
	age_range = [35, 65]

	var_exp_vals = [0.1, 0.5]

	save_dir = 'dev'

	# Set random seed for reproducibility.
	np.random.seed(36)

	# Generate sample IDs to be in the format {IID}_{IID}
	# where IID is a unique integer.
	iids = np.arange(n_samples)
	iids = np.array([f'{iid}_{iid}' for iid in iids])

	# Randomly split iids 8:1:1 into train/val/test sets and save.
	np.random.shuffle(iids)
	train_iids = iids[:int(0.8 * n_samples)]
	val_iids = iids[int(0.8 * n_samples):int(0.9 * n_samples)]
	test_iids = iids[int(0.9 * n_samples):]

	np.savetxt(os.path.join(save_dir, 'train_samples.txt'), train_iids,
				fmt='%s')
	np.savetxt(os.path.join(save_dir, 'val_samples.txt'), val_iids,
				fmt='%s')
	np.savetxt(os.path.join(save_dir, 'test_samples.txt'), test_iids,
				fmt='%s')
	
	# Generate covariates.
	ages = np.random.randint(age_range[0], age_range[1] + 1, n_samples)
	sexes = np.random.randint(0, 2, n_samples)

	# Save covariates as whitespace-delimited file.
	cov_df = pd.DataFrame({
		'IID': iids,
		'age': ages,
		'sex': sexes
	})
	cov_df.to_csv(
		os.path.join(save_dir, 'covariates.tsv'),
		sep='\t',
		index=False
	)

	# Generate phenotype.

	# Calculate covariate component of phenotype.
	weights = np.random.normal(0, 1, 4)

	cov_comp = (
		weights[0] * (ages / 10) +
		weights[1] * sexes +
		weights[2] * (ages / 10) * sexes +
		np.sqrt(np.abs((weights[3] - sexes) * (ages / 10) ** 2))
	)

	# Plot covariate component.
	sns.displot(
		cov_comp,
		kde=True,
		rug=True,
	)
	plt.show()

	# Calculate variance of random variable
	signal_var = np.var(cov_comp)

	# Calculate variance of noise for desired variance explained levels
	noise_vars = [signal_var * (1 - var_exp) / var_exp for var_exp in var_exp_vals]

	# For each variance explained level, generate noise, add to 
	# signal, and save.

	for i, noise_var in enumerate(noise_vars):
		
		# Generate noise.
		noise = np.random.normal(
			loc=0, scale=np.sqrt(noise_var), size=n_samples
		)

		# Add noise to covariate component.
		pheno = cov_comp + noise

		# Plot phenotype.
		sns.displot(
			pheno,
			kde=True,
			rug=True,
		)
		plt.show()

		# Save phenotype as whitespace-delimited file.
		pheno_df = pd.DataFrame({
			'IID': iids,
			'phenotype': pheno
		})
		pheno_df.to_csv(
			os.path.join(
				save_dir, 
				f"phenotype_{str(var_exp_vals[i]).replace('.', '_')}.tsv"
			),
			sep='\t',
			index=False
		)
