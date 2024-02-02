from setuptools import setup, find_packages

setup(
    name='DeeperNull',
    version='0.1.0',
    packages=find_packages(),
    author='Ross DeVito',
    author_email='rdevito@ucsd.edu',
    description='DeepNull style models with other features',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=[
		'matplotlib',
        'numpy',
        'pandas',
		'pytorch-lightning',
        'scikit-learn',
        'scipy',
        'seaborn',
		'torch>=2.1',
		'tqdm',
		'xgboost',
    ],
)