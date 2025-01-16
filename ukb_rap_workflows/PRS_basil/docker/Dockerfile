FROM jupyter/r-notebook:latest

# Copy in plink2 binary from local directory
COPY plink2 /usr/local/bin/plink2

# Install zstandard (zstd) vis conda
RUN conda install -c conda-forge zstandard

# Install R packages from CRAN for BASIL
RUN R -e "install.packages(c('devtools', 'tidyverse', 'data.table', 'glmnet', 'ROCR', 'crochet', 'dotCall64'), repos = 'http://cran.us.r-project.org', dependencies=TRUE)"

# Install R packages from GitHub
RUN R -e "Sys.setenv(TAR = '/bin/tar'); devtools::install_github('chrchang/plink-ng', subdir='2.0/pgenlibr');"
RUN R -e "Sys.setenv(TAR = '/bin/tar'); devtools::install_github('chrchang/plink-ng', subdir='2.0/cindex');"
RUN R -e "Sys.setenv(TAR = '/bin/tar'); devtools::install_github(c('junyangq/glmnetPlus'))"
RUN R -e "Sys.setenv(TAR = '/bin/tar'); devtools::install_github('RossDeVito/snpnet', force=TRUE)"

# Install R packages for script
RUN R -e "install.packages('argparse', repos = 'http://cran.us.r-project.org')"

# Copy in run_basil.R script
COPY run_basil.R /home/run_basil.R

# Copy in python covar preprocessing script and requirements
COPY prepro_covar.py /home/prepro_covar.py
RUN pip install pandas
