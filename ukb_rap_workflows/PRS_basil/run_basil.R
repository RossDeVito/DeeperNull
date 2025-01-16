# Scrit to fit BASIL PRS model and then make prediction on test
# validation set samples.
#
# Predictions are saved as test_preds.csv and val_preds.csv with columns
# 'IID' and 'pred'.
#
# Args:
#   --pheno_file: Path to phenotype file.
#   --pheno_name: Name of phenotype column in phenotype file.
#   --covar_file: Path to covariate file.
#   --geno_file: Path to genotype PGEN file.
#   --train_samples: Path to file with list of samples to use for training.
#   --val_samples: Path to file with list of samples to use for validation.
#       Predictions for these samples will be saved in val_preds.csv.
#   --test_samples: Path to file with list of samples. Predictions for these
#       samples will be saved in test_preds.csv.
#   --alpha: Alpha value for BASIL model. 1 is LASSO, 0 is ridge, and
#       anything in between is elastic net.
#   --n_iter: Number of iterations for BASIL model. Default is 50.


options(error=traceback)

library(argparse)

# Parse arguments function
parse_args <- function() {
    # create parser object
    parser <- ArgumentParser()

    # add arguments
    parser$add_argument(
        "--pheno_file",
        help="Path to phenotype file"
    )
    parser$add_argument(
        "--pheno_name",
        help="Name of phenotype column in phenotype file"
    )
    parser$add_argument(
        "--covar_file",
        help="Path to covariate file"
    )
    parser$add_argument(
        "--geno_file",
        help="Path to genotype PGEN file"
    )
    parser$add_argument(
        "--train_samples",
        help="Path to file with list of samples to use for training"
    )
    parser$add_argument(
        "--val_samples",
        help="Path to file with list of samples to use for validation"
    )
    parser$add_argument(
        "--test_samples",
        help="Path to file with list of samples to use for testing"
    )
    parser$add_argument(
        "--alpha",
        type="double",
        help="Alpha value for BASIL model. 1 is LASSO, 0 is ridge, and anything in between is elastic net"
    )
    parser$add_argument(
        "--n_iter",
        type="integer",
        help="Number of iterations for BASIL model. Default is 50"
    )

    # parse arguments
    args <- parser$parse_args()
    return(args)
}


### Main script ###

# Load libraries
library(snpnet)
library(parallel)
library(jsonlite)

# Get number of cores
num.cores <- detectCores(logical = FALSE)
print(paste("Number of cores:", num.cores))

# Parse arguments
args <- parse_args()

# Create fit config
fit.config <- list(
    nCores = num.cores,
    niter = args$n_iter,  # max number of iterations (default 50)
    use.glmnetPlus = TRUE,  # recommended for faster computation
    early.stopping = TRUE,  # whether to stop based on validation performance (default TRUE)
    plink2.path = "plink2",   # path to plink2 program
    zstdcat.path = "zstdcat",  # path to zstdcat program
    results.dir = ".",
    KKT.verbose = TRUE,
    verbose = TRUE
)

## Create combined covariate and phenotype file ##

# Load phenotype and covariate files
pheno.data <- read.table(
    args$pheno_file,
    header = TRUE,
    sep = "",
	check.names = TRUE
)
covar.data <- read.table(
    args$covar_file,
    header = TRUE,
    sep = "",
	check.names = TRUE
)

# Update `args$pheno_name` to match how R changes the column name
args$pheno_name <- gsub("-", ".", args$pheno_name)

# Inner join and get covariate names
nongeno.data <- merge(pheno.data, covar.data, by = "IID")
covariates <- setdiff(names(nongeno.data), c("IID", args$pheno_name))

## Add column indicating train/val/test split and format IDs ##

# Read in sample IDs
train.ids <- read.table(
    args$train_samples,
    header = FALSE,
    sep = "",
    col.names = c("IID")
)
val.ids <- read.table(
    args$val_samples,
    header = FALSE,
    sep = "",
    col.names = c("IID")
)
test.ids <- read.table(
    args$test_samples,
    header = FALSE,
    sep = "",
    col.names = c("IID")
)

# Add column indicating train/val/test split
nongeno.data$split <- "none"
nongeno.data$split[nongeno.data$IID %in% train.ids$IID] <- "train"
nongeno.data$split[nongeno.data$IID %in% val.ids$IID] <- "val"
nongeno.data$split[nongeno.data$IID %in% test.ids$IID] <- "test"

print(table(nongeno.data$split))

# Split 'IID' column by '_' and assign to 'FID' and 'IID' columns
split_ids <- strsplit(nongeno.data$IID, "_")
nongeno.data$FID <- sapply(split_ids, "[", 1)
nongeno.data$IID <- sapply(split_ids, "[", 2)

## Save covariate and phenotype files ##

write.table(
    nongeno.data,
    file = "nongeno_data.tsv",
    sep = "\t",
    row.names = FALSE
)

## Fit BASIL model ##

# # Get available memory
# memory_info <- system("vmstat -s -S M", intern = TRUE)
# available_memory <- as.numeric(
#     gsub("M.*", "", memory_info[grep("free memory", memory_info)])
# )
# print(paste("Available memory:", available_memory, "MB"))

# Fit
print("Fitting BASIL model...")

start_time <- Sys.time()

fit_snpnet <- snpnet(
    genotype.pfile = args$geno_file,
    phenotype.file = "nongeno_data.tsv",
    phenotype = args$pheno_name,
    covariates = covariates,
    configs = fit.config,
    family = "gaussian",
    split.col = "split",
    # mem = available_memory,
    alpha=args$alpha,
)

# Save runtime
end_time <- Sys.time()
execution_time <- difftime(end_time, start_time, units = "secs")
execution_time_json <- toJSON(
    list(runtime_seconds = as.numeric(execution_time)),
    auto_unbox = TRUE
)
write(execution_time_json, file = "runtime.json")

# Save features included in model
write.csv(
    fit_snpnet$features.to.keep,
    file = "included_features.csv",
    row.names = FALSE
)

# Make predictions
snpnet_preds = predict_snpnet(
    fit = fit_snpnet,
    new_genotype_file=args$geno_file,
    new_phenotype_file = "nongeno_data.tsv",
    phenotype = args$pheno_name,
    covariate_names = covariates,
    split_col = "split",
    split_name = list("val", "test"),
    family = "gaussian"
)

min_lambda_col <- names(which.max(snpnet_preds$metric$val))
min_lambda_col

# Extract predictions and row names for val set
val_predictions <- snpnet_preds$prediction$val[, min_lambda_col]
val_IIDs <- names(val_predictions)

# Save predictions for val set
val_preds <- data.frame(IID = val_IIDs, pred = val_predictions)
write.csv(val_preds, "val_preds.csv", row.names = FALSE)

# Extract predictions and row names for test set
test_predictions <- snpnet_preds$prediction$test[, min_lambda_col]
test_IIDs <- names(test_predictions)

# Save predictions for test set
test_preds <- data.frame(IID = test_IIDs, pred = test_predictions)
write.csv(test_preds, "test_preds.csv", row.names = FALSE)
