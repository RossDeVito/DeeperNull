## Init geno (in /rdevito/nonlin_prs)

- Get list of variants with MAF >= 0.01 and INFO > 0.3 (double check INFO threshold)

## Step 1

- Keep White British sample that pass initial QC
- Keep variants that pass initial QC
- Filter out multi-allelic variants

## Step 2

- Combine per-chromosome BGEN files into one BGEN file

## Step 3

- Remove variants and samples with missing rate > 20%
- Remove variants missing rate > 2%
- Remove samples with missing rate > 2%
- Also create BGEN version