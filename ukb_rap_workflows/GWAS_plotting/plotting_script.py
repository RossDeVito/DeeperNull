import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from geneview import manhattanplot, qqplot


# Read summary stats
ss_df = pd.read_csv(
	f"{args.file_path}",
	sep='\s+'
)

# Set 0 p-vals to next lowest
low_fill_val = ss_df[ss_df.P != 0].P.min()
ss_df.loc[ss_df.P == 0, 'P'] = low_fill_val

# Plot Manhattan
ax = manhattanplot(
	data=ss_df,
	pv="P",
)
plt.title(f"{args.pheno} Manhattan Plot")
plt.savefig(
	"manhattan_plot.png",
	dpi=400
)

# Plot QQ
ax = qqplot(
	data=ss_df["P"]
)
plt.savefig(
	"qq_plot.png",
	dpi=400
)