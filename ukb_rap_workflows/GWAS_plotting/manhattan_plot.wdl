version 1.0

workflow manhattan_plot {
    input {
        File summary_stats
		String phenotype
    }

    call manhattan_plot_task {
        input:
			summary_stats = summary_stats,
			phenotype = phenotype
    }

    output {
        File manhattan_plot = manhattan_plot_task.manhattan_plot
		File qq_plot = manhattan_plot_task.qq_plot
    }

    meta {
        description: "Create Manhattan and QQ plots"
    }
}

task manhattan_plot_task {
    input {
        File summary_stats
		String phenotype
    }

    command <<<
		# Install geneview for plotting
		pip install geneview

		# Plot Manhattan and QQ
		file_path=~{summary_stats}
		pheno=~{phenotype}

		python3 - <<END_SCRIPT
		import numpy as np
		import pandas as pd
		import matplotlib.pyplot as plt
		from geneview import manhattanplot, qqplot

		ss_df = pd.read_csv(
			"$file_path",
			sep='\\s+'
		)

		ss_df = ss_df[~ss_df['P'].isna()]

		low_fill_val = ss_df[ss_df.P != 0].P.min()
		ss_df.loc[ss_df.P == 0, 'P'] = low_fill_val

		ax = manhattanplot(
			data=ss_df,
			pv='P',
		)
		plt.title("${pheno} Manhattan Plot")
		plt.savefig(
			'manhattan_plot.png',
			dpi=400
		)
		plt.close()

		ax = qqplot(
			data=ss_df['P']
		)
		plt.savefig(
			'qq_plot.png',
			dpi=400
		)
		END_SCRIPT

    >>>

    runtime {
        docker: "gcr.io/ucsd-medicine-cast/nonlin_prs_prs_score_preds:latest"
    }

    output {
        File manhattan_plot = "manhattan_plot.png"
		File qq_plot = "qq_plot.png"
    }
}