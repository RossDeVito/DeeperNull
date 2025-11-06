"""
Simplified 1-SII bar plotting:
- Choose phenotypes in PHENOS_TO_PLOT
- For each phenotype, show a horizontal bar plot of 1-SII values:
	• Diagonal entries = individual effects
	• Upper-triangle entries = pairwise interactions
"""

import os
import json
import math
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- CONFIG ----------
SAVE_DIR = "agg_shap"
AGG_METHOD = "mean"  # 'mean' | 'median' | 'std'
TOP_K = None         # e.g., 25 to show only top 25 effects/interactions; or None for all
SORT_DESC = True     # sort bars by value (descending)
PLOT_SAVE_DIR = "plots"

# Comment/uncomment to choose which phenotypes to show (one figure per phenotype)
PHENOS_TO_PLOT = [
	"standing_height_50",
	"body_fat_percentage_23099",
	"platelet_count_30080",
	"glycated_haemoglobin_30750",
	"vitamin_d_30890",
	"diastolic_blood_pressure_4079",
	"systolic_blood_pressure_4080",
	"FEV1_3063",
	"FVC_3062",
	"HDL_cholesterol_30760",
	"LDL_direct_30780",
	"triglycerides_30870",
	"c-reactive_protein_30710",
	"creatinine_30700",
	"alanine_aminotransferase_30620",
	"aspartate_aminotransferase_30650",
	"asthma_42015",
	"depression_20438",
	"diabetes_2443",
]

# Display names for features
FEATURE_NAMES_MAP = {
	'age_at_recruitment_21022': 'Age',
	'sex_31': 'Sex',
	'birth_coord_north_129': 'Birth location (north)',
	'birth_coord_east_130': 'Birth location (east)',
	'home_location_north_coord_22704': 'Home location (north)',
	'home_location_east_coord_22702': 'Home location (east)',
	'time_of_day': 'Time of day',
	'day_of_year': 'Day of year',
	'month_of_year': 'Month of year',
}

# Matplotlib figure DPI (export quality)
import matplotlib as mpl
mpl.rcParams['savefig.dpi'] = 1200


def load_shap_values(save_dir: str, phenos: List[str]) -> Dict[str, dict]:
	"""Load aggregated SHAP/1-SII JSONs for selected phenotypes."""
	out = {}
	for pheno in phenos:
		with open(os.path.join(save_dir, f"{pheno}_agg_shap.json"), "r") as f:
			out[pheno] = json.load(f)
	return out


def build_1sii_long_df(
	all_shap: Dict[str, dict],
	pheno: str,
	agg_method: str = "mean",
	feature_map: Dict[str, str] = None
) -> pd.DataFrame:
	"""
	Return a tidy DataFrame with rows for:
	  - individual effects (diagonal)
	  - pairwise interactions (upper triangle)
	Columns: ['Label', 'Value', 'Kind']
	"""
	shap = all_shap[pheno]
	features_raw = shap["feature_names"]
	features = [feature_map.get(f, f) for f in features_raw] if feature_map else features_raw

	M = np.array(shap['1-SII'][agg_method])  # n_features x n_features

	rows = []
	n = len(features)
	for i in range(n):
		# Diagonal: individual effect
		rows.append({
			"Label": f"{features[i]}",
			"Value": float(M[i, i]),
			"Kind": "Individual",
		})
		# Upper triangle: pairwise i<j
		for j in range(i + 1, n):
			rows.append({
				"Label": f"{features[i]} × {features[j]}",
				"Value": float(M[i, j]),
				"Kind": "Pair",
			})

	df = pd.DataFrame(rows)
	return df


def plot_1sii_bars(
	df: pd.DataFrame,
	pheno: str,
	top_k: int = None,
	sort_desc: bool = True,
	save_path: str = None
):
	"""
	Horizontal barplot of 1-SII values (individual + pairs) for one phenotype.
	Individual bars are colored differently from pairwise interactions.
	"""
	# Sort and optionally truncate
	df_plot = df.sort_values("Value", ascending=not sort_desc)
	if top_k is not None:
		df_plot = df_plot.head(top_k)

	# Order labels in plotting order (top at top)
	order = list(df_plot["Label"])[::-1]  # reverse so largest appears at top in hplot
	height = max(2.0, 0.28 * len(df_plot) + 1.2)  # adaptive height

	# Colors: make "Individual" pop, keep "Pair" lighter
	palette = {
		"Individual": "#D52B00",  # vivid red-orange (stands out)
		"Pair": "#9aa5b1",        # neutral gray-blue
	}

	fig, ax = plt.subplots(figsize=(9, height))
	sns.barplot(
		data=df_plot,
		y="Label",
		x="Value",
		hue="Kind",
		order=order,
		orient="h",
		dodge=False,
		palette=palette,
		ax=ax,
		linewidth=0.5,
		edgecolor="black",
	)

	ax.set_title(f"1-SII — {pheno} ({AGG_METHOD})")
	ax.set_xlabel("1-SII")
	ax.set_ylabel("")

	# Clean up grid & legend
	ax.grid(axis="x", linestyle="-", alpha=0.5)
	ax.grid(axis="y", linestyle="-", alpha=0.2)
	ax.set_axisbelow(True)
	ax.legend(title="", loc="best", frameon=False)

	plt.tight_layout()
	
	if save_path is not None:
		plt.savefig(save_path)
	else:
		plt.show()
		  
	plt.close()


if __name__ == "__main__":
	if not PHENOS_TO_PLOT:
		raise ValueError("No phenotypes specified in PHENOS_TO_PLOT")

	# Load only the phenotypes we plan to plot
	all_shap = load_shap_values(SAVE_DIR, PHENOS_TO_PLOT)

	# Plot one phenotype per figure, sequentially
	for ph in PHENOS_TO_PLOT:
		df_1sii = build_1sii_long_df(
			all_shap=all_shap,
			pheno=ph,
			agg_method=AGG_METHOD,
			feature_map=FEATURE_NAMES_MAP
		)
		plot_1sii_bars(
			df=df_1sii,
			pheno=ph,
			top_k=TOP_K,
			sort_desc=SORT_DESC,
			save_path=os.path.join(PLOT_SAVE_DIR, f"{ph}_1sii_bar_plot.png")
		)