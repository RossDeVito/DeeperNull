"""Evaluate and plot aggregated Shapley and 1-SII values."""

import os
import json
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import matplotlib
import seaborn as sns
from shapiq.plot.network import(
	_order_nodes,
	_add_weight_to_edges_in_graph
)
import networkx as nx


# Set figure DPI for saving figs
import matplotlib as mpl
mpl.rcParams['savefig.dpi'] = 1200


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



def load_shap_values(save_dir, phenos):
	"""Load aggregated SHAP values."""
	all_shap = {}
	for pheno in phenos:
		with open(f'{save_dir}/{pheno}_agg_shap.json', 'r') as f:
			all_shap[pheno] = json.load(f)

	return all_shap


# --- Modified network_plot function ---
def network_plot(
	first_order_values=None,
	second_order_values=None,
	feature_names=None,
	center_text=None,
	draw_labels=True,
	label_fontsize=None,
	edge_color_mode='gray', # New argument: 'gray', 'mean', 'dominant_node'
	show=False,
	ax=None,
):
	"""Draws a simplified interaction network plot with edge color options.

	Nodes fill/border colors use 'tab10' by feature order. Sizes use first_order_values.
	Edge widths use second_order_values. Edge colors determined by edge_color_mode.

	Args:
		first_order_values: Positive first order values (n_features,).
		second_order_values: Positive second order interaction values (n_features, n_features).
		feature_names: Feature names for labels and node color order.
		center_text: Text to display in the center. Defaults to None.
		draw_labels: Whether to draw feature name labels next to nodes. Defaults to True.
		label_fontsize: Font size for labels. Defaults to None (matplotlib default).
		edge_color_mode: How to color edges: 'gray' (default), 'mean' (average of
						 connected node colors), 'dominant_node' (color of node with
						 higher first_order_value).
		draw_legend: Whether to draw a legend. Defaults to False. (Requires adapted function).
		show: Whether to show the plot. Defaults to False.
		ax: A matplotlib Axes object to draw onto. If None, a new figure/axis is created.

	Returns:
		A tuple (figure, axis) if show=False, otherwise None.
	"""
	valid_edge_modes = ['gray', 'mean', 'dominant_node']
	if edge_color_mode not in valid_edge_modes:
		raise ValueError(f"edge_color_mode must be one of {valid_edge_modes}, got '{edge_color_mode}'")

	if ax is None:
		fig, current_ax = plt.subplots(figsize=(6, 6))
		created_fig = True
	else:
		current_ax = ax
		fig = current_ax.get_figure()
		created_fig = False
	current_ax.axis("off")

	# --- Input Validation ---
	if first_order_values is None or second_order_values is None:
		raise ValueError("Both first_order_values and second_order_values must be provided.")
	if np.any(first_order_values < 0) or np.any(second_order_values < 0):
		 print("Warning: Negative values detected; node sizes/edge widths assume positive values.")
	n_features = first_order_values.shape[0]
	if feature_names is None: feature_names = [str(i + 1) for i in range(n_features)]
	elif len(feature_names) != n_features: raise ValueError(f"Feature name length mismatch.")

	# --- Graph Creation and Node Attributes ---
	graph = nx.complete_graph(n_features)
	nodes_visit_order = _order_nodes(len(graph.nodes))

	# Add node sizes/linewidths and edge widths (color logic moved)
	_add_weight_to_edges_in_graph(
		graph=graph, first_order_values=first_order_values, second_order_values=second_order_values,
		n_features=n_features, feature_names=feature_names, nodes_visit_order=nodes_visit_order,
	)

	# Assign Node Fill/Border Colors and store original index
	# Use matplotlib.colormaps to get colors (addresses deprecation)
	tab10_colors = matplotlib.colormaps['tab10'].colors
	for i in range(n_features):
		node = nodes_visit_order[i]
		color_index = i % len(tab10_colors)
		current_color = tab10_colors[color_index]
		graph.nodes[node]['node_color'] = current_color # Fill color
		graph.nodes[node]['edgecolors'] = current_color # Border color
		graph.nodes[node]['original_index'] = i        # Store original index for edge coloring

	# --- Set Edge Colors based on mode ---
	for u, v in graph.edges():
		color_u = graph.nodes[u]['node_color']
		color_v = graph.nodes[v]['node_color']

		if edge_color_mode == 'gray':
			graph.edges[u, v]['color'] = 'gray'
		elif edge_color_mode == 'mean':
			# Calculate mean RGB color
			mean_color = tuple(np.mean([color_u, color_v], axis=0))
			graph.edges[u, v]['color'] = mean_color
		elif edge_color_mode == 'dominant_node':
			# Get original indices and values to find dominant node
			idx_u = graph.nodes[u]['original_index']
			idx_v = graph.nodes[v]['original_index']
			val_u = first_order_values[idx_u]
			val_v = first_order_values[idx_v]
			# Assign color of the node with the higher first_order_value
			dominant_color = color_u if val_u >= val_v else color_v
			graph.edges[u, v]['color'] = dominant_color

	# --- Get Attributes for Drawing ---
	node_colors = [graph.nodes[node]['node_color'] for node in graph.nodes()]
	node_edge_colors = [graph.nodes[node]['edgecolors'] for node in graph.nodes()]
	node_sizes = list(nx.get_node_attributes(graph, "node_size").values()) or [300] * n_features
	node_line_widths = list(nx.get_node_attributes(graph, "linewidths").values()) or [1.0] * n_features
	# Get edge colors and widths assigned above or by helper
	edge_colors = [graph.edges[u,v]['color'] for u,v in graph.edges()] or ['gray'] * graph.number_of_edges()
	edge_widths = list(nx.get_edge_attributes(graph, "width").values()) or [1.0] * graph.number_of_edges()

	# Calculate edge alphas
	max_width = max(edge_widths) if edge_widths else 1.0
	min_alpha, max_alpha = 0.1, 0.8 # Adjust alpha range as needed
	if max_width > 0: edge_alphas = [min_alpha + (max_alpha - min_alpha) * (w / max_width) for w in edge_widths]
	else: edge_alphas = [min_alpha] * len(edge_widths)

	# --- Drawing ---
	pos = nx.circular_layout(graph)
	nx.draw_networkx_edges(graph, pos, width=edge_widths, edge_color=edge_colors, alpha=edge_alphas, ax=current_ax)
	nx.draw_networkx_nodes(
		graph, pos, node_color=node_colors, node_size=node_sizes,
		linewidths=node_line_widths, edgecolors=node_edge_colors, ax=current_ax,
	)

	# Draw Labels (Optional)
	if draw_labels:
		# --- Tunable Parameters ---
		vertical_adjust_factor = 0.1  # Controls vertical shift based on y-sign
		# horizontal_adjust_factor = 0.3 # REMOVED - Replaced by alignment logic
		alignment_threshold = 0.2      # X-value threshold for switching to left/right alignment
		# --- End Tunable Parameters ---

		for i, node_idx in enumerate(nodes_visit_order):
			node = node_idx
			(x_pos, y_pos) = pos[node] # Node position
			label = feature_names[i]   # Label string

			# --- Use Original Logic for base distance/angle ---
			linewidth = graph.nodes[node].get("linewidths", 1.0)
			radius = 1.15 + linewidth / 300 # Original radius

			theta = np.arctan2(x_pos, y_pos)
			label_display = label
			if abs(theta) <= 0.001: label_display = "\n" + label # Add newline if near y-axis
			theta = np.pi / 2 - theta
			if theta < 0: theta += 2 * np.pi

			# Calculate base text position
			x_text = radius * np.cos(theta)
			y_text = radius * np.sin(theta)
			# --- End Original Logic ---

			# --- Apply Vertical Adjustment ---
			y_sign = np.sign(y_text)
			if abs(y_text) < 1e-6: y_sign = 1 # Default up if close to 0
			y_final = y_text + y_sign * vertical_adjust_factor
			# --- End Vertical Adjustment ---

			# --- Determine Horizontal Alignment based on x_text ---
			if x_text > alignment_threshold:
				ha = 'left'  # Align left if text is significantly to the right
			elif x_text < -alignment_threshold:
				ha = 'right' # Align right if text is significantly to the left
			else: # Includes cases between -threshold and +threshold
				ha = 'center'# Default to center alignment near the middle
			# --- End Horizontal Alignment Determination ---

			# Draw the text label using base x_text and adjusted y_final, with conditional alignment
			current_ax.text(
				x_text, y_final, label_display, # Use base x_text, adjusted y_final
				horizontalalignment=ha,         # Use determined alignment
				verticalalignment="center",     # Keep vertical center alignment
				transform=current_ax.transData,
				fontsize=label_fontsize
			)

	# Center Text (Optional)
	if center_text is not None:
		background_color, line_color = '#ffffff', '#000000'
		current_ax.text(
			0, 0, center_text,
			horizontalalignment="center", verticalalignment="center",
			bbox=dict(facecolor=background_color, alpha=0.5, edgecolor=line_color, pad=7),
			color="black", fontsize=plt.rcParams.get("font.size", 10) + 3,
			transform=current_ax.transData, zorder=15
		)

	# --- Final Adjustments ---
	limit_buffer = 1.7 if draw_labels else 1.5
	current_ax.set_xlim(-limit_buffer, limit_buffer)
	current_ax.set_ylim(-limit_buffer, limit_buffer)
	current_ax.set_aspect('equal', adjustable='box')

	# --- Return or Show ---
	if created_fig and show:
		plt.show()
		return None
	return fig, current_ax


if __name__ == '__main__':

	# Options
	save_dir = 'agg_shap'
	phenos = [
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

	# Load aggregated SHAP values
	all_shap = load_shap_values(save_dir, phenos)

	# Convert Shapley values to DataFrame that can be used with seaborn
	shap_rows = []

	for pheno, shap in all_shap.items():
		
		# Add mean Shapley values
		for feat, val in zip(shap['feature_names'], shap['Shapley']['mean']):
			shap_rows.append({
				'Phenotype': pheno,
				'Feature': FEATURE_NAMES_MAP[feat],
				'Value type': 'Mean Shapley value',
				'Value': val
			})
		
		# Add median Shapley values
		for feat, val in zip(shap['feature_names'], shap['Shapley']['median']):
			shap_rows.append({
				'Phenotype': pheno,
				'Feature': FEATURE_NAMES_MAP[feat],
				'Value type': 'Median Shapley value',
				'Value': val
			})

		# Add std dev of Shapley values
		for feat, val in zip(shap['feature_names'], shap['Shapley']['std']):
			shap_rows.append({
				'Phenotype': pheno,
				'Feature': FEATURE_NAMES_MAP[feat],
				'Value type': 'Std. dev. of Shapley values',
				'Value': val
			})

	shap_df = pd.DataFrame(shap_rows)

	# Plot agg Shapley values without interactions
	# Plot should be 4x4 grid of bar plots plotted using sns.catplot
	
	measure = 'Mean Shapley value'
	# measure = 'Median Shapley value'
	# measure = 'Std. dev. of Shapley values'

	g = sns.catplot(
		data=shap_df[shap_df['Value type'] == measure],
		x='Feature',
		y='Value',
		col='Phenotype',
		hue='Feature',
		kind='bar',
		col_wrap=4,
		sharey=False,
		height=2.5,
		aspect=1,
		legend=True,
		legend_out=True
	)

	g.set_titles(col_template="{col_name}")

	# Remove x-axis tick labels from each subplot
	for ax in g.axes.flat:
		ax.set_xticklabels([])

	# Add minor y-axis grid lines
	for ax in g.axes.flat:
		ax.set_axisbelow(True)
		ax.yaxis.set_minor_locator(AutoMinorLocator())
		ax.grid(which='minor', axis='y', linestyle='-', alpha=0.3)
		ax.grid(which='major', axis='y', linestyle='-', alpha=0.6)

	plt.suptitle(f'{measure} by phenotype')

	plt.show()

	# Plot single phenotype as network
	pheno = 'standing_height_50'

	network_plot(
		first_order_values=np.array(all_shap[pheno]['Shapley']['mean']),
		second_order_values=np.array(all_shap[pheno]['1-SII']['mean']),
		feature_names=[
			FEATURE_NAMES_MAP[feat] for feat in all_shap[pheno]['feature_names']
		],
	)

	plt.show()

	# Plot max width 4 grid of network plots
	max_cols = 4
	agg_method = ['mean', 'median', 'std'][0]

	# Calculate number of rows and columns based on number of phenotypes
	n_pheno = len(phenos)
	n_cols = min(max_cols, n_pheno)
	n_rows = math.ceil(n_pheno / n_cols)

	per_plot_size = 2
	fig, axs = plt.subplots(
		n_rows,
		n_cols,
		figsize=(per_plot_size * n_cols, per_plot_size * n_rows)
	)

	# Make sure axs is 2D
	axs = np.atleast_2d(axs)

	for i, pheno in enumerate(phenos):
		ax = axs[i // n_cols, i % n_cols]

		network_plot(
			first_order_values=np.array(all_shap[pheno]['Shapley'][agg_method]),
			second_order_values=np.array(all_shap[pheno]['1-SII'][agg_method]),
			feature_names=[
				FEATURE_NAMES_MAP[feat] for feat in all_shap[pheno]['feature_names']
			],
			ax=ax,
			edge_color_mode='mean',
			draw_labels=False,
		)

		ax.set_title(pheno)

	# Turn off unused subplots if any
	for j in range(n_pheno, n_rows * n_cols):
		axs[j // n_cols, j % n_cols].axis('off')

	plt.tight_layout()
	if agg_method == 'std':
		agg_method_title = 'Std. deviation'
	else:
		agg_method_title = agg_method.capitalize()
	plt.suptitle(f'{agg_method_title} of Shapley and 1-SII values')

	# tight layout to avoid overlap with suptitle
	plt.subplots_adjust(top=0.9)

	# Create legend
	leg_fig, leg_ax = network_plot(
		first_order_values=np.ones_like(all_shap[pheno]['Shapley']['mean']) * 2,
		second_order_values=np.zeros_like(all_shap[pheno]['1-SII']['mean']),
		feature_names=[
			FEATURE_NAMES_MAP[feat] for feat in all_shap[pheno]['feature_names']
		],
		# draw_labels=False,
	)
	leg_ax.set_title('Legend')

	plt.show()
