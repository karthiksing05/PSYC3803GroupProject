from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso, Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from pathlib import Path
try:
	import pacmap
	HAS_PACMAP = True
except Exception:
	HAS_PACMAP = False


def main():
	# file paths
	scores_fp = "data/DDM_Scores.csv"
	feats_fp = "data/resting_subject_features.csv"

	# features to use as X
	feature_cols = [
		"posterior_alpha",
		"posterior_beta",
		"frontal_theta",
		"frontal_theta_beta_ratio",
		"global_alpha",
		"global_beta",
		"IAF",
		"aperiodic_exponent",
	]

	# load data
	df_scores = pd.read_csv(scores_fp)
	df_feats = pd.read_csv(feats_fp)

	# evaluate for all available Detailed_Condition values
	conditions = df_scores["Detailed_Condition"].dropna().unique()

	out_dir = Path("output")
	out_dir.mkdir(parents=True, exist_ok=True)

	def sanitize(name: str) -> str:
		return "".join([c if c.isalnum() or c in ('_', '-') else '_' for c in str(name)])

	targets = ["Score", "a", "v"]

	for cond in sorted(conditions):
		cond_key = sanitize(cond)
		cond_dir = out_dir / cond_key
		cond_dir.mkdir(parents=True, exist_ok=True)

		df_scores_cond = df_scores[df_scores["Detailed_Condition"] == cond][["Subject"] + targets]
		df_feats_sub = df_feats[["Subject"] + feature_cols]
		df = pd.merge(df_scores_cond, df_feats_sub, on="Subject", how="inner")
		df = df.dropna(subset=targets + feature_cols)

		if df.shape[0] < 5:
			(cond_dir / "note.txt").write_text(f"Not enough subjects after merge for condition {cond} (n={df.shape[0]})")
			print(f"Skipping {cond}: insufficient data (n={df.shape[0]})")
			continue

		y = df[targets].values
		X = df[feature_cols].values

		scaler = StandardScaler()
		Xs = scaler.fit_transform(X)

		lin = LinearRegression()
		ridge = Ridge(alpha=1.0)
		lasso = MultiOutputRegressor(Lasso(alpha=1.0, max_iter=10000))

		lin.fit(Xs, y)
		ridge.fit(Xs, y)
		lasso.fit(Xs, y)

		preds_lin = lin.predict(Xs)
		preds_ridge = ridge.predict(Xs)
		preds_lasso = lasso.predict(Xs)

		r2_lin = r2_score(y, preds_lin, multioutput='raw_values')
		r2_ridge = r2_score(y, preds_ridge, multioutput='raw_values')
		r2_lasso = r2_score(y, preds_lasso, multioutput='raw_values')

		df_lin_coef = pd.DataFrame(lin.coef_.T, index=feature_cols, columns=targets)
		df_ridge_coef = pd.DataFrame(ridge.coef_.T, index=feature_cols, columns=targets)
		lasso_coefs = np.vstack([est.coef_ for est in lasso.estimators_])
		df_lasso_coef = pd.DataFrame(lasso_coefs.T, index=feature_cols, columns=targets)

		intercept_lin = lin.intercept_
		intercept_ridge = ridge.intercept_
		intercept_lasso = np.array([est.intercept_ for est in lasso.estimators_])

		def write_results_file(path: Path, title: str, df_coef: pd.DataFrame, intercepts, r2_vals):
			lines = [title, "\nCoefficients (rows=features, cols=targets):\n", df_coef.to_string(), "\nIntercepts:\n", str(dict(zip(targets, intercepts))), "\nR^2 per target:\n", str(dict(zip(targets, r2_vals)))]
			path.write_text("\n".join(lines))

		write_results_file(cond_dir / f"linear_results_{cond_key}.txt", f"LinearRegression Results ({cond})", df_lin_coef, intercept_lin, r2_lin)
		write_results_file(cond_dir / f"lasso_results_{cond_key}.txt", f"Lasso (MultiOutput) Results ({cond})", df_lasso_coef, intercept_lasso, r2_lasso)
		write_results_file(cond_dir / f"ridge_results_{cond_key}.txt", f"Ridge Results ({cond})", df_ridge_coef, intercept_ridge, r2_ridge)

		# PCA and plotting per condition
		pca = PCA(n_components=2)
		X_pca = pca.fit_transform(Xs)

		n_targets = y.shape[1]
		fig, axes = plt.subplots(2, 2, figsize=(12, 10))
		axes = axes.flatten()

		for i, ax in enumerate(axes[:n_targets]):
			vals = y[:, i]
			sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=vals, cmap='viridis', s=35)
			cb = fig.colorbar(sc, ax=ax)
			cb.set_label(targets[i])
			ax.set_xlabel('PCA 1')
			ax.set_ylabel('PCA 2')
			ax.set_title(f'PCA(2) colored by {targets[i]}')

		if n_targets < len(axes):
			for ax in axes[n_targets:]:
				ax.axis('off')

		fig.tight_layout()
		plot_path = cond_dir / f"pca_2d_scatter_targets_{cond_key}.png"
		fig.savefig(plot_path, dpi=150, bbox_inches='tight')
		plt.close(fig)

		# PaCMAP visualization (optional) with safety checks
		pac_note_path = cond_dir / "pacmap_note.txt"
		if HAS_PACMAP:
			n_samples = Xs.shape[0]
			if n_samples < 3:
				pac_note_path.write_text(f"Not enough samples for PaCMAP (n={n_samples}); need >=3")
				print(f"Skipping PaCMAP for {cond}: not enough samples (n={n_samples})")
			else:
				try:
					# ensure float32 input and set reasonable n_neighbors
					Xs_pac = np.asarray(Xs, dtype=np.float32)
					n_neighbors = min(10, max(2, n_samples - 1))
					pm = pacmap.PaCMAP(n_components=2, n_neighbors=n_neighbors)
					X_pac = pm.fit_transform(Xs_pac)
					fig, axes = plt.subplots(2, 2, figsize=(12, 10))
					axes = axes.flatten()
					for i, ax in enumerate(axes[:len(targets)]):
						vals = y[:, i]
						sc = ax.scatter(X_pac[:, 0], X_pac[:, 1], c=vals, cmap='viridis', s=35)
						cb = fig.colorbar(sc, ax=ax)
						cb.set_label(targets[i])
						ax.set_xlabel('PaCMAP 1')
						ax.set_ylabel('PaCMAP 2')
						ax.set_title(f'PaCMAP(2) colored by {targets[i]}')
					if len(targets) < len(axes):
						for ax in axes[len(targets):]:
							ax.axis('off')
					fig.tight_layout()
					pm_path = cond_dir / f"pacmap_2d_scatter_targets_{cond_key}.png"
					fig.savefig(pm_path, dpi=150, bbox_inches='tight')
					plt.close(fig)
					pac_note_path.write_text(f"PaCMAP plot saved: {pm_path}")
					print(f"Saved PaCMAP plot for condition {cond} -> {pm_path}")
				except Exception as e:
					pac_note_path.write_text(f"PaCMAP failed: {e}")
					print(f"PaCMAP failed for {cond}: {e}")
		else:
			# note that pacmap isn't available
			pac_note_path.write_text("pacmap not installed; to enable PaCMAP visualizations run: pip install pacmap")

		print(f"Saved results and plot for condition {cond} -> {cond_dir}")


if __name__ == '__main__':
	main()
