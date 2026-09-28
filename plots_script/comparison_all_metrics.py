import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter


# ============================================================
# Configuration
# ============================================================

files = glob.glob("results/*_metrics.csv")

if not files:
    raise SystemExit("no result csv files found in results/")

metrics = [
    "latency",
    "amse",
    "energy",
    "cvar5",
    "cvar95",
    "cvar90",
    "cvar99",
    "num_sat",
    "num_uav",
    "num_ground",
]

# IMPORTANT:
# Keep the proposed algorithm name consistent everywhere.
KNOWN_ALGOS = [
    "dr",
    "da",
    "lop",
    "go",
    "nrs",
    "random",
    "fedsn",
    "hsfl",
]

DISPLAY_NAME = {
    "dr_greedy": "DR-Greedy",
    "lop": "LOP",
    "go": "GO",
    "nrs": "NRS",
    "random": "Random",
    "da": "DA",
    "fedsn": "FedSN",
    "hsfl": "HSFL",
}


# ============================================================
# IEEE-style plotting configuration
# ============================================================

# Similar visual appearance to LaTeX/IEEE papers.
# Computer Modern is used if available; otherwise matplotlib
# falls back to its default serif font.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": [
        "CMU Serif",
        "Computer Modern Roman",
        "DejaVu Serif",
    ],

    "font.size": 11,
    "axes.titlesize": 11,
    "axes.labelsize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,

    "axes.linewidth": 0.8,

    "lines.linewidth": 1.5,

    "figure.dpi": 200,
    "savefig.dpi": 300,

    "mathtext.fontset": "cm",

    "axes.grid": True,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.30,

    "legend.frameon": True,
    "legend.framealpha": 0.90,
    "legend.edgecolor": "0.7",
})


# ============================================================
# Consistent colors, markers and line styles
# ============================================================

# DR-Greedy gets a fixed, distinctive color.
# It will therefore look identical in every figure.
DR_GREEDY_COLOR = "#D62728"   # strong red

# Colors for the remaining algorithms.
COLORS = {
    "dr_greedy": DR_GREEDY_COLOR,
    "da": "#1F77B4",
    "lop": "#2CA02C",
    "go": "#9467BD",
    "nrs": "#8C564B",
    "random": "#17BECF",
    "fedsn": "#FF7F0E",
    "hsfl": "#7F7F7F",
}

MARKERS = {
    "dr_greedy": "o",
    "da": "s",
    "lop": "^",
    "go": "D",
    "nrs": "v",
    "random": "P",
    "fedsn": "X",
    "hsfl": "*",
}

LINE_STYLES = {
    "dr_greedy": "-",       # proposed method
    "da": "--",
    "lop": "-.",
    "go": ":",
    "nrs": "--",
    "random": "-.",
    "fedsn": ":",
    "hsfl": "--",
}


# ============================================================
# Filename parsing
# ============================================================

def parse_algo_and_run(base_name: str):
    """
    Parse <algo>[_<tag>] from filename stem.

    Example:
        dr_greedy_metrics.csv
        dr_greedy_run1_metrics.csv
        da_run2_metrics.csv

    The parser is robust to algorithm names containing underscores.
    """

    for algo in sorted(KNOWN_ALGOS, key=len, reverse=True):

        if base_name == algo:
            return algo, "run0"

        prefix = algo + "_"

        if base_name.startswith(prefix):
            return algo, base_name[len(prefix):]

    return None, None


# ============================================================
# Load all result files
# ============================================================

rows = []

for f in files:

    base = os.path.basename(f).replace("_metrics.csv", "")

    algo, run_tag = parse_algo_and_run(base)

    if algo is None:
        print(f"skipping unrecognized metrics file: {base}")
        continue

    df = pd.read_csv(f)

    df["algo"] = algo
    df["run"] = run_tag

    rows.append(df)


if not rows:
    raise SystemExit("no recognized result csvs after parsing")


all_df = pd.concat(rows, ignore_index=True)


# ============================================================
# Algorithm ordering
# ============================================================

algorithms = sorted(
    all_df["algo"].unique(),
    key=lambda x: KNOWN_ALGOS.index(x)
)


# ============================================================
# Plot titles and axis labels
# ============================================================

PLOT_INFO = {
    "latency": {
        "title": "Latency",
        "ylabel": "Latency",
        "log": False,
    },

    "amse": {
        "title": "AMSE",
        "ylabel": "AMSE",
        "log": True,
    },

    "energy": {
        "title": "Energy Consumption",
        "ylabel": "Energy (K)",
        "log": False,
    },

    "cvar5": {
        "title": r"CVaR@5\%",
        "ylabel": r"CVaR@5\%",
        "log": True,
    },

    "cvar95": {
        "title": r"CVaR@95\%",
        "ylabel": r"CVaR@95\%",
        "log": True,
    },

    "cvar90": {
        "title": r"CVaR@90\%",
        "ylabel": r"CVaR@90\%",
        "log": True,
    },

    "cvar99": {
        "title": r"CVaR@99\%",
        "ylabel": r"CVaR@99\%",
        "log": True,
    },

    "num_sat": {
        "title": "Number of Satellites",
        "ylabel": "Number of Satellites",
        "log": False,
    },

    "num_uav": {
        "title": "Number of UAVs",
        "ylabel": "Number of UAVs",
        "log": False,
    },

    "num_ground": {
        "title": "Number of Ground BSs",
        "ylabel": "Number of Ground BSs",
        "log": False,
    },
}


# ============================================================
# Output directory
# ============================================================

os.makedirs("plots", exist_ok=True)


# ============================================================
# Generate one figure per metric
# ============================================================

for metric in metrics:

    if metric not in all_df.columns:
        print(f"skipping {metric}: column not found")
        continue

    info = PLOT_INFO[metric]

    # --------------------------------------------------------
    # IEEE-friendly figure size
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(8, 4.6))    


    # --------------------------------------------------------
    # Plot every algorithm
    # --------------------------------------------------------

    for algo in algorithms:

        df_a = all_df[all_df["algo"] == algo]

        grouped = df_a.groupby("step")

        mu = grouped[metric].mean().sort_index()

        std = (
            grouped[metric]
            .std(ddof=1)
            .fillna(0.0)
            .reindex(mu.index)
        )

        n = (
            grouped[metric]
            .count()
            .reindex(mu.index)
            .clip(lower=1)
        )

        # 95% confidence interval
        ci = 1.96 * std / np.sqrt(n)

        x = mu.index.values
        y = mu.values

        lower = (mu - ci).values
        upper = (mu + ci).values


        # ----------------------------------------------------
        # Log-scale plots cannot have <= 0 values
        # ----------------------------------------------------

        if info["log"]:

            lower = np.clip(lower, 1e-12, None)
            upper = np.clip(upper, 1e-12, None)
            y = np.clip(y, 1e-12, None)


        # ----------------------------------------------------
        # Plot line
        # ----------------------------------------------------

        ax.plot(
            x,
            y,

            label=DISPLAY_NAME.get(algo, algo),

            color=COLORS.get(algo, None),

            linestyle=LINE_STYLES.get(algo, "-"),

            marker=MARKERS.get(algo, "o"),

            linewidth=1.6,

            markersize=4.5,

            markevery=max(1, len(x) // 12),

            markeredgewidth=0.6,

            markeredgecolor="black",

            zorder=3 if algo == "dr_greedy" else 2,
        )


        # ----------------------------------------------------
        # Confidence interval
        # ----------------------------------------------------

        ax.fill_between(
            x,
            lower,
            upper,

            color=COLORS.get(algo, None),

            alpha=0.08,

            linewidth=0,

            zorder=1,
        )


    # ========================================================
    # Axis formatting
    # ========================================================

    ax.set_xlabel(
        "Simulation Step",
        fontsize=14,
    )

    ax.set_ylabel(
        info["ylabel"],
        fontsize=14,
    )


    # --------------------------------------------------------
    # Log scale
    # --------------------------------------------------------

    if info["log"]:
        ax.set_yscale("log")
        
    # Format energy axis as K (thousands)
    if metric == "energy":
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda x, pos: f"{x/1000:g}K")
        )


    # ========================================================
    # Reduce unnecessary white space
    # ========================================================

    # Determine visible data range.
    visible_values = []

    for algo in algorithms:

        df_a = all_df[all_df["algo"] == algo]

        values = df_a[metric].dropna().values

        if len(values) > 0:
            visible_values.extend(values)


    if visible_values:

        visible_values = np.asarray(visible_values)

        visible_values = visible_values[
            np.isfinite(visible_values)
        ]

        if len(visible_values) > 0:

            ymin = visible_values.min()
            ymax = visible_values.max()

            if info["log"]:

                # Small logarithmic margin.
                ax.set_ylim(
                    ymin / 1.15,
                    ymax * 1.15,
                )

            else:

                data_range = ymax - ymin

                if data_range == 0:
                    data_range = max(abs(ymax), 1.0)

                # Only a small amount of space around the data.
                margin = data_range * 0.06

                ax.set_ylim(
                    ymin - margin,
                    ymax + margin,
                )


    # ========================================================
    # Grid
    # ========================================================

    ax.grid(
        True,
        which="major",
        linestyle="--",
        linewidth=0.5,
        alpha=0.30,
    )

    ax.grid(
        True,
        which="minor",
        linestyle=":",
        linewidth=0.35,
        alpha=0.20,
    )


    # ========================================================
    # Legend — centered above the plot
    # ========================================================
    legend = fig.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=4,
        fontsize=10,
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        edgecolor="0.45",
        facecolor="white",
        borderpad=0.6,
        handlelength=2.8,
        handletextpad=0.6,
        columnspacing=1.5,
        labelspacing=0.5,
    )


    # ========================================================
    # Spines / layout
    # ========================================================

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(
        direction="out",
        length=3.5,
        width=0.7,
    )

    ax.margins(x=0.01)

    # Leave dedicated space above the axes for the legend.
    fig.subplots_adjust(
        left=0.09,
        right=0.98,
        bottom=0.16,
        top=0.78,
    )


    # ========================================================
    # Save individual figure
    # ========================================================

    output_path = os.path.join(
        "plots",
        f"{metric}.png",
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.04,
    )

    plt.close(fig)

    print(f"saved {output_path}")
    

# ========================================================
# Pareto Front Analysis
# ========================================================

PARETO_PLOTS = [
    (
        "latency",
        "amse",
        "Latency",
        "AMSE",
        "pareto_latency_amse.png",
    ),
    (
        "latency",
        "energy",
        "Latency",
        "Energy",
        "pareto_latency_energy.png",
    ),
    (
        "amse",
        "energy",
        "AMSE",
        "Energy",
        "pareto_amse_energy.png",
    ),
]


def get_pareto_mask(x, y):
    """
    Return a boolean mask identifying Pareto-optimal points.

    Both x and y are assumed to be minimized.
    """
    points = np.column_stack((x, y))

    is_pareto = np.ones(len(points), dtype=bool)

    for i in range(len(points)):
        if not is_pareto[i]:
            continue

        # A point is dominated if another point is:
        # <= in both objectives and < in at least one.
        dominated = np.all(points <= points[i], axis=1) & \
                    np.any(points < points[i], axis=1)

        if np.any(dominated):
            is_pareto[i] = False

    return is_pareto


for x_metric, y_metric, x_label, y_label, filename in PARETO_PLOTS:

    # ----------------------------------------------------
    # Prepare data
    # ----------------------------------------------------

    pareto_df = df[
        df[x_metric].notna() &
        df[y_metric].notna()
    ].copy()

    if pareto_df.empty:
        continue

    x = pareto_df[x_metric].to_numpy()
    y = pareto_df[y_metric].to_numpy()

    # ----------------------------------------------------
    # Find global Pareto front
    # ----------------------------------------------------

    pareto_mask = get_pareto_mask(x, y)

    pareto_points = pareto_df[pareto_mask].copy()

    # Sort Pareto points by x for drawing the front
    pareto_points = pareto_points.sort_values(
        by=x_metric
    )

    # ----------------------------------------------------
    # Create figure
    # ----------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(8, 4.6)
    )

    # ----------------------------------------------------
    # Plot all experimental points
    # ----------------------------------------------------

    for algo in KNOWN_ALGOS:

        algo_df = pareto_df[
            pareto_df["algo"] == algo
        ]

        if algo_df.empty:
            continue

        ax.scatter(
            algo_df[x_metric],
            algo_df[y_metric],
            color=COLORS.get(algo, "#000000"),
            marker=MARKERS.get(algo, "o"),
            s=42,
            alpha=0.55,
            edgecolors="black",
            linewidths=0.4,
            zorder=2,
        )

    # ----------------------------------------------------
    # Highlight global Pareto front
    # ----------------------------------------------------

    ax.plot(
        pareto_points[x_metric],
        pareto_points[y_metric],
        color="black",
        linestyle="--",
        linewidth=1.8,
        zorder=4,
    )

    ax.scatter(
        pareto_points[x_metric],
        pareto_points[y_metric],
        facecolors="none",
        edgecolors="black",
        marker="o",
        s=90,
        linewidths=1.5,
        zorder=5,
    )

    # ----------------------------------------------------
    # Label Pareto points with algorithm names
    # ----------------------------------------------------

    for _, row in pareto_points.iterrows():

        algo = row["algo"]

        ax.annotate(
            DISPLAY_NAME.get(algo, algo),
            (
                row[x_metric],
                row[y_metric],
            ),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            color=COLORS.get(algo, "#000000"),
            zorder=6,
        )

    # ----------------------------------------------------
    # Axis labels
    # ----------------------------------------------------

    ax.set_xlabel(
        x_label,
        fontsize=16,
    )

    ax.set_ylabel(
        y_label,
        fontsize=16,
    )

    ax.tick_params(
        axis="both",
        labelsize=10,
    )

    # ----------------------------------------------------
    # Energy axis in K
    # ----------------------------------------------------

    if y_metric == "energy":

        ax.yaxis.set_major_formatter(
            FuncFormatter(
                lambda value, position:
                f"{value / 1000:g}K"
            )
        )

    if x_metric == "energy":

        ax.xaxis.set_major_formatter(
            FuncFormatter(
                lambda value, position:
                f"{value / 1000:g}K"
            )
        )

    # ----------------------------------------------------
    # Grid
    # ----------------------------------------------------

    ax.grid(
        True,
        which="major",
        linestyle="--",
        linewidth=0.5,
        alpha=0.30,
    )

    ax.margins(
        x=0.05,
        y=0.08,
    )

    # ----------------------------------------------------
    # Layout
    # ----------------------------------------------------

    fig.subplots_adjust(
        left=0.10,
        right=0.98,
        bottom=0.16,
        top=0.95,
    )

    # ----------------------------------------------------
    # Save
    # ----------------------------------------------------

    output_path = os.path.join(
        "plots",
        filename,
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.04,
    )

    plt.close(fig)

    print(
        f"Saved Pareto plot: {output_path}"
    )


# ============================================================
# Summary table
# ============================================================

valid_metrics = [
    m for m in metrics
    if m in all_df.columns
]

summary = (
    all_df
    .groupby("algo")[valid_metrics]
    .mean()
    .sort_values("amse")
)

summary.index = [
    DISPLAY_NAME.get(idx, idx)
    for idx in summary.index
]

summary.to_csv(
    "results/summary_metrics.csv"
)

print("saved results/summary_metrics.csv")