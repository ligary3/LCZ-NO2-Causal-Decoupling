# ============================================================
# Optimized LCZ attribute-based group analysis for NO2
#
# Representative days:
#   2024-01-11 Winter stagnant day
#   2024-07-18 Summer convective day
#
# Group definition:
#   Compact built: LCZ 1-3
#   Open built: LCZ 4-6
#   Large low-rise / industrial: LCZ 8, 10
#   Sparsely built: LCZ 9
#   Greenhouse / low-plant transitional areas: LCZ 7, D
#   Woody vegetation: LCZ A-C
#   Paved / bare: LCZ E-F
#   Water: LCZ G
#
# Interpretation:
#   Descriptive LCZ attribute-group contrast, not DML and not causal effect.
#   LCZ 7 and LCZ D are combined because LCZ 7 in the study region may
#   include greenhouse/lightweight structures and can functionally resemble
#   peri-urban low-plant surfaces in NO2 spatial statistics.
# ============================================================

import os
import glob
import re
import warnings
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ============================================================
# 0. Global plotting style
# ============================================================

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.linewidth"] = 1.2
plt.rcParams["axes.unicode_minus"] = False

BASE_FONT = 14

# ============================================================
# 1. Paths and settings
# ============================================================

lcz_dir = r"E:\lunwen3\BTHLCZ100"
no2_root = r"E:\lunwen3\process\预测"

out_dir = r"E:\lunwen3\process\空气质量数据处理\LCZ属性分组分析_合并LCZ7D"
os.makedirs(out_dir, exist_ok=True)

date_list = [
    "20240111",
    "20240718"
]

preferred_keyword = "Ultimate_RBF_v7"

NO2_MIN_VALID = 0
NO2_MAX_VALID = 300

max_sample_per_group = 80000
random_seed = 42

# ============================================================
# 2. LCZ group definition
# ============================================================

# LCZ code:
# 1-10 built types
# 11=A, 12=B, 13=C, 14=D, 15=E, 16=F, 17=G

LCZ_GROUPS = {
    "Compact built\n(LCZ 1-3)": [1, 2, 3],
    "Open built\n(LCZ 4-6)": [4, 5, 6],
    "Large low-rise /\nindustrial\n(LCZ 8, 10)": [8, 10],
    "Sparsely built\n(LCZ 9)": [9],
    "Greenhouse / low-plant\ntransitional\n(LCZ 7, D)": [7, 14],
    "Woody vegetation\n(LCZ A-C)": [11, 12, 13],
    "Paved / bare\n(LCZ E-F)": [15, 16],
    "Water\n(LCZ G)": [17],
}

GROUP_ORDER = [
    "Compact built\n(LCZ 1-3)",
    "Open built\n(LCZ 4-6)",
    "Large low-rise /\nindustrial\n(LCZ 8, 10)",
    "Sparsely built\n(LCZ 9)",
    "Greenhouse / low-plant\ntransitional\n(LCZ 7, D)",
    "Woody vegetation\n(LCZ A-C)",
    "Paved / bare\n(LCZ E-F)",
    "Water\n(LCZ G)"
]

GROUP_LABEL_SHORT = {
    "Compact built\n(LCZ 1-3)": "Compact built\n(1-3)",
    "Open built\n(LCZ 4-6)": "Open built\n(4-6)",
    "Large low-rise /\nindustrial\n(LCZ 8, 10)": "Large low-rise /\nindustrial\n(8,10)",
    "Sparsely built\n(LCZ 9)": "Sparsely built\n(9)",
    "Greenhouse / low-plant\ntransitional\n(LCZ 7, D)": "Greenhouse /\nlow-plant\n(7,D)",
    "Woody vegetation\n(LCZ A-C)": "Woody vegetation\n(A-C)",
    "Paved / bare\n(LCZ E-F)": "Paved / bare\n(E-F)",
    "Water\n(LCZ G)": "Water\n(G)",
}

GROUP_COLORS = {
    "Compact built\n(LCZ 1-3)": "#C44E52",
    "Open built\n(LCZ 4-6)": "#DD8452",
    "Large low-rise /\nindustrial\n(LCZ 8, 10)": "#4C4C4C",
    "Sparsely built\n(LCZ 9)": "#8C8C8C",
    "Greenhouse / low-plant\ntransitional\n(LCZ 7, D)": "#B39B00",
    "Woody vegetation\n(LCZ A-C)": "#55A868",
    "Paved / bare\n(LCZ E-F)": "#8172B2",
    "Water\n(LCZ G)": "#4C72B0",
}

DATE_LABELS = {
    "20240111": "Winter stagnant day",
    "20240718": "Summer convective day"
}

DATE_COLORS = {
    "20240111": "#4C72B0",
    "20240718": "#C44E52"
}

DATE_MARKERS = {
    "20240111": "o",
    "20240718": "s"
}

# ============================================================
# 3. Helper functions: path matching
# ============================================================

def parse_date_from_path(path):
    m = re.search(r"(20\d{6})", path)
    if m is None:
        raise ValueError(f"Cannot parse date from path: {path}")
    return m.group(1)


def month_to_season(month):
    if month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    elif month in [9, 10, 11]:
        return "Autumn"
    else:
        return "Winter"


def get_lcz_year_for_date(yyyymmdd):
    year = int(yyyymmdd[:4])
    month = int(yyyymmdd[4:6])

    # January and February use previous-year winter LCZ
    if month in [1, 2]:
        return year - 1
    return year


def get_lcz_path_for_date(yyyymmdd):
    year = get_lcz_year_for_date(yyyymmdd)
    month = int(yyyymmdd[4:6])
    season = month_to_season(month)

    path = os.path.join(
        lcz_dir,
        f"{season}_LCZ_BTH_{year}_100m.tif"
    )

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"LCZ raster not found:\n{path}\n"
            f"Please check lcz_dir and winter-year rule."
        )

    return path, season, year


def get_no2_paths(no2_root, date_list=None, preferred_keyword="Ultimate_RBF_v7"):
    all_paths = glob.glob(
        os.path.join(no2_root, "*", "*NO2*Prediction*100m*.tif")
    )

    if len(all_paths) == 0:
        raise FileNotFoundError(f"No NO2 prediction rasters found under:\n{no2_root}")

    if date_list is None:
        return sorted(all_paths)

    selected = []

    for d in date_list:
        candidates = [p for p in all_paths if d in os.path.basename(p) or d in p]

        if len(candidates) == 0:
            print(f"Warning: no NO2 raster found for {d}")
            continue

        preferred = [
            p for p in candidates
            if preferred_keyword in os.path.basename(p)
        ]

        if len(preferred) > 0:
            chosen = sorted(preferred)[-1]
        else:
            print(f"Warning: no {preferred_keyword} raster found for {d}.")
            print("Available candidates:")
            for p in sorted(candidates):
                print("  ", p)
            chosen = sorted(candidates)[-1]
            print("Fallback selected:", chosen)

        print(f"Selected NO2 raster for {d}: {chosen}")
        selected.append(chosen)

    return selected

# ============================================================
# 4. Helper functions: raster reading and alignment
# ============================================================

def align_lcz_to_no2(lcz_path, no2_path):
    with rasterio.open(no2_path) as no2_src:
        no2 = no2_src.read(1).astype("float32")
        no2_transform = no2_src.transform
        no2_crs = no2_src.crs
        no2_shape = no2_src.shape
        no2_nodata = no2_src.nodata

    with rasterio.open(lcz_path) as lcz_src:
        lcz_raw = lcz_src.read(1)
        lcz_nodata = lcz_src.nodata

        need_align = (
            lcz_src.crs != no2_crs or
            lcz_src.transform != no2_transform or
            lcz_src.shape != no2_shape
        )

        if need_align:
            print("  LCZ and NO2 grid differ. Aligning LCZ to NO2 grid...")
            lcz_aligned = np.full(no2_shape, 0, dtype="int16")

            reproject(
                source=lcz_raw,
                destination=lcz_aligned,
                src_transform=lcz_src.transform,
                src_crs=lcz_src.crs,
                dst_transform=no2_transform,
                dst_crs=no2_crs,
                resampling=Resampling.nearest,
                src_nodata=lcz_nodata,
                dst_nodata=0
            )
        else:
            print("  LCZ and NO2 grid already aligned.")
            lcz_aligned = lcz_raw.astype("int16")

    if no2_nodata is not None:
        no2[no2 == no2_nodata] = np.nan

    no2[no2 <= NO2_MIN_VALID] = np.nan
    no2[no2 > NO2_MAX_VALID] = np.nan

    lcz_aligned = lcz_aligned.astype("int16")
    lcz_aligned[(lcz_aligned < 1) | (lcz_aligned > 17)] = 0

    return no2, lcz_aligned


def summarize_array(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return {
            "N": 0,
            "Mean_NO2": np.nan,
            "Median_NO2": np.nan,
            "Q1_NO2": np.nan,
            "Q3_NO2": np.nan,
            "P05_NO2": np.nan,
            "P95_NO2": np.nan,
            "Std_NO2": np.nan
        }

    return {
        "N": int(len(x)),
        "Mean_NO2": float(np.mean(x)),
        "Median_NO2": float(np.median(x)),
        "Q1_NO2": float(np.percentile(x, 25)),
        "Q3_NO2": float(np.percentile(x, 75)),
        "P05_NO2": float(np.percentile(x, 5)),
        "P95_NO2": float(np.percentile(x, 95)),
        "Std_NO2": float(np.std(x))
    }

# ============================================================
# 5. Extract group samples and summary
# ============================================================

rng = np.random.default_rng(random_seed)

no2_paths = get_no2_paths(
    no2_root=no2_root,
    date_list=date_list,
    preferred_keyword=preferred_keyword
)

all_summary_rows = []
sample_rows = []

for no2_path in no2_paths:
    yyyymmdd = parse_date_from_path(no2_path)
    lcz_path, season, lcz_year = get_lcz_path_for_date(yyyymmdd)

    print("\n" + "=" * 100)
    print(f"Processing date: {yyyymmdd}")
    print(f"Season: {season}, LCZ year: {lcz_year}")
    print("=" * 100)

    no2, lcz = align_lcz_to_no2(lcz_path, no2_path)

    valid = np.isfinite(no2) & (lcz >= 1) & (lcz <= 17)
    total_valid_pixels = int(valid.sum())

    daily_mean = float(np.nanmean(no2[valid]))
    daily_median = float(np.nanmedian(no2[valid]))

    print(f"  Valid pixels: {total_valid_pixels}")
    print(f"  Daily mean NO2: {daily_mean:.3f}")
    print(f"  Daily median NO2: {daily_median:.3f}")
    print(f"  NO2 range: {np.nanmin(no2):.3f} - {np.nanmax(no2):.3f}")

    for group_name in GROUP_ORDER:
        codes = LCZ_GROUPS[group_name]

        mask = valid & np.isin(lcz, codes)
        values = no2[mask]

        stats = summarize_array(values)

        pixel_share = stats["N"] / total_valid_pixels * 100 if total_valid_pixels > 0 else np.nan

        stats.update({
            "Date": yyyymmdd,
            "Date_label": DATE_LABELS.get(yyyymmdd, yyyymmdd),
            "Season": season,
            "LCZ_year": lcz_year,
            "Group": group_name,
            "Group_short": GROUP_LABEL_SHORT[group_name],
            "LCZ_codes": ",".join([str(c) for c in codes]),
            "Total_valid_pixels": total_valid_pixels,
            "Pixel_share_percent": pixel_share,
            "Daily_mean_NO2": daily_mean,
            "Daily_median_NO2": daily_median,
            "Mean_deviation_from_daily_mean": stats["Mean_NO2"] - daily_mean,
            "Median_deviation_from_daily_median": stats["Median_NO2"] - daily_median
        })

        all_summary_rows.append(stats)

        finite_values = values[np.isfinite(values)]

        if len(finite_values) > 0:
            if len(finite_values) > max_sample_per_group:
                sampled = rng.choice(finite_values, size=max_sample_per_group, replace=False)
            else:
                sampled = finite_values

            tmp = pd.DataFrame({
                "Date": yyyymmdd,
                "Date_label": DATE_LABELS.get(yyyymmdd, yyyymmdd),
                "Season": season,
                "Group": group_name,
                "Group_short": GROUP_LABEL_SHORT[group_name],
                "NO2": sampled
            })

            sample_rows.append(tmp)

df_summary = pd.DataFrame(all_summary_rows)

if len(sample_rows) > 0:
    df_samples = pd.concat(sample_rows, ignore_index=True)
else:
    df_samples = pd.DataFrame(columns=["Date", "Date_label", "Season", "Group", "Group_short", "NO2"])

out_summary_csv = os.path.join(out_dir, "SCI_Figure_LCZ_Group_Summary.csv")
out_samples_csv = os.path.join(out_dir, "SCI_Figure_LCZ_Group_RawPixelSamples.csv")

df_summary.to_csv(out_summary_csv, index=False, encoding="utf-8-sig")
df_samples.to_csv(out_samples_csv, index=False, encoding="utf-8-sig")

print("\nSaved summary:")
print(out_summary_csv)
print("\nSaved sampled pixel data for boxplot:")
print(out_samples_csv)

print("\nSummary preview:")
print(df_summary[[
    "Date", "Group", "LCZ_codes", "N", "Pixel_share_percent",
    "Mean_NO2", "Mean_deviation_from_daily_mean",
    "Median_NO2", "Median_deviation_from_daily_median"
]])

# ============================================================
# 5.1 Diagnostics: merged LCZ 7 + D group, and separate LCZ 7/D
# ============================================================

MERGED_GROUP_NAME = "Greenhouse / low-plant\ntransitional\n(LCZ 7, D)"

print("\n" + "=" * 90)
print("Merged LCZ 7 + LCZ D group diagnostic")
print("=" * 90)
print(df_summary[df_summary["Group"] == MERGED_GROUP_NAME][[
    "Date", "N", "Pixel_share_percent", "Mean_NO2",
    "Daily_mean_NO2", "Mean_deviation_from_daily_mean",
    "Median_NO2", "Median_deviation_from_daily_median"
]])

# Additional diagnostic: separate LCZ 7 and LCZ D statistics, not used in figures
separate_diag_rows = []

for no2_path in no2_paths:
    yyyymmdd = parse_date_from_path(no2_path)
    lcz_path, season, lcz_year = get_lcz_path_for_date(yyyymmdd)

    no2, lcz = align_lcz_to_no2(lcz_path, no2_path)
    valid = np.isfinite(no2) & (lcz >= 1) & (lcz <= 17)

    daily_mean = float(np.nanmean(no2[valid]))
    total_valid_pixels = int(valid.sum())

    for code, label in [(7, "LCZ 7"), (14, "LCZ D")]:
        mask = valid & (lcz == code)
        values = no2[mask]
        stats = summarize_array(values)

        separate_diag_rows.append({
            "Date": yyyymmdd,
            "Class": label,
            "Code": code,
            "N": stats["N"],
            "Pixel_share_percent": stats["N"] / total_valid_pixels * 100,
            "Mean_NO2": stats["Mean_NO2"],
            "Median_NO2": stats["Median_NO2"],
            "Daily_mean_NO2": daily_mean,
            "Mean_deviation_from_daily_mean": stats["Mean_NO2"] - daily_mean
        })

df_separate_diag = pd.DataFrame(separate_diag_rows)
out_separate_diag_csv = os.path.join(out_dir, "Diagnostic_LCZ7_LCZD_Separate_Stats.csv")
df_separate_diag.to_csv(out_separate_diag_csv, index=False, encoding="utf-8-sig")

print("\n" + "=" * 90)
print("Separate LCZ 7 and LCZ D diagnostic")
print("=" * 90)
print(df_separate_diag)
print("\nSaved separate LCZ 7 / LCZ D diagnostic:")
print(out_separate_diag_csv)

# ============================================================
# 6. Optimized Figure 1: deviation bar plot
# ============================================================

def plot_group_deviation_bar(df_summary, out_dir):
    fig, ax = plt.subplots(figsize=(18.5, 8.2), dpi=300)

    width = 0.36
    x = np.arange(len(GROUP_ORDER))

    y_all = df_summary["Mean_deviation_from_daily_mean"].dropna().values
    ymin = min(0, np.min(y_all)) - 4
    ymax = max(0, np.max(y_all)) + 5

    for idx, d in enumerate(date_list):
        sub = df_summary[df_summary["Date"] == d].copy()
        sub = sub.set_index("Group").reindex(GROUP_ORDER).reset_index()

        y = sub["Mean_deviation_from_daily_mean"].values
        offset = -width / 2 if idx == 0 else width / 2

        bars = ax.bar(
            x + offset,
            y,
            width=width,
            color=DATE_COLORS.get(d, "#777777"),
            alpha=0.88,
            edgecolor="black",
            linewidth=1.1,
            label=DATE_LABELS.get(d, d)
        )

        for bar, val in zip(bars, y):
            if np.isfinite(val):
                if val >= 0:
                    y_text = val + 0.65
                    va = "bottom"
                else:
                    y_text = val - 0.65
                    va = "top"

                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    y_text,
                    f"{val:+.2f}",
                    ha="center",
                    va=va,
                    fontsize=13,
                    fontweight="bold",
                    color=DATE_COLORS.get(d, "black")
                )

    ax.axhline(0, color="black", linewidth=1.5)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [GROUP_LABEL_SHORT[g] for g in GROUP_ORDER],
        fontsize=13,
        fontweight="bold",
        rotation=18,
        ha="right",
        rotation_mode="anchor"
    )

    ax.set_ylim(ymin, ymax)

    ax.set_ylabel(
        "NO$_2$ deviation from daily mean ($\\mu$g/m$^3$)",
        fontsize=18,
        fontweight="bold",
        labelpad=12
    )

    ax.set_xlabel(
        "LCZ attribute-based group",
        fontsize=18,
        fontweight="bold",
        labelpad=16
    )

    ax.tick_params(axis="y", labelsize=15)
    ax.grid(axis="y", linestyle=":", alpha=0.45)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        fontsize=15,
        frameon=True,
        edgecolor="#333333",
        loc="upper right"
    )

    plt.subplots_adjust(bottom=0.25, left=0.08, right=0.98, top=0.96)

    out_png = os.path.join(out_dir, "SCI_Figure_LCZ_Group_Deviation_Bar_Merged7D.png")
    out_pdf = os.path.join(out_dir, "SCI_Figure_LCZ_Group_Deviation_Bar_Merged7D.pdf")

    plt.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.savefig(out_pdf, dpi=600, bbox_inches="tight")
    plt.show()

    print("Saved grouped deviation bar plot:")
    print(out_png)
    print(out_pdf)

# ============================================================
# 7. Optimized Figure 2: boxplot
# ============================================================

def plot_group_boxplot(df_samples, out_dir):
    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(22, 8.2),
        dpi=300,
        sharey=False
    )

    for ax, d in zip(axes, date_list):
        sub = df_samples[df_samples["Date"] == d].copy()
        date_label = DATE_LABELS.get(d, d)

        data = []
        colors = []

        for group in GROUP_ORDER:
            vals = sub[sub["Group"] == group]["NO2"].dropna().values
            data.append(vals)
            colors.append(GROUP_COLORS[group])

        bp = ax.boxplot(
            data,
            patch_artist=True,
            showfliers=False,
            widths=0.58,
            medianprops=dict(color="black", linewidth=1.6),
            boxprops=dict(linewidth=1.2),
            whiskerprops=dict(linewidth=1.1),
            capprops=dict(linewidth=1.1)
        )

        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.78)
            patch.set_edgecolor("black")

        means = [np.nanmean(v) if len(v) > 0 else np.nan for v in data]

        ax.plot(
            np.arange(1, len(GROUP_ORDER) + 1),
            means,
            color="black",
            linestyle="--",
            linewidth=2.1,
            marker="o",
            markersize=6.5,
            markerfacecolor="white",
            markeredgecolor="black",
            label="Group mean"
        )

        ax.set_xticks(np.arange(1, len(GROUP_ORDER) + 1))
        ax.set_xticklabels(
            [GROUP_LABEL_SHORT[g] for g in GROUP_ORDER],
            fontsize=11.5,
            fontweight="bold",
            rotation=22,
            ha="right",
            rotation_mode="anchor"
        )

        ax.set_title(
            date_label,
            fontsize=19,
            fontweight="bold",
            pad=12
        )

        ax.set_xlabel(
            "LCZ attribute-based group",
            fontsize=17,
            fontweight="bold",
            labelpad=16
        )

        ax.grid(axis="y", linestyle=":", alpha=0.45)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="y", labelsize=14)

        ax.legend(
            fontsize=13,
            frameon=True,
            edgecolor="#333333",
            loc="upper right"
        )

    axes[0].set_ylabel(
        "Predicted NO$_2$ ($\\mu$g/m$^3$)",
        fontsize=18,
        fontweight="bold",
        labelpad=12
    )

    plt.subplots_adjust(bottom=0.31, left=0.06, right=0.98, top=0.90, wspace=0.16)

    out_png = os.path.join(out_dir, "SCI_Figure_LCZ_Group_NO2_Boxplot_Merged7D.png")
    out_pdf = os.path.join(out_dir, "SCI_Figure_LCZ_Group_NO2_Boxplot_Merged7D.pdf")

    plt.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.savefig(out_pdf, dpi=600, bbox_inches="tight")
    plt.show()

    print("Saved grouped NO2 boxplot:")
    print(out_png)
    print(out_pdf)

# ============================================================
# 8. Optimized Figure 3: dumbbell plot
# ============================================================

def plot_group_dumbbell(df_summary, out_dir):
    fig, ax = plt.subplots(figsize=(14, 8.8), dpi=300)

    pivot = (
        df_summary
        .pivot(index="Group", columns="Date", values="Mean_deviation_from_daily_mean")
        .reindex(GROUP_ORDER)
    )

    y_pos = np.arange(len(GROUP_ORDER))

    all_values = []
    if "20240111" in pivot.columns:
        all_values.extend(pivot["20240111"].dropna().values.tolist())
    if "20240718" in pivot.columns:
        all_values.extend(pivot["20240718"].dropna().values.tolist())

    xmin = min(all_values) - 5
    xmax = max(all_values) + 5

    for i, group in enumerate(GROUP_ORDER):
        winter_val = pivot.loc[group, "20240111"] if "20240111" in pivot.columns else np.nan
        summer_val = pivot.loc[group, "20240718"] if "20240718" in pivot.columns else np.nan

        if np.isfinite(winter_val) and np.isfinite(summer_val):
            ax.plot(
                [summer_val, winter_val],
                [i, i],
                color="#A6A6A6",
                linewidth=2.4,
                alpha=0.95,
                zorder=1
            )

            ax.scatter(
                summer_val,
                i,
                s=120,
                color=DATE_COLORS["20240718"],
                edgecolor="black",
                linewidth=1.2,
                marker=DATE_MARKERS["20240718"],
                label=DATE_LABELS["20240718"] if i == 0 else None,
                zorder=3
            )

            ax.scatter(
                winter_val,
                i,
                s=130,
                color=DATE_COLORS["20240111"],
                edgecolor="black",
                linewidth=1.2,
                marker=DATE_MARKERS["20240111"],
                label=DATE_LABELS["20240111"] if i == 0 else None,
                zorder=3
            )

            if summer_val >= 0:
                summer_dx = -0.80
                summer_ha = "right"
            else:
                summer_dx = 0.80
                summer_ha = "left"

            if winter_val >= 0:
                winter_dx = 0.95
                winter_ha = "left"
            else:
                winter_dx = -0.95
                winter_ha = "right"

            summer_dy = -0.17
            winter_dy = 0.17

            ax.text(
                summer_val + summer_dx,
                i + summer_dy,
                f"{summer_val:+.2f}",
                ha=summer_ha,
                va="center",
                fontsize=13,
                fontweight="bold",
                color=DATE_COLORS["20240718"],
                zorder=5
            )

            ax.text(
                winter_val + winter_dx,
                i + winter_dy,
                f"{winter_val:+.2f}",
                ha=winter_ha,
                va="center",
                fontsize=13,
                fontweight="bold",
                color=DATE_COLORS["20240111"],
                zorder=5
            )

    ax.axvline(0, color="black", linewidth=1.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        [GROUP_LABEL_SHORT[g] for g in GROUP_ORDER],
        fontsize=14,
        fontweight="bold"
    )

    ax.invert_yaxis()
    ax.set_xlim(xmin, xmax)

    ax.set_xlabel(
        "NO$_2$ deviation from daily mean ($\\mu$g/m$^3$)",
        fontsize=18,
        fontweight="bold",
        labelpad=12
    )

    ax.set_ylabel(
        "LCZ attribute-based group",
        fontsize=18,
        fontweight="bold",
        labelpad=12
    )

    ax.tick_params(axis="x", labelsize=15)
    ax.grid(axis="x", linestyle=":", alpha=0.45)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        fontsize=15,
        frameon=True,
        edgecolor="#333333",
        loc="lower right"
    )

    plt.subplots_adjust(left=0.27, right=0.97, bottom=0.13, top=0.97)

    out_png = os.path.join(out_dir, "SCI_Figure_LCZ_Group_NO2_Dumbbell_Merged7D.png")
    out_pdf = os.path.join(out_dir, "SCI_Figure_LCZ_Group_NO2_Dumbbell_Merged7D.pdf")

    # plt.savefig(out_png, dpi=600, bbox_inches="tight")
    # plt.savefig(out_pdf, dpi=600, bbox_inches="tight")
    plt.show()

    print("Saved grouped NO2 dumbbell plot:")
    print(out_png)
    print(out_pdf)

# ============================================================
# 9. Run plotting
# ============================================================

# plot_group_deviation_bar(df_summary, out_dir)
# plot_group_boxplot(df_samples, out_dir)
plot_group_dumbbell(df_summary, out_dir)

# ============================================================
# 10. Finished
# ============================================================

print("\nOptimized LCZ attribute-based group analysis with merged LCZ 7 and D completed.")
print("Output directory:")
print(out_dir)