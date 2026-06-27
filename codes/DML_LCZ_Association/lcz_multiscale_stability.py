# ============================================================
# Raster-based multi-scale robustness analysis for LCZ attribute groups
#
# Unified Multi-Panel Plotting Script (3 Panels Side-by-Side)
# Panel A: Winter Heatmap | Panel B: Summer Heatmap | Panel C: Attenuation Line
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
import matplotlib as mpl
import matplotlib.patheffects as pe
from mpl_toolkits.axes_grid1 import make_axes_locatable

warnings.filterwarnings("ignore")

# ============================================================
# 0. Global plotting style
# ============================================================

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.linewidth"] = 1.2
plt.rcParams["axes.unicode_minus"] = False

# 统一调整论文大字号
num = 14

# ============================================================
# 1. User paths and settings
# ============================================================

lcz_dir = r"E:\lunwen3\BTHLCZ100"
no2_root = r"E:\lunwen3\process\预测"

out_dir = r"E:\lunwen3\process\空气质量数据处理\多尺度稳健性分析_属性组_冬夏分开"
os.makedirs(out_dir, exist_ok=True)

date_list = [
    "20240111",
    "20240718"
]

preferred_keyword = "Ultimate_RBF_v7"
scales = [100, 200, 500, 1000]

NO2_MIN_VALID = 0
NO2_MAX_VALID = 300

# ============================================================
# 2. LCZ attribute-group definition (优化了过长标签，防止重叠)
# ============================================================

LCZ_GROUPS = {
    "Compact built\n(1-3)": [1, 2, 3],
    "Open built\n(4-6)": [4, 5, 6],
    "Large low-rise / industrial\n(8,10)": [8, 10], 
    "Sparsely built\n(9)": [9],
    "Greenhouse / low-plant\n(7,D)": [7, 14],
    "Woody vegetation\n(A-C)": [11, 12, 13],
    "Paved / bare\n(E-F)": [15, 16],
    "Water\n(G)": [17],
}

GROUP_ORDER = [
    "Compact built\n(1-3)",
    "Open built\n(4-6)",
    "Large low-rise / industrial\n(8,10)",
    "Sparsely built\n(9)",
    "Greenhouse / low-plant\n(7,D)",
    "Woody vegetation\n(A-C)",
    "Paved / bare\n(E-F)",
    "Water\n(G)",
]

GROUP_CODE_LABEL = {
    "Compact built\n(1-3)": "1,2,3",
    "Open built\n(4-6)": "4,5,6",
    "Large low-rise / industrial\n(8,10)": "8,10",
    "Sparsely built\n(9)": "9",
    "Greenhouse / low-plant\n(7,D)": "7,14",
    "Woody vegetation\n(A-C)": "11,12,13",
    "Paved / bare\n(E-F)": "15,16",
    "Water\n(G)": "17",
}

DATE_LABELS = {
    "20240111": "Winter stagnant day",
    "20240718": "Summer convective day"
}

# ============================================================
# 3. Helper functions: file matching & data alignment
# ============================================================

def parse_date_from_path(path):
    m = re.search(r"(20\d{6})", path)
    if m is None: raise ValueError(f"Cannot parse date from path: {path}")
    return m.group(1)

def month_to_season(month):
    if month in [3, 4, 5]: return "Spring"
    elif month in [6, 7, 8]: return "Summer"
    elif month in [9, 10, 11]: return "Autumn"
    else: return "Winter"

def get_lcz_year_for_date(yyyymmdd):
    year = int(yyyymmdd[:4])
    month = int(yyyymmdd[4:6])
    if month in [1, 2]: return year - 1
    return year

def get_lcz_path_for_date(yyyymmdd):
    year = get_lcz_year_for_date(yyyymmdd)
    month = int(yyyymmdd[4:6])
    season = month_to_season(month)
    path = os.path.join(lcz_dir, f"{season}_LCZ_BTH_{year}_100m.tif")
    if not os.path.exists(path): raise FileNotFoundError(f"LCZ raster not found:\n{path}")
    return path, season, year

def get_no2_paths(no2_root, date_list=None, preferred_keyword="Ultimate_RBF_v7"):
    all_paths = glob.glob(os.path.join(no2_root, "*", "*NO2*Prediction*100m*.tif"))
    if len(all_paths) == 0: raise FileNotFoundError(f"No NO2 prediction rasters found under:\n{no2_root}")
    if date_list is None: return sorted(all_paths)
    selected = []
    for d in date_list:
        candidates = [p for p in all_paths if d in os.path.basename(p) or d in p]
        if len(candidates) == 0: continue
        preferred = [p for p in candidates if preferred_keyword in os.path.basename(p)]
        chosen = sorted(preferred)[-1] if len(preferred) > 0 else sorted(candidates)[-1]
        selected.append(chosen)
    return selected

def align_lcz_to_no2(lcz_path, no2_path):
    with rasterio.open(no2_path) as no2_src:
        no2 = no2_src.read(1).astype("float32")
        no2_meta = no2_src.meta.copy()
        no2_transform = no2_src.transform
        no2_crs = no2_src.crs
        no2_shape = no2_src.shape
        no2_nodata = no2_src.nodata

    with rasterio.open(lcz_path) as lcz_src:
        lcz_raw = lcz_src.read(1)
        lcz_nodata = lcz_src.nodata
        need_align = (lcz_src.crs != no2_crs or lcz_src.transform != no2_transform or lcz_src.shape != no2_shape)

        if need_align:
            lcz_aligned = np.full(no2_shape, 0, dtype="int16")
            reproject(
                source=lcz_raw, destination=lcz_aligned,
                src_transform=lcz_src.transform, src_crs=lcz_src.crs,
                dst_transform=no2_transform, dst_crs=no2_crs,
                resampling=Resampling.nearest, src_nodata=lcz_nodata, dst_nodata=0
            )
        else:
            lcz_aligned = lcz_raw.astype("int16")

    if no2_nodata is not None: no2[no2 == no2_nodata] = np.nan
    no2[(no2 <= NO2_MIN_VALID) | (no2 > NO2_MAX_VALID)] = np.nan
    lcz_aligned[(lcz_aligned < 1) | (lcz_aligned > 17)] = 0
    return no2, lcz_aligned, no2_meta

# ============================================================
# 4. Helper functions: block aggregation
# ============================================================

def trim_to_factor(arr, factor):
    h, w = arr.shape
    return arr[:(h // factor) * factor, :(w // factor) * factor]

def block_mean(arr, factor):
    arr = trim_to_factor(arr, factor)
    h, w = arr.shape
    return np.nanmean(arr.reshape(h // factor, factor, w // factor, factor), axis=(1, 3))

def block_group_fraction(lcz, factor, lcz_codes):
    lcz = trim_to_factor(lcz, factor)
    h, w = lcz.shape
    mask = np.isin(lcz, lcz_codes).astype("float32")
    return np.mean(mask.reshape(h // factor, factor, w // factor, factor), axis=(1, 3)) * 100.0

def compute_group_weighted_stats(no2_block, group_frac_block):
    no2 = no2_block.ravel()
    w = group_frac_block.ravel()
    valid = np.isfinite(no2) & np.isfinite(w) & (w > 0)
    if valid.sum() == 0:
        return {"Weighted_mean_NO2": np.nan, "N_cells_nonzero": 0, "Total_weight": 0.0, "Mean_fraction": np.nan}
    return {
        "Weighted_mean_NO2": float(np.average(no2[valid], weights=w[valid])),
        "N_cells_nonzero": int(valid.sum()),
        "Total_weight": float(np.sum(w[valid])),
        "Mean_fraction": float(np.mean(w[valid]))
    }

# ============================================================
# 5. Core multi-scale processing & Data Compilation
# ============================================================

no2_paths = get_no2_paths(no2_root=no2_root, date_list=date_list, preferred_keyword=preferred_keyword)
all_results = []

for no2_path in no2_paths:
    yyyymmdd = parse_date_from_path(no2_path)
    lcz_path, season, lcz_year = get_lcz_path_for_date(yyyymmdd)
    no2_100m, lcz_100m, _ = align_lcz_to_no2(lcz_path, no2_path)

    for scale in scales:
        factor = int(scale / 100)
        no2_block = block_mean(no2_100m, factor)
        overall_mean = float(np.nanmean(no2_block[np.isfinite(no2_block)]))

        for group_name in GROUP_ORDER:
            lcz_codes = LCZ_GROUPS[group_name]
            group_frac = block_group_fraction(lcz_100m, factor, lcz_codes)
            stats = compute_group_weighted_stats(no2_block, group_frac)
            deviation = stats["Weighted_mean_NO2"] - overall_mean

            row = {
                "Date": yyyymmdd, "Date_label": DATE_LABELS.get(yyyymmdd, yyyymmdd),
                "Season": season, "Scale_m": scale, "Group": group_name,
                "LCZ_codes": GROUP_CODE_LABEL[group_name], "NO2_deviation_from_daily_mean": deviation
            }
            row.update(stats)
            all_results.append(row)

df_results = pd.DataFrame(all_results)
df_summary = df_results.groupby(["Date", "Date_label", "Season", "Scale_m", "Group", "LCZ_codes"], as_index=False).agg(
    Mean_deviation=("NO2_deviation_from_daily_mean", "mean")
)
df_att = df_summary.dropna(subset=["Mean_deviation"]).groupby(["Date", "Date_label", "Scale_m"], as_index=False).agg(
    Mean_abs_deviation=("Mean_deviation", lambda x: float(np.mean(np.abs(x))))
)

# ============================================================
# 6. Unified Multi-Panel Visualization (横向合并、严丝合缝)
# ============================================================

# 修改：增加高度至 9.5，为 Y 轴标签留出充足空间
fig = plt.figure(figsize=(25.0, 9.5), dpi=300)

# ------------------------------------------------------------
# 子图 (a): 建立左侧的 Winter 热力图
# ------------------------------------------------------------
# 显式精确分配绘图区位置 [left, bottom, width, height]
ax_win = fig.add_axes([0.08, 0.15, 0.22, 0.72])

sub_win = df_summary[df_summary["Date"] == "20240111"].copy()
heat_win = sub_win.pivot(index="Group", columns="Scale_m", values="Mean_deviation").reindex(GROUP_ORDER).reindex(columns=scales)
data_win = heat_win.values.astype(float)
vmax_win = max(np.nanmax(np.abs(data_win)), 1)

norm_win = mpl.colors.TwoSlopeNorm(vmin=-vmax_win, vcenter=0, vmax=vmax_win)
im_win = ax_win.imshow(data_win, cmap="RdBu_r", norm=norm_win, aspect="auto")

ax_win.set_xticks(np.arange(len(scales)))
ax_win.set_xticklabels([f"{s} m" for s in scales], fontsize=11 + num)
ax_win.set_xlabel("Spatial aggregation scale", fontsize=12 + num, fontweight="bold", labelpad=10)
ax_win.set_yticks(np.arange(len(GROUP_ORDER)))
ax_win.set_yticklabels(GROUP_ORDER, fontsize=9 + num) # 常规不加粗
ax_win.set_ylabel("LCZ attribute-based group", fontsize=12 + num, fontweight="bold", labelpad=10)

ax_win.set_title("(a)", fontsize=14 + num, fontweight="bold", pad=15, loc="left")

# 标注数字与网格线
for i in range(data_win.shape[0]):
    for j in range(data_win.shape[1]):
        val = data_win[i, j]
        if np.isfinite(val):
            color = "white" if abs(val) > 0.55 * vmax_win else "black"
            t = ax_win.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=8 + num, fontweight="bold", color=color)
            t.set_path_effects([pe.withStroke(linewidth=2.5, foreground="white" if color != "white" else "black", alpha=0.45)])

ax_win.set_xticks(np.arange(-0.5, len(scales), 1), minor=True)
ax_win.set_yticks(np.arange(-0.5, len(GROUP_ORDER), 1), minor=True)
ax_win.grid(which="minor", color="white", linestyle="-", linewidth=1.5)
ax_win.tick_params(which="minor", bottom=False, left=False)

# 加挂 Colorbar
div_win = make_axes_locatable(ax_win)
cax_win = div_win.append_axes("right", size="5%", pad=0.12)
cbar_win = plt.colorbar(im_win, cax=cax_win)
cbar_win.ax.tick_params(labelsize=9 + num)
cbar_win.set_label("Deviation ($\\mu$g/m$^3$)", fontsize=9 + num, fontweight="bold", labelpad=8)


# ------------------------------------------------------------
# 子图 (b): 建立中间的 Summer 热力图 (完全隐去 Y 轴，确保与子图 a 的尺寸镜像相等)
# ------------------------------------------------------------
# 宽度 0.22, 高度 0.72 必须与子图 a 绝对一致，完全不受 Y 轴文字消失的影响
ax_sum = fig.add_axes([0.36, 0.15, 0.22, 0.72])

sub_sum = df_summary[df_summary["Date"] == "20240718"].copy()
heat_sum = sub_sum.pivot(index="Group", columns="Scale_m", values="Mean_deviation").reindex(GROUP_ORDER).reindex(columns=scales)
data_sum = heat_sum.values.astype(float)
vmax_sum = max(np.nanmax(np.abs(data_sum)), 1)

norm_sum = mpl.colors.TwoSlopeNorm(vmin=-vmax_sum, vcenter=0, vmax=vmax_sum)
im_sum = ax_sum.imshow(data_sum, cmap="RdBu_r", norm=norm_sum, aspect="auto")

ax_sum.set_xticks(np.arange(len(scales)))
ax_sum.set_xticklabels([f"{s} m" for s in scales], fontsize=11 + num)
ax_sum.set_xlabel("Spatial aggregation scale", fontsize=12 + num, fontweight="bold", labelpad=10)

ax_sum.set_yticks(np.arange(len(GROUP_ORDER)))
ax_sum.set_yticklabels([]) # 彻底隐去名字

ax_sum.set_title("(b)", fontsize=14 + num, fontweight="bold", pad=15, loc="left")

# 标注数字与网格线
for i in range(data_sum.shape[0]):
    for j in range(data_sum.shape[1]):
        val = data_sum[i, j]
        if np.isfinite(val):
            color = "white" if abs(val) > 0.55 * vmax_sum else "black"
            t = ax_sum.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=8 + num, fontweight="bold", color=color)
            t.set_path_effects([pe.withStroke(linewidth=2.5, foreground="white" if color != "white" else "black", alpha=0.45)])

ax_sum.set_xticks(np.arange(-0.5, len(scales), 1), minor=True)
ax_sum.set_yticks(np.arange(-0.5, len(GROUP_ORDER), 1), minor=True)
ax_sum.grid(which="minor", color="white", linestyle="-", linewidth=1.5)
ax_sum.tick_params(which="minor", bottom=False, left=False)

# 加挂 Colorbar
div_sum = make_axes_locatable(ax_sum)
cax_sum = div_sum.append_axes("right", size="5%", pad=0.12)
cbar_sum = plt.colorbar(im_sum, cax=cax_sum)
cbar_sum.ax.tick_params(labelsize=9 + num)
cbar_sum.set_label("Deviation ($\\mu$g/m$^3$)", fontsize=9 + num, fontweight="bold", labelpad=8)


# ------------------------------------------------------------
# 子图 (c): 建立右侧的 衰减曲线图
# ------------------------------------------------------------
ax_line = fig.add_axes([0.68, 0.15, 0.25, 0.72])

date_style = {
    "20240111": {"label": "Winter stagnant day", "color": "#4C72B0", "marker": "o"},
    "20240718": {"label": "Summer convective day", "color": "#C44E52", "marker": "s"}
}

for idx, d in enumerate(list(df_att["Date"].unique())):
    sub = df_att[df_att["Date"] == d].sort_values("Scale_m")
    style = date_style.get(d, {"label": d, "color": "#55A868", "marker": "o"})

    ax_line.plot(sub["Scale_m"], sub["Mean_abs_deviation"], marker=style["marker"], linewidth=2.8, markersize=9,
                 color=style["color"], markerfacecolor="white", markeredgewidth=2, label=style["label"])

    y_offset = 0.04 * max(df_att["Mean_abs_deviation"].max(), 1)
    for _, row in sub.iterrows():
        ax_line.text(row["Scale_m"], row["Mean_abs_deviation"] + y_offset, f"{row['Mean_abs_deviation']:.2f}",
                     ha="center", va="bottom", fontsize=7 + num, fontweight="bold", color=style["color"])

ax_line.set_xlabel("Spatial aggregation scale (m)", fontsize=11 + num, fontweight="bold", labelpad=10)

ax_line.set_ylabel("Mean absolute group-level NO$_2$ contrast ($\\mu$g/m$^3$)", fontsize=11 + num, fontweight="bold", labelpad=12, y=0.48)

ax_line.tick_params(axis="x", labelsize=10 + num)
ax_line.tick_params(axis="y", labelsize=10 + num)

ax_line.grid(True, linestyle=":", alpha=0.45)
ax_line.spines["top"].set_visible(False)
ax_line.spines["right"].set_visible(False)
ax_line.legend(fontsize=9 + num, frameon=True, edgecolor="#333333", loc="center right")
ax_line.set_ylim(0, df_att["Mean_abs_deviation"].max() * 1.15)

ax_line.set_title("(c)", fontsize=14 + num, fontweight="bold", pad=15, loc="left")

# ============================================================
# 8. Output and Save Complete Paper-Ready Image
# ============================================================
out_final_png = os.path.join(out_dir, "SCI_Figure13_Combined_Multiscale_Robustness.png")
out_final_pdf = os.path.join(out_dir, "SCI_Figure13_Combined_Multiscale_Robustness.pdf")

plt.savefig(out_final_png, dpi=600, bbox_inches="tight")
plt.savefig(out_final_pdf, dpi=600, bbox_inches="tight")
plt.show()

print("\n[SUCCESS] Unified single-row multi-panel figure completed.")