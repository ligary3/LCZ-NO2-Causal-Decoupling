# ============================================================
# Figure 10. Full-control DML-adjusted conditional LCZ-NO2 associations
#
# Interpretation:
#   Conditional statistical association, not causal effect.
#   Estimated per 10-percentage-point higher LCZ coverage.
#
# Controls:
#   Time + terrain + meteorology + road-source proxies + NTL + POP
#
# Not included by default:
#   - Other LCZ variables, because LCZs are compositional
#   - AEF PCs, because they may encode surface semantics overlapping with LCZs
#   - TROPOMI/GEOS NO2 background, because these are NO2-like outcome priors
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from sklearn.model_selection import KFold
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

# ============================================================
# 0. Plot settings
# ============================================================

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.linewidth"] = 1.2

num = 8

print("Starting Figure 10: full-control DML-adjusted conditional LCZ-NO2 association forest plot...")

# ============================================================
# 1. Paths
# ============================================================

data_path = r"E:\lunwen3\process\空气质量数据处理\根据站点对应位置准备训练数据\训练数据\融合大表\BTH_Master_Seamless_SpatialRBF_v7.csv"

out_dir = r"E:\lunwen3\process\空气质量数据处理\DML_Figure10_FullControl"
os.makedirs(out_dir, exist_ok=True)

out_fig_png = os.path.join(out_dir, "SCI_Figure10_DML_FullControl_LCZ_Association.png")
out_fig_pdf = os.path.join(out_dir, "SCI_Figure10_DML_FullControl_LCZ_Association.pdf")
out_csv = os.path.join(out_dir, "SCI_Figure10_DML_FullControl_LCZ_Association.csv")

# ============================================================
# 2. Load data
# ============================================================

df_master = pd.read_csv(data_path)

TARGET = "NO2"

lcz_cols = [c for c in df_master.columns if c.startswith("LCZ_")]
aef_cols = [c for c in df_master.columns if "AEF_PC" in c]

# ============================================================
# 3. Control variables
# ============================================================

# Baseline spatiotemporal and terrain controls
baseline_controls = [
    "Year_Factor",
    "Month",
    "DayOfYear",
    "DEM",
    "DSM"
]

# Meteorological / dispersion controls
meteorology_controls = [
    "WS",
    "WD_sin",
    "WD_cos",
    "t2m_c",
    "surface_pressure",
    "ERA5_RH",
    "ERA5_BLH",
    "ssr_value",
    "Ventilation_Index"
]

# Road-source proxy controls
road_controls = [
    "Dist_to_Road",
    "Road_Gaussian_300m",
    "Road_Gaussian_1000m",
    "Road_Gaussian_3000m"
]

# Socioeconomic / activity-background controls
socioeconomic_controls = [
    "NTL",
    "POP"
]

# Optional: pollution-background priors
# 建议默认 False。因为这些变量本身与 NO2 高度相关，可能会过度吸收 LCZ-NO2 空间关系。
INCLUDE_POLLUTION_BACKGROUND = False

pollution_background_controls = [
    "TROPOMI_NO2_Seamless",
    "geos_no2_ppb",
    "TROPOMI_BLH_Ratio_Seamless"
]

control_vars = (
    baseline_controls
    + meteorology_controls
    + road_controls
    + socioeconomic_controls
)

if INCLUDE_POLLUTION_BACKGROUND:
    control_vars = control_vars + pollution_background_controls

# 仅保留真实存在的控制变量
missing_controls = [c for c in control_vars if c not in df_master.columns]
if len(missing_controls) > 0:
    print("Warning: missing control variables will be ignored:")
    for c in missing_controls:
        print("  ", c)

control_vars = [c for c in control_vars if c in df_master.columns]

print("\nControl variables used in DML:")
for c in control_vars:
    print("  -", c)

# ============================================================
# 4. Target LCZ classes
# ============================================================

target_lczs = [
    "LCZ_1",
    "LCZ_3",
    "LCZ_4",
    "LCZ_7",
    "LCZ_8",
    "LCZ_9",
    "LCZ_10",
    "LCZ_11",
    "LCZ_14",
    "LCZ_15"
]

rename_dict = {
    "LCZ_1": "LCZ 1",
    "LCZ_3": "LCZ 3",
    "LCZ_4": "LCZ 4",
    "LCZ_7": "LCZ 7",
    "LCZ_8": "LCZ 8",
    "LCZ_9": "LCZ 9",
    "LCZ_10": "LCZ 10",
    "LCZ_11": "LCZ A",
    "LCZ_14": "LCZ D",
    "LCZ_15": "LCZ E"
}

# 兼容可能存在的字母列名
def resolve_lcz_col(lcz_name, df):
    if lcz_name in df.columns:
        return lcz_name

    alt = (
        lcz_name
        .replace("LCZ_11", "LCZ_A")
        .replace("LCZ_12", "LCZ_B")
        .replace("LCZ_13", "LCZ_C")
        .replace("LCZ_14", "LCZ_D")
        .replace("LCZ_15", "LCZ_E")
        .replace("LCZ_16", "LCZ_F")
        .replace("LCZ_17", "LCZ_G")
    )

    if alt in df.columns:
        return alt

    return None

resolved_targets = []
for lcz in target_lczs:
    col = resolve_lcz_col(lcz, df_master)
    if col is None:
        print(f"Warning: skip {lcz}, column not found.")
    else:
        resolved_targets.append((lcz, col))

print("\nTarget LCZ variables:")
for raw, col in resolved_targets:
    print(f"  {rename_dict[raw]}: {col}")

# ============================================================
# 5. Prepare DML sample
# ============================================================

needed_cols = [TARGET] + control_vars + [col for _, col in resolved_targets]
df_base = df_master.dropna(subset=needed_cols).copy()

# 样本量设置
# 如果机器允许，可以调到 40000 或 50000；如果慢就用 20000。
N_DML = 40000

if len(df_base) > N_DML:
    df_dml = df_base.sample(n=N_DML, random_state=42).reset_index(drop=True)
else:
    df_dml = df_base.reset_index(drop=True)

print(f"\nDML sample size: {len(df_dml)}")

Y = df_dml[TARGET].values.astype(float)
W = df_dml[control_vars].values.astype(float)

# ============================================================
# 6. DML estimation function
# ============================================================

def dml_plr_single_treatment(Y, T, W, n_splits=3, random_state=42):
    """
    Partially linear DML:
        Y = theta T + g(W) + U
        T = m(W) + V

    Returns beta, se, z, p, residuals.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    Y_res = np.zeros_like(Y, dtype=float)
    T_res = np.zeros_like(T, dtype=float)

    for train_idx, test_idx in kf.split(W):
        W_train, W_test = W[train_idx], W[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        T_train, T_test = T[train_idx], T[test_idx]

        y_model = xgb.XGBRegressor(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="reg:squarederror",
            n_jobs=-1,
            random_state=random_state
        )

        t_model = xgb.XGBRegressor(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="reg:squarederror",
            n_jobs=-1,
            random_state=random_state + 1
        )

        y_model.fit(W_train, Y_train)
        t_model.fit(W_train, T_train)

        Y_res[test_idx] = Y_test - y_model.predict(W_test)
        T_res[test_idx] = T_test - t_model.predict(W_test)

    denom = np.sum(T_res ** 2)

    if denom <= 1e-12:
        return np.nan, np.nan, np.nan, np.nan, Y_res, T_res

    beta = np.sum(T_res * Y_res) / denom

    residual = Y_res - beta * T_res
    n = len(Y)
    se = np.sqrt(np.sum(residual ** 2) / (n - 2)) / np.sqrt(denom)

    z = beta / se if se > 0 else np.nan

    # normal approximation p value
    # avoid scipy dependency
    p = 2 * (1 - 0.5 * (1 + np.math.erf(abs(z) / np.sqrt(2)))) if np.isfinite(z) else np.nan

    return beta, se, z, p, Y_res, T_res

# ============================================================
# 7. Estimate DML associations
# ============================================================

results = []

for raw_lcz, col_name in resolved_targets:
    print(f"\nEstimating DML association for {rename_dict[raw_lcz]} ({col_name})...")

    T = df_dml[col_name].values.astype(float)

    beta, se, z, p, Y_res, T_res = dml_plr_single_treatment(
        Y=Y,
        T=T,
        W=W,
        n_splits=3,
        random_state=42
    )

    scale_factor = 10.0  # per 10-percentage-point increase

    effect = beta * scale_factor
    lb = (beta - 1.96 * se) * scale_factor
    ub = (beta + 1.96 * se) * scale_factor

    if np.isfinite(p):
        if p < 0.01:
            sig = "***"
        elif p < 0.05:
            sig = "**"
        elif p < 0.1:
            sig = "*"
        else:
            sig = ""
    else:
        sig = ""

    results.append({
        "LCZ_raw": raw_lcz,
        "LCZ": rename_dict[raw_lcz],
        "Column": col_name,
        "Effect": effect,
        "LB": lb,
        "UB": ub,
        "SE": se * scale_factor,
        "Z": z,
        "P_value": p,
        "Star": sig,
        "Mean_T": float(np.mean(T)),
        "SD_T": float(np.std(T)),
        "N": len(df_dml)
    })

df_res = pd.DataFrame(results)
df_res = df_res.sort_values(by="Effect", ascending=True).reset_index(drop=True)

df_res.to_csv(out_csv, index=False, encoding="utf-8-sig")

print("\nFigure 10 full-control DML results:")
print(df_res[["LCZ", "Effect", "LB", "UB", "SE", "P_value", "Star", "Mean_T", "SD_T"]])
print("\nSaved CSV:")
print(out_csv)

# ============================================================
# 8. Plot Figure 10
# ============================================================

fig, ax = plt.subplots(figsize=(11.5, 9.5), dpi=300)

y_pos = np.arange(len(df_res))
effects = df_res["Effect"].values

colors = ["#E64B35" if val > 0 else "#4DBBD5" for val in effects]

x_min = min(df_res["LB"].min(), -0.5)
x_max = max(df_res["UB"].max(), 0.5)
pad = 0.12 * (x_max - x_min)

# Background regions
ax.axvspan(x_min - pad, 0, facecolor="#E3F2FD", alpha=0.35, zorder=0)
ax.axvspan(0, x_max + pad, facecolor="#FFEBEE", alpha=0.35, zorder=0)

# Zero line
ax.axvline(x=0, color="#333333", linestyle="--", linewidth=2.0, zorder=1)

for i in range(len(df_res)):
    eff = df_res.loc[i, "Effect"]
    lb = df_res.loc[i, "LB"]
    ub = df_res.loc[i, "UB"]
    star = df_res.loc[i, "Star"]
    c = colors[i]

    ax.errorbar(
        eff,
        y_pos[i],
        xerr=[[eff - lb], [ub - eff]],
        fmt="o",
        color=c,
        ecolor=c,
        elinewidth=3.2,
        capsize=7,
        capthick=3.2,
        markersize=16,
        markerfacecolor="white",
        markeredgewidth=3.2,
        zorder=3
    )

    label = f"{eff:+.3f}{star}"

    # 标注稍微偏移，避免和误差线重叠
    y_offset = 0.28
    txt = ax.text(
        eff,
        y_pos[i] + y_offset,
        label,
        fontsize=16,
        fontweight="bold",
        color=c,
        ha="center",
        va="center",
        zorder=4
    )
    txt.set_path_effects([pe.withStroke(linewidth=4, foreground="white")])

# Axis labels
ax.set_yticks(y_pos)
ax.set_yticklabels(df_res["LCZ"], fontsize=18)
ax.tick_params(axis="x", labelsize=17)

ax.set_xlabel(
    "DML-adjusted conditional association with NO₂\n"
    "(μg/m³ per 10-percentage-point LCZ increase)",
    fontsize=19,
    labelpad=16
)

# Top labels
ax.text(
    0.84,
    1.02,
    "Higher NO$_2$ association $\\rightarrow$",
    transform=ax.transAxes,
    fontsize=18,
    color="#C62828",
    fontweight="bold",
    ha="center",
    va="bottom"
)

ax.text(
    0.16,
    1.02,
    "$\\leftarrow$ Lower NO$_2$ association",
    transform=ax.transAxes,
    fontsize=18,
    color="#1565C0",
    fontweight="bold",
    ha="center",
    va="bottom"
)

# Legend
legend_elements = [
    Line2D(
        [0], [0],
        marker="o",
        color="w",
        label="Positive association",
        markerfacecolor="white",
        markeredgecolor="#E64B35",
        markersize=13
    ),
    Line2D(
        [0], [0],
        marker="o",
        color="w",
        label="Negative association",
        markerfacecolor="white",
        markeredgecolor="#4DBBD5",
        markersize=13
    ),
    Line2D(
        [0], [0],
        color="#7F8C8D",
        lw=3,
        label="95% confidence interval"
    ),
    Line2D(
        [0], [0],
        color="none",
        label="*** p < 0.01; ** p < 0.05; * p < 0.1"
    )
]

ax.legend(
    handles=legend_elements,
    loc="lower right",
    bbox_to_anchor=(0.98, 0.05),
    fontsize=15,
    frameon=True,
    edgecolor="#333333"
)

# Styling
ax.grid(axis="x", linestyle=":", color="gray", alpha=0.5, linewidth=1.0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.set_xlim(x_min - pad, x_max + pad)
ax.set_ylim(-0.8, len(df_res) - 0.2)

plt.tight_layout()

# plt.savefig(out_fig_png, bbox_inches="tight", dpi=600)
# plt.savefig(out_fig_pdf, bbox_inches="tight", dpi=600)

plt.show()

print("\nFigure 10 completed.")
print(out_fig_png)
print(out_fig_pdf)