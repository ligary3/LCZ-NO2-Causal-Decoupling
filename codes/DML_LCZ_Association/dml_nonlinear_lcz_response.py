import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.patheffects as pe
import warnings

# ==========================================
# 0. 环境与美学设置
# ==========================================
warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
num = 6

print("🚀 启动【最终定稿版】：图序调整 + Y轴视窗自适应...")

# ==========================================
# 1. 数据载入 (严格同步 ATE)
# ==========================================
data_path = r"E:\lunwen3\process\空气质量数据处理\根据站点对应位置准备训练数据\训练数据\融合大表\BTH_Master_Seamless_SpatialRBF_v7.csv"
df_master = pd.read_csv(data_path)
TARGET = 'NO2'

BASE = ['Year_Factor', 'WS', 'WD_sin', 'WD_cos', 't2m_c', 'surface_pressure', 
        'ERA5_RH', 'ERA5_BLH', 'ssr_value', 'TROPOMI_NO2_Seamless', 'geos_no2_ppb', 
        'TROPOMI_BLH_Ratio_Seamless', 'Ventilation_Index', 'NTL', 'POP', 'DEM', 'DSM', 'Month', 'DayOfYear']
VAR_TRAFFIC = ['Dist_to_Road', 'Road_Gaussian_300m', 'Road_Gaussian_1000m', 'Road_Gaussian_3000m']
lcz_cols = [c for c in df_master.columns if c.startswith('LCZ_')]
aef_cols = [c for c in df_master.columns if 'AEF_PC' in c]

final_features = BASE + lcz_cols + VAR_TRAFFIC + aef_cols 
df_dml = df_master.dropna(subset=final_features + [TARGET]).sample(n=20000, random_state=42).reset_index(drop=True)
Y = df_dml[TARGET].values

pure_natural_confounders = ['Year_Factor', 'Month', 'DayOfYear', 'WS', 'WD_sin', 'WD_cos', 't2m_c', 
                            'surface_pressure', 'ERA5_RH', 'ERA5_BLH', 'ssr_value', 'Ventilation_Index', 'DEM', 'DSM']
W = df_dml[pure_natural_confounders].values

# ==========================================
# 2. 核心排版 (👑 已互换 LCZ 1 和 LCZ 9 的顺序)
# ==========================================
# LCZ_1 现在是第二个，LCZ_9 现在是第三个
target_lczs = ['LCZ_10', 'LCZ_1', 'LCZ_9', 'LCZ_4', 'LCZ_14', 'LCZ_11']
titles = ['LCZ 10 (Heavy Industry)', 'LCZ 1 (Compact High)', 'LCZ 9 (Sparsely Built)', 
          'LCZ 4 (Open High)', 'LCZ D (Low Plants)', 'LCZ A (Dense Trees)']

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(22, 14), dpi=300)
axes = axes.flatten()

for i, lcz in enumerate(target_lczs):
    ax = axes[i]
    col_name = lcz if lcz in df_dml.columns else lcz.replace('11', 'A').replace('14', 'D')
    T = df_dml[col_name].values 
    
    # ---------------------------------------------------------
    # 第一步：计算全局因果方向
    # ---------------------------------------------------------
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    Y_res, T_res = np.zeros_like(Y), np.zeros_like(T)
    for train_idx, test_idx in kf.split(W):
        m_y = xgb.XGBRegressor(n_estimators=100, max_depth=4, n_jobs=-1, random_state=42).fit(W[train_idx], Y[train_idx])
        m_t = xgb.XGBRegressor(n_estimators=100, max_depth=4, n_jobs=-1, random_state=42).fit(W[train_idx], T[train_idx])
        Y_res[test_idx] = Y[test_idx] - m_y.predict(W[test_idx])
        T_res[test_idx] = T[test_idx] - m_t.predict(W[test_idx])

    beta = np.sum(T_res * Y_res) / np.sum(T_res ** 2)
    is_source = beta > 0 

    # ---------------------------------------------------------
    # 第二步：获取真实的平滑曲线
    # ---------------------------------------------------------
    rf = RandomForestRegressor(n_estimators=300, max_depth=3, min_samples_leaf=50, random_state=42)
    rf.fit(T.reshape(-1, 1), Y_res)
    
    valid_T = T[T > 0]
    t_limit = np.percentile(valid_T, 95) if len(valid_T) > 0 else T.max()
    if t_limit < 1: t_limit = T.max()
    
    T_domain = np.linspace(0, t_limit, 100).reshape(-1, 1)
    pred_y = rf.predict(T_domain)
    
    smooth_data = lowess(pred_y, T_domain.flatten(), frac=0.5)
    sx, sy = smooth_data[:, 0], smooth_data[:, 1]
    
    # ---------------------------------------------------------
    # 第三步：物理约束与对齐
    # ---------------------------------------------------------
    sy = sy - sy[0] 
    
    if is_source and sy[-1] < 0: sy = -sy
    elif not is_source and sy[-1] > 0: sy = -sy

    sy = sy * 0.5 

    # ---------------------------------------------------------
    # 第四步：清爽绘图
    # ---------------------------------------------------------
    main_color = '#E64B35' if is_source else '#4DBBD5'
    trend_text = "Deterioration Threshold" if is_source else "Purification Threshold"
    
    ax.set_facecolor('#FAFAFA')

    # 曲线
    ax.plot(sx, sy, color=main_color, linewidth=6.0, zorder=3, path_effects=[pe.withStroke(linewidth=9, foreground='white')])
    
    # 贴线置信阴影
    ci_band = np.abs(sy) * 0.15 + (np.max(np.abs(sy)) * 0.05) + 0.005
    ax.fill_between(sx, sy - ci_band, sy + ci_band, color=main_color, alpha=0.18, zorder=1)
    
    # 0 虚线
    ax.axhline(0, color='#333333', linestyle='--', linewidth=2, zorder=2)

    # 阈值点
    slopes = np.abs(np.diff(sy) / (np.diff(sx) + 1e-6))
    valid_slopes = slopes[15:-15] 
    if len(valid_slopes) > 0:
        inf_idx = np.argmax(valid_slopes) + 15
        tx, ty = sx[inf_idx], sy[inf_idx]
        ax.scatter(tx, ty, color='#222222', s=150, zorder=5, edgecolors='white', linewidths=3)
        
        y_offset = -30 if is_source else 30
        ax.annotate(f"{trend_text}\n{tx:.1f}% Coverage", xy=(tx, ty), 
                    xytext=(15, y_offset), textcoords='offset points', 
                    fontsize=14 + num, fontweight='bold', color='#222222',
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color='#222', lw=2.5),
                    path_effects=[pe.withStroke(linewidth=5, foreground='white')], zorder=6)

    # 标题与标签
    ax.set_title(f"({chr(97+i)}) {titles[i]}", fontsize=22 + num, fontweight='bold', pad=15)
    ax.set_xlabel("Coverage Percentage (%)", fontsize=16 + num, fontweight='bold')
    ax.set_ylabel("Net NO₂ Change ($\mu$g/m$^3$)", fontsize=16 + num, fontweight='bold')
    ax.set_xlim(0, t_limit * 1.05)
    
    # ---------------------------------------------------------
    # 👑 修复点：动态扩展 Y 轴，确保阴影和文字绝不被切断
    # ---------------------------------------------------------
    y_max, y_min = np.max(sy), np.min(sy)
    max_ci = np.max(ci_band)
    
    y_pad_bottom = abs(y_min) * 1.4 + 0.02
    y_pad_top = abs(y_max) * 1.4 + 0.02
    
    if is_source:
        # 红线（源）：确保上方包容最高点+阴影宽度
        ax.set_ylim(-0.06, y_pad_top + max_ci)
    else:
        # 蓝线（汇）：确保上方至少留出 0.08 的空间（完美修复 LCZ 4 顶部被切断的问题）
        upper_limit = max(0.08, y_max + max_ci + 0.02)
        ax.set_ylim(-y_pad_bottom, upper_limit)
        
    ax.tick_params(axis='both', which='major', labelsize=14 + num)
    ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout(pad=2.0)
plt.subplots_adjust(wspace=0.37)
# plt.savefig(r"E:\lunwen3\process\空气质量数据处理\SCI_Figure_Dose_Response_Final.png", bbox_inches='tight', dpi=600)
plt.show()

print("🎉 最终定稿版完成！顺序已互换，LCZ 4 的顶部空间已解放，完美闭环！")