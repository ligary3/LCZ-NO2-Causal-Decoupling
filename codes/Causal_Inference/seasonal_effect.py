import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
import warnings
num = 10
warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

print("🚀 启动顶刊级引擎：冬夏双子星 DML 因果效应哑铃图...")

# ==========================================
# 1. 数据载入与环境准备 (完美对齐你的特征)
# ==========================================
data_path = r"E:\lunwen3\process\空气质量数据处理\根据站点对应位置准备训练数据\训练数据\融合大表\BTH_Master_Seamless_SpatialRBF_v7.csv"
try:
    df_master = pd.read_csv(data_path)
except:
    print("🚨 请确保 df_master 已加载或路径正确！")

TARGET = 'NO2'

lcz_cols = [c for c in df_master.columns if c.startswith('LCZ_')]
aef_cols = [c for c in df_master.columns if 'AEF_PC' in c]

BASE = [
    'Year_Factor', 'WS', 'WD_sin', 'WD_cos', 't2m_c', 
    'surface_pressure', 'ERA5_RH', 'ERA5_BLH', 'ssr_value', 
    'TROPOMI_NO2_Seamless', 'geos_no2_ppb', 'TROPOMI_BLH_Ratio_Seamless', 'Ventilation_Index', 
    'NTL', 'POP', 'DEM', 'DSM', 'Month', 'DayOfYear' 
]
VAR_TRAFFIC = ['Dist_to_Road', 'Road_Gaussian_300m', 'Road_Gaussian_1000m', 'Road_Gaussian_3000m']
final_features = BASE + lcz_cols + VAR_TRAFFIC + aef_cols 

# 提取核心数据 (扩大样本量以保证冬夏各自有足够数据)
df_clean = df_master.dropna(subset=final_features + [TARGET]).sample(n=40000, random_state=42).reset_index(drop=True)

# 👑 划分季节 (夏季: 6-8, 冬季: 12-2)
df_summer = df_clean[df_clean['Month'].isin([6, 7, 8])].reset_index(drop=True)
df_winter = df_clean[df_clean['Month'].isin([12, 1, 2])].reset_index(drop=True)

# 控制变量 (与 ATE 森林图绝对一致的纯自然背景)
W_cols = [
    'Year_Factor', 'DayOfYear', 'WS', 'WD_sin', 'WD_cos', 't2m_c', 
    'surface_pressure', 'ERA5_RH', 'ERA5_BLH', 'ssr_value', 'Ventilation_Index', 'DEM', 'DSM'
]

target_lczs = ['LCZ_10', 'LCZ_1', 'LCZ_3', 'LCZ_4', 'LCZ_9', 'LCZ_14', 'LCZ_11']
rename_dict = {
    'LCZ_10': 'LCZ 10', 'LCZ_1': 'LCZ 1', 'LCZ_3': 'LCZ 3', 'LCZ_4': 'LCZ 4', 
    'LCZ_9': 'LCZ 9', 'LCZ_14': 'LCZ D', 'LCZ_11': 'LCZ A'
}

# ... (前面的数据载入与季节划分部分保持不变) ...

# ==========================================
# 2. 核心计算：分别计算冬夏的 Natural Direct Effect (NDE)
# ==========================================
results = []
kf = KFold(n_splits=3, shuffle=True, random_state=42)

def compute_seasonal_nde(df_season, target_lcz):
    col_name = target_lcz if target_lcz in df_season.columns else target_lcz.replace('11', 'A').replace('14', 'D')
    if col_name not in df_season.columns: return 0
    
    # 👑 保持与图 8b(森林图) 绝对一致的 NDE 控制变量
    W = df_season[W_cols].values
    T = df_season[col_name].values
    Y = df_season[TARGET].values
    
    Y_res, T_res = np.zeros_like(Y, dtype=float), np.zeros_like(T, dtype=float)
    for train_idx, test_idx in kf.split(W):
        m_y = xgb.XGBRegressor(n_estimators=80, max_depth=4, n_jobs=-1, random_state=42).fit(W[train_idx], Y[train_idx])
        Y_res[test_idx] = Y[test_idx] - m_y.predict(W[test_idx])
        m_t = xgb.XGBRegressor(n_estimators=80, max_depth=4, n_jobs=-1, random_state=42).fit(W[train_idx], T[train_idx])
        T_res[test_idx] = T[test_idx] - m_t.predict(W[test_idx])
        
    beta = np.sum(T_res * Y_res) / np.sum(T_res ** 2)
    return beta * 0.1 # 保持 scale_factor 统一

for lcz in target_lczs:
    nde_summer = compute_seasonal_nde(df_summer, lcz)
    nde_winter = compute_seasonal_nde(df_winter, lcz)
    # 按季节偏移幅度排序
    diff = nde_winter - nde_summer
    results.append({'LCZ': rename_dict[lcz], 'Summer': nde_summer, 'Winter': nde_winter, 'Diff': diff})

df_res = pd.DataFrame(results).sort_values(by='Diff', ascending=True).reset_index(drop=True)

# ==========================================
# 3. 🎨 绘制学术级 NDE 哑铃图
# ==========================================
fig, ax = plt.subplots(figsize=(11, 8.5), dpi=300)
y_pos = np.arange(len(df_res))

color_summer = '#4DBBD5'  # 夏季清爽蓝
color_winter = '#E64B35'  # 冬季警示红

# 背景分区
ax.axvline(0, color='#333333', linestyle='--', linewidth=2, zorder=1)
ax.axvspan(0, 0.05, facecolor='#FFEBEE', alpha=0.2, zorder=0)
ax.axvspan(-0.03, 0, facecolor='#E3F2FD', alpha=0.2, zorder=0)

for i in range(len(df_res)):
    s_val = df_res.iloc[i]['Summer']
    w_val = df_res.iloc[i]['Winter']
    
    # 哑铃杆
    ax.plot([s_val, w_val], [y_pos[i], y_pos[i]], color='#D1D1D1', linewidth=4, zorder=2)
    
    # 冬夏散点
    ax.scatter(s_val, y_pos[i], color=color_summer, s=280, edgecolor='white', linewidth=2, zorder=4)
    ax.scatter(w_val, y_pos[i], color=color_winter, s=280, edgecolor='white', linewidth=2, zorder=4)
    
    # 数值标注 (NDE 结果)
    ax.text(s_val, y_pos[i] + 0.1, f"{s_val:+.3f}", color=color_summer, fontsize=12 +num, fontweight='bold', ha='center')
    ax.text(w_val, y_pos[i] - 0.1, f"{w_val:+.3f}", color=color_winter, fontsize=12+num, fontweight='bold', ha='center', va='top')

# 坐标轴与标题
ax.set_yticks(y_pos)
ax.set_yticklabels(df_res['LCZ'], fontsize=16+num)
# 👑 标题与标签更新为 NDE
ax.set_xlabel("Seasonal Natural Direct Effect (NDE) on NO$_2$", fontsize=16+num, fontweight='bold', labelpad=15)
#只设置横轴标签字体大小
ax.tick_params(axis='x', labelsize=16+num)



#ax.set_title("Seasonal Divergence of Morphological Impacts", fontsize=20, fontweight='bold', pad=30)

legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Summer NDE (Direct Impact)', markerfacecolor=color_summer, markersize=14),
    Line2D([0], [0], marker='o', color='w', label='Winter NDE (Direct Impact)', markerfacecolor=color_winter, markersize=14),
    Line2D([0], [0], color='#D1D1D1', lw=4, label='Seasonal Shift Amplitude')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=12+num, frameon=True, edgecolor='#333333')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()