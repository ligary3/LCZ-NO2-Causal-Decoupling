import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from scipy.stats import norm
import warnings

# 忽略警告并设置字体
warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
num = 8
print("🚀 启动 DML 全景版：融合 10 大核心形态 + 三级显著性星号...")

# ==========================================
# 1. 路径设置与数据载入
# ==========================================
data_path = r"E:\lunwen3\process\空气质量数据处理\根据站点对应位置准备训练数据\训练数据\融合大表\BTH_Master_Seamless_SpatialRBF_v7.csv"
df_master = pd.read_csv(data_path)
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

# ==========================================
# 2. DML 数据过滤与准备 (锁定纯自然背景)
# ==========================================
df_dml = df_master.dropna(subset=final_features + [TARGET]).sample(n=20000, random_state=42).reset_index(drop=True)
Y = df_dml[TARGET].values

pure_natural_confounders = [
    'Year_Factor', 'Month', 'DayOfYear', 
    'WS', 'WD_sin', 'WD_cos', 't2m_c', 'surface_pressure', 
    'ERA5_RH', 'ERA5_BLH', 'ssr_value', 'Ventilation_Index', 
    'DEM', 'DSM'
]
W = df_dml[pure_natural_confounders].values

# 👑 核心融合：你原本挑选的经典 + 我们确立的重排源 (9, 10)
target_lczs = ['LCZ_1', 'LCZ_3', 'LCZ_4', 'LCZ_7', 'LCZ_8', 'LCZ_9', 'LCZ_10', 'LCZ_11', 'LCZ_14', 'LCZ_15']

rename_dict = {
    'LCZ_1': 'LCZ 1', 'LCZ_3': 'LCZ 3', 'LCZ_4': 'LCZ 4', 
    'LCZ_7': 'LCZ 7', 'LCZ_8': 'LCZ 8', 'LCZ_9': 'LCZ 9', 'LCZ_10': 'LCZ 10', 
    'LCZ_11': 'LCZ A', 'LCZ_14': 'LCZ D', 'LCZ_15': 'LCZ E'
}

# ==========================================
# 3. 👑 原生 DML 双重交叉拟合 (计算 Z-score)
# ==========================================
results = []
kf = KFold(n_splits=3, shuffle=True, random_state=42)

for lcz in target_lczs:
    # 兼容处理你的数据列名 (比如 LCZ_15 可能在表里叫 LCZ_E)
    col_name = lcz if lcz in df_dml.columns else lcz.replace('11', 'A').replace('14', 'D').replace('15', 'E')
    if col_name not in df_dml.columns: 
        print(f"⚠️ 跳过 {lcz}，因为在数据表中找不到该列名")
        continue
        
    T = df_dml[col_name].values
    Y_res = np.zeros_like(Y, dtype=float)
    T_res = np.zeros_like(T, dtype=float)
    
    for train_idx, test_idx in kf.split(W):
        W_train, W_test = W[train_idx], W[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        T_train, T_test = T[train_idx], T[test_idx]
        
        m_y = xgb.XGBRegressor(n_estimators=100, max_depth=4, n_jobs=-1, random_state=42).fit(W_train, Y_train)
        Y_res[test_idx] = Y_test - m_y.predict(W_test)
        
        m_t = xgb.XGBRegressor(n_estimators=100, max_depth=4, n_jobs=-1, random_state=42).fit(W_train, T_train)
        T_res[test_idx] = T_test - m_t.predict(W_test)
        
    beta = np.sum(T_res * Y_res) / np.sum(T_res ** 2)
    se = np.sqrt(np.sum((Y_res - beta * T_res) ** 2) / (len(T_res) - 2)) / np.sqrt(np.sum(T_res ** 2))
    
    # 计算 Z-score 进而得到星号级别
    z_score = abs(beta / se)
    if z_score > 2.576: sig = "***"
    elif z_score > 1.960: sig = "**"
    elif z_score > 1.645: sig = "*"
    else: sig = ""
    
    scale_factor = 0.1 
    results.append({
        'LCZ': rename_dict[lcz],
        'Effect': beta * scale_factor,
        'LB': (beta - 1.96 * se) * scale_factor,
        'UB': (beta + 1.96 * se) * scale_factor,
        'Star': sig
    })

# 👑 按效应大小进行完美升序排列，形成瀑布阶梯
df_res = pd.DataFrame(results).sort_values(by='Effect', ascending=True)

# ==========================================
# 4. 🎨 绘制 SCI 因果森林图 (三级显著性标注)
# ==========================================
plt.figure(figsize=(11, 9.5), dpi=300) # 高度稍微调大一点适应 10 个特征
ax = plt.gca()

y_pos = np.arange(len(df_res))
effects = df_res['Effect'].values
colors = ['#E64B35' if val > 0 else '#4DBBD5' for val in effects]

# 零线基准
plt.axvline(x=0, color='#333333', linestyle='--', linewidth=2.0, zorder=1)

for i in range(len(df_res)):
    # 绘制点和 95% 置信区间
    plt.errorbar(effects[i], y_pos[i], 
                 xerr=[[effects[i]-df_res.iloc[i]['LB']], [df_res.iloc[i]['UB']-effects[i]]], 
                 fmt='o', color=colors[i], ecolor=colors[i], elinewidth=3.5, capsize=8, capthick=3.5, 
                 markersize=14+ num, markerfacecolor='white', markeredgewidth=3.5, zorder=3)
    
    # 👑 标注数值与星号，带有强力白边防遮挡
    txt_str = f"{effects[i]:+.3f}{df_res.iloc[i]['Star']}"
    txt = plt.text(effects[i], y_pos[i] + 0.28, txt_str, 
                   fontsize=12+ num, fontweight='bold', color=colors[i], ha='center', va='center', zorder=4)
    txt.set_path_effects([pe.withStroke(linewidth=4, foreground='white')])

# 设置阴影区 (区分 Source 和 Sink 的背景光谱)
x_min, x_max = ax.get_xlim()
plt.axvspan(0, x_max * 1.1, facecolor='#FFEBEE', alpha=0.4, zorder=0) 
plt.axvspan(x_min * 1.1, 0, facecolor='#E3F2FD', alpha=0.4, zorder=0) 

# 坐标轴美化
plt.yticks(y_pos, df_res['LCZ'], fontsize=15+ num, )
plt.xticks(fontsize=14+ num)
plt.xlabel("Natural Direct Effect (NDE)\nNet Change in NO$_2$ (μg/m³) per 10% Area Increase", 
           fontsize=16+ num, labelpad=15)#fontweight='bold', 
#plt.title("Double Machine Learning (DML): Source-Sink Spectrum of LCZs", fontsize=20, fontweight='bold', pad=35)

# 外部顶部指示
plt.text(0.85, 1.02, 'Source Effect $\\rightarrow$', transform=ax.transAxes, 
         fontsize=14+ num, color='#C62828', fontweight='bold', ha='center', va='bottom')
plt.text(0.15, 1.02, '$\\leftarrow$ Sink Effect', transform=ax.transAxes, 
         fontsize=14+ num, color='#1565C0', fontweight='bold', ha='center', va='bottom')

# 完善图例：三级显著性说明
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Source Effect', markerfacecolor='white', markeredgecolor='#E64B35', markersize=14),
    Line2D([0], [0], marker='o', color='w', label='Sink Effect', markerfacecolor='white', markeredgecolor='#4DBBD5', markersize=14),
    Line2D([0], [0], color='#7F8C8D', lw=3.5, label='95% Confidence Interval (CI)'),
    Line2D([0], [0], color='none', label='*** p < 0.01; ** p < 0.05; * p < 0.1')
]
lgd = plt.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(0.98, 0.05), fontsize=12+ num, frameon=True)
plt.setp(lgd.get_texts())

plt.grid(axis='x', linestyle=':', color='gray', alpha=0.5, linewidth=1.0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim(-0.8, len(df_res) - 0.2)

# 导出高清图
out_fig = r"E:\lunwen3\process\空气质量数据处理\SCI_Figure_DML_Forest_Spectrum.png"
# plt.savefig(out_fig, bbox_inches='tight', dpi=600)

plt.tight_layout()
plt.show()

print("🎉 ATE 全景森林图生成完毕！快看看 10 种形态的阶梯分布！")