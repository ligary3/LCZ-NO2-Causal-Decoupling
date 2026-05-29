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

print("🚀 启动顶刊级引擎：重构空间背景的 CMA 因果中介分析...")

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

df_dml = df_master.dropna(subset=final_features + [TARGET]).sample(n=20000, random_state=42).reset_index(drop=True)
Y = df_dml[TARGET].values

# ==========================================
# 2. 👑 核心修复：保持与森林图(NDE) 严格一致的控制逻辑
# ==========================================

# 1. 基础时空背景 (W_base)：计算【总效应 TE】。
# 必须只包含 LCZ 无法改变的先验变量，且必须与 NDE 模型的基准一致！
# 注意：不要放 NTL 和 POP，因为它们是 LCZ 的“同生变量”
base_confounders = ['Year_Factor', 'Month', 'DayOfYear', 'DEM', 'DSM'] 
W_base = df_dml[base_confounders].values

# 2. 全变量背景 (W_full)：计算【直接效应 ADE/NDE】。
# 逻辑：W_base + 气象中介变量
# 👑 这里的变量列表必须和你【直接效应森林图】代码里的 W 变量一模一样！
mediation_vars = [
    'WS', 'WD_sin', 'WD_cos', 't2m_c', 'surface_pressure', 
    'ERA5_RH', 'ERA5_BLH', 'ssr_value', 'Ventilation_Index'
]
full_confounders = base_confounders + mediation_vars
W_full = df_dml[full_confounders].values
target_lczs = ['LCZ_10', 'LCZ_1', 'LCZ_4', 'LCZ_14', 'LCZ_9', 'LCZ_11']
rename_dict = {
    'LCZ_10': 'LCZ 10', 'LCZ_9': 'LCZ 9', 'LCZ_1': 'LCZ 1', 
    'LCZ_4': 'LCZ 4', 'LCZ_14': 'LCZ D', 'LCZ_11': 'LCZ A'
}

# 此时：
# TE = DML(W_base) -> 包含气象路径的总影响
# ADE = DML(W_full) -> 冻结气象后的纯物理影响（将完美等于你的森林图数值！）
# ACME = TE - ADE -> 气象中介影响
# ==========================================
# 3. 跑 DML 
# ==========================================
results = []
kf = KFold(n_splits=3, shuffle=True, random_state=42)

def compute_dml_effect(W_matrix, T_array, Y_array):
    Y_res, T_res = np.zeros_like(Y_array, dtype=float), np.zeros_like(T_array, dtype=float)
    for train_idx, test_idx in kf.split(W_matrix):
        m_y = xgb.XGBRegressor(n_estimators=100, max_depth=4, n_jobs=-1, random_state=42).fit(W_matrix[train_idx], Y_array[train_idx])
        m_t = xgb.XGBRegressor(n_estimators=100, max_depth=4, n_jobs=-1, random_state=42).fit(W_matrix[train_idx], T_array[train_idx])
        Y_res[test_idx] = Y_array[test_idx] - m_y.predict(W_matrix[test_idx])
        T_res[test_idx] = T_array[test_idx] - m_t.predict(W_matrix[test_idx])
    beta = np.sum(T_res * Y_res) / np.sum(T_res ** 2)
    return beta * 0.1 

for lcz in target_lczs:
    col_name = lcz if lcz in df_dml.columns else lcz.replace('11', 'A').replace('14', 'D')
    if col_name not in df_dml.columns: continue

    T = df_dml[col_name].values
    te = compute_dml_effect(W_base, T, Y)
    ade = compute_dml_effect(W_full, T, Y) 
    acme = te - ade
    results.append({'LCZ': rename_dict[lcz], 'TE': te, 'ADE': ade, 'ACME': acme})

df_res = pd.DataFrame(results).sort_values(by='TE', ascending=True).reset_index(drop=True)

# ==========================================
# 4. 🎨 智能堆叠绘图 (标注 TE 数值 + 严谨标题版)
# ==========================================
fig, ax = plt.subplots(figsize=(14, 8.5), dpi=300)
y_pos = np.arange(len(df_res))
bar_height = 0.55 

for i in range(len(df_res)):
    row = df_res.iloc[i]
    te, ade, acme = row['TE'], row['ADE'], row['ACME']
    
    direct_col = '#C62828' if ade > 0 else '#1565C0'  
    indirect_col = '#FFCDD2' if acme > 0 else '#BBDEFB' 
    
    # 智能堆叠
    if ade * acme > 0:
        ax.barh(y_pos[i], ade, height=bar_height, color=direct_col, edgecolor='white', linewidth=1.5, zorder=3)
        ax.barh(y_pos[i], acme, height=bar_height, left=ade, color=indirect_col, edgecolor='white', linewidth=1.5, zorder=3)
    else:
        ax.barh(y_pos[i], ade, height=bar_height, color=direct_col, edgecolor='white', linewidth=1.5, zorder=3)
        ax.barh(y_pos[i], acme, height=bar_height, color=indirect_col, edgecolor='white', linewidth=1.5, zorder=3)
    
    # 总效应黑菱形
    ax.scatter(te, y_pos[i], color='#222222', marker='D', s=130, zorder=5, edgecolors='white', linewidths=2.5)
    
    # 👑 标注修改：直接标注 TE 的物理数值，并保留 Offset/Mediated 逻辑说明
    if abs(te) > 0.0005:
        # 标注文本：TE 数值 + 机制分类
        mechanism = "Mediated" if ade * acme > 0 else "Offset"
        txt_str = f"TE: {te:+.3f}"
        
        # 动态调整文字位置，防止重叠
        txt_x = te + (0.002 if te > 0 else -0.002)
        txt = ax.text(txt_x, y_pos[i], txt_str, 
                      va='center', ha='left' if te > 0 else 'right', 
                      fontsize=13+num, fontweight='bold', color='#222222', zorder=6)
        txt.set_path_effects([pe.withStroke(linewidth=4, foreground='white')])

# 优化坐标轴范围
x_min, x_max = ax.get_xlim()
ax.set_xlim(x_min * 1.4, x_max * 1.4)

ax.axvline(0, color='black', linestyle='-', linewidth=2, zorder=1)
ax.axvspan(0, ax.get_xlim()[1], facecolor='#FFEBEE', alpha=0.3, zorder=0)
ax.axvspan(ax.get_xlim()[0], 0, facecolor='#E3F2FD', alpha=0.3, zorder=0)

ax.set_yticks(y_pos)
ax.set_yticklabels(df_res['LCZ'], fontsize=16+num)

# 👑 标题修改：使用更正式的学术表达
ax.set_xlabel("Decomposition of Causal Effects on NO$_2$ (Total, Direct, and Indirect)", 
             fontsize=17+num, fontweight='bold', labelpad=15)

ax.tick_params(axis='x', labelsize=14+num)
ax.grid(axis='x', linestyle='--', alpha=0.5)

# 图例优化
legend_elements = [
    Line2D([0], [0], color='#C62828', lw=8, label='Direct Effect (Morphology)'),
    Line2D([0], [0], color='#FFCDD2', lw=8, label='Indirect Source (Meteorological Trapping)'),
    Line2D([0], [0], color='#1565C0', lw=8, label='Direct Effect (Sink)'),
    Line2D([0], [0], color='#BBDEFB', lw=8, label='Indirect Sink (Meteorological Ventilation)'),
    Line2D([0], [0], marker='D', color='w', label='Total Net Effect (TE)', markerfacecolor='#222222', markersize=14)
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=12+num, framealpha=0.95, edgecolor='#333333')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()