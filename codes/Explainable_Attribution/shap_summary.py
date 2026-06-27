import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import shap
import matplotlib.pyplot as plt
import warnings
import matplotlib.patheffects as pe
warnings.filterwarnings('ignore')
print("🚀 启动 XGBoost 黑盒解析：顶刊级 SHAP 二合一复合引擎 (Colorbar 字体升级版)...")
num = 6
# ==========================================
# 1. 载入模型与特征清单
# ==========================================
model_path = r"E:\lunwen3\process\空气质量数据处理\BTH_NO2_Final_Downscale_RBF_v7.json"
feature_list_path = r"E:\lunwen3\process\空气质量数据处理\BTH_NO2_Final_FeatureList_RBF_v7.joblib"

model = xgb.XGBRegressor()
model.load_model(model_path)
features = joblib.load(feature_list_path)

data_path = r"E:\lunwen3\process\空气质量数据处理\根据站点对应位置准备训练数据\训练数据\融合大表\BTH_Master_Seamless_SpatialRBF_v7.csv"
df_master = pd.read_csv(data_path)
df_clean = df_master.dropna(subset=features).copy()

# ==========================================
# 2. 👑 极其严苛的物理空间分层抽样 (80% 建成区聚焦)
# ==========================================
# 按照你的专业定义：建成区 = LCZ 1-8, 10, 以及 E(15, 硬化地表)
urban_indices = [1, 2, 3, 4, 5, 6, 7, 8, 10, 15]
urban_cols = [f'LCZ_{i}' for i in urban_indices if f'LCZ_{i}' in features]

is_urban = df_clean[urban_cols].sum(axis=1) > 0
df_urban = df_clean[is_urban]
df_rural = df_clean[~is_urban]

# 强制 80% 建成区 / 20% 自然区，最大程度逼出局地特征
n_urban = 8000
n_rural = 7000
df_sample = pd.concat([
    df_urban.sample(n=min(n_urban, len(df_urban)), random_state=42),
    df_rural.sample(n=min(n_rural, len(df_rural)), random_state=42)
]).reset_index(drop=True)

X_sample = df_sample[features]

# ==========================================
# 3. 👑 顶级 SCI 规范命名映射
# ==========================================
rename_dict = {
    'TROPOMI_BLH_Ratio_Seamless': 'TROPOMI/BLH Ratio',
    'geos_no2_ppb': 'GEOS-CF NO₂', # 完美下标
    'DayOfYear': 'Day of Year',
    'Year_Factor': 'Year',
    't2m_c': 'Temperature (T2m)',
    'TROPOMI_NO2_Seamless': 'TROPOMI NO₂ Column',
    'ERA5_RH': 'Relative Humidity (RH)',
    'ssr_value': 'Net Solar Radiation (SSR)',
    'ERA5_BLH': 'Boundary Layer Height (BLH)',
    'WD_cos': 'Wind Direction (cos)',
    'WD_sin': 'Wind Direction (sin)',
    'DEM': 'Elevation (DEM)',
    'DSM': 'Surface Model (DSM)',
    'POP': 'Population Density (PD)',
    'NTL': 'Nighttime Light (NTL)',
    'Month': 'Month',
    'Ventilation_Index': 'Ventilation Index (VI)',
    'WS': 'Wind Speed (WS)',
    'Dist_to_Road': 'Distance to Road (DTR)',
    'Road_Gaussian_300m': 'Road Density (300m)',
    'Road_Gaussian_1000m': 'Road Density (1000m)',
    'Road_Gaussian_3000m': 'Road Density (3000m)',
    'surface_pressure': 'Surface Pressure (SP)'
}
lcz_map = {11:'A', 12:'B', 13:'C', 14:'D', 15:'E', 16:'F', 17:'G'}
for i in range(1, 18):
    if i <= 10:
        rename_dict[f'LCZ_{i}'] = f'LCZ {i}'
    else:
        rename_dict[f'LCZ_{i}'] = f'LCZ {lcz_map[i]}'

for i in range(1, 20):
    rename_dict[f'AEF_PC{i}'] = f'AEF PC{i}'

# ==========================================
# 4. 计算 SHAP 值
# ==========================================
print("🧠 正在解析深层机制...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

# 转换名称
X_sample_plot = X_sample.rename(columns=rename_dict)

# ==========================================
# 5. 计算重要性
# ==========================================
max_display = 20
mean_abs_shap = np.abs(shap_values).mean(axis=0)
total_mean_abs_shap = mean_abs_shap.sum()

sorted_idx = np.argsort(mean_abs_shap)
top_idx = sorted_idx[-max_display:]

bar_values = mean_abs_shap[top_idx]
bar_pcts = (bar_values / total_mean_abs_shap) * 100

# ==========================================
# 6. 🎨 绘制带光晕特效的无敌神图
# ==========================================
print("🎨 正在渲染自带【文字白边光晕特效】及【精细化字号控制】的双坐标系 Combo Plot...")

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

fig, ax1 = plt.subplots(figsize=(15, 10), dpi=300)

# 【第一层：蜂拥图】
shap.summary_plot(shap_values, X_sample_plot, max_display=max_display, 
                  cmap='coolwarm', show=False, alpha=0.75, plot_size=(15, 10))

ax1.set_xlabel('SHAP value (Impact on predicted NO₂, μg/m³)', fontsize=16+num, fontweight='bold', labelpad=12)
ax1.tick_params(axis='x', labelsize=14+num)
for tick in ax1.get_yticklabels():
    tick.set_fontweight('bold')
    tick.set_fontsize(14+num)

# ==========================================
# 👑 核心修复：捕获 Colorbar 并修改字体大小
# ==========================================
# shap.summary_plot 会自动在图表右侧生成一个颜色条坐标系
# 它是 fig 中的最后一个 axes，我们把它抓出来！
cb_ax = fig.axes[-1] 

# 修改 Colorbar 的刻度标签 (Low, High) 字号
cb_ax.tick_params(labelsize=14+num)

# 修改 Colorbar 的标题 (Feature value) 字号
cb_ax.set_ylabel('Feature value', fontsize=16+num)
# ==========================================

# 【第二层：柱状图】
ax2 = ax1.twiny()
y_pos = np.arange(len(top_idx))

bars = ax2.barh(y_pos, bar_values, color='#8CA6CE', alpha=0.3, edgecolor='none', height=0.6)

ax2.set_xlim(0, np.max(bar_values) * 1.35) 

# 👑 终极白边特效：使用 PathEffects 给文字加上一层 3 像素的白边
for i, (val, pct) in enumerate(zip(bar_values, bar_pcts)):
    txt = ax1.text(val + (np.max(bar_values) * 0.015), i, f"{pct:.1f}%", 
                   transform=ax2.transData, 
                   va='center', ha='left', fontsize=13+num, fontweight='bold', 
                   color='#222222', zorder=100)
    # 这一行就是点睛之笔！
    txt.set_path_effects([pe.withStroke(linewidth=3, foreground='white')])

ax2.set_xlabel('Mean |SHAP Value| (Feature Importance)', fontsize=16+num, fontweight='bold', labelpad=15)
ax2.tick_params(axis='x', labelsize=14+num, colors='#333333')
ax2.spines['top'].set_color('#333333')

ax1.set_zorder(ax2.get_zorder() + 1)
ax1.patch.set_visible(False) 

plt.tight_layout()
out_fig = r"E:\lunwen3\process\空气质量数据处理\SCI_Figure_SHAP_Combo_Plot_Glow.png"
# plt.savefig(out_fig, bbox_inches='tight', dpi=600)
plt.show()

print("🎉 绘图完成！图例字体已完美放大并对齐主坐标轴！")