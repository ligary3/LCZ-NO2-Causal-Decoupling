import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. SCI 顶刊全局图表格式设置
# ==========================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'  
plt.rcParams['axes.linewidth'] = 1.2       

# ==========================================
# 2. 读取数据与预处理 (替换为你的真实路径)
# ==========================================
file_path = r"E:\lunwen3\process\空气质量数据处理\根据站点对应位置准备训练数据\训练数据\BTH_AEF_64D_2018_2024.csv"
df = pd.read_csv(file_path)

aef_cols = [f'A{str(i).zfill(2)}' for i in range(64)]
X = df[aef_cols].values

# 🚨 学术铁律：PCA 前标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================================
# 3. 执行 PCA 降维
# ==========================================
pca = PCA().fit(X_scaled)
evr = pca.explained_variance_ratio_ * 100        
cum_evr = np.cumsum(evr)                         

# ==========================================
# 4. 👑 绘制 Q1 区标准的“双色映射”碎石图
# ==========================================
fig, ax1 = plt.subplots(figsize=(8, 5.5), dpi=300)

n_components_to_show = 15
x_labels = [f'PC{i+1}' for i in range(n_components_to_show)]

color_bar = '#4C72B0'  # 莫兰迪蓝
color_line = '#C44E52' # 警示红

# 左轴：柱状图 (单个 PC 解释率)
bars = ax1.bar(x_labels, evr[:n_components_to_show], color=color_bar, alpha=0.8, edgecolor='black', linewidth=0.8, label='Individual Variance')
ax1.set_xlabel('Principal Components (PCs)', fontsize=14, fontweight='bold')
# 让左轴文字带点蓝，暗示它对应柱子
ax1.set_ylabel('Explained Variance (%)', fontsize=14, fontweight='bold', color='#334C75') 
ax1.tick_params(axis='y', labelcolor='#334C75', labelsize=12)
ax1.tick_params(axis='x', labelsize=12)
ax1.spines['left'].set_color(color_bar) # 
ax1.set_ylim(0, max(evr) + 5)

# 右轴：折线图 (累计解释率)
ax2 = ax1.twinx()
line = ax2.plot(x_labels, cum_evr[:n_components_to_show], color=color_line, marker='o', markersize=6, linewidth=2.5, markeredgecolor='white', label='Cumulative Variance')

# 🌟 核心美学升级：右轴全面变红
ax2.set_ylabel('Cumulative Explained Variance (%)', fontsize=14, fontweight='bold', color=color_line)
ax2.tick_params(axis='y', labelcolor=color_line, color=color_line, labelsize=12) # 刻度文字和刻度线全变红
ax2.spines['right'].set_color(color_line) # 右边框变红
ax2.spines['right'].set_linewidth(1.5)
ax2.set_ylim(0, 105)
ax2.spines['left'].set_color('#334C75') 

# 阈值线
threshold = cum_evr[13] # 提取第 14 个主成分的真实累计方差 (假设接近 85%-90%)
ax2.axhline(y=threshold, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, zorder=0)
ax2.text(8.5, threshold + 2, f'PC14 Threshold (~{threshold:.1f}%)', color='gray', fontsize=12, va='bottom', ha='center', fontstyle='italic')

# 合并图例
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='center right', fontsize=11, frameon=False)

# 去除顶部边框
ax1.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)
#左边刻度线也设置蓝色
ax1.tick_params(axis='y', labelcolor='#334C75',color='#334C75', labelsize=12)
plt.tight_layout()
plt.savefig("SCI_Figure_AEF_PCA_ScreePlot_Colored.png", bbox_inches='tight')
plt.show()

# ==========================================
# 5. 提取特征导出
# ==========================================
TARGET_DIMS = 14
pca_final = PCA(n_components=TARGET_DIMS)
X_pca = pca_final.fit_transform(X_scaled)

df_pca = df[['year', 'station']].copy()
for i in range(TARGET_DIMS):
    df_pca[f'AEF_PC{i+1}'] = X_pca[:, i]

output_name = f"BTH_AEF_PCA_{TARGET_DIMS}D_Final.csv"
#df_pca.to_csv(output_name, index=False)
print(f"✅ 完美出图并导出！")