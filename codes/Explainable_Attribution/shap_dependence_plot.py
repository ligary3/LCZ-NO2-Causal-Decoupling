import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe  # 👑 引入文字描边特效库

from statsmodels.nonparametric.smoothers_lowess import lowess
import warnings

warnings.filterwarnings('ignore')
print("🚀 启动顶刊 3x3 九宫格 SHAP 机制矩阵 (带白色光晕的终极典藏版)...")

# ==========================================

# 1. 严格对齐你最新字典的 9 大特征

# ==========================================

features_to_plot = [

    'TROPOMI NO₂ Column',            # 1. 卫星背景 (匹配字典)

    'Temperature (T2m)',             # 2. 气温

    'Relative Humidity (RH)',        # 3. 湿度

    'Wind Speed (WS)',               # 4. 风速

    'Distance to Road (DTR)',        # 5. 👑 交通距离 (修补了这里的 KeyError)

    'LCZ 1',                         # 6. 紧凑高层

    'LCZ 3',                         # 7. 紧凑低层

    'LCZ 8',                         # 8. 大型低层

    'LCZ A'                          # 9. 密集林地

]



# 交互特征 (颜色映射) —— 同样严格对齐字典

interaction_features = [

    'Temperature (T2m)',             # 卫星背景 vs 温度

    'Relative Humidity (RH)',        # 温度 vs 湿度

    'Temperature (T2m)',             # 湿度 vs 温度

    'LCZ 1',                         # 风速 vs 紧凑高层

    'LCZ 1',                         # 距道路 vs 紧凑高层

    'Wind Speed (WS)',               # 紧凑高层 vs 风速

    'Wind Speed (WS)',               # 紧凑低层 vs 风速

    'Wind Speed (WS)',               # 工业区 vs 风速

    'Distance to Road (DTR)'         # 林地 vs 交通距离 (修补了这里)

]

# ==========================================
# 2. 🎨 绘制 3x3 象限级阈值矩阵
# ==========================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

fig, axes = plt.subplots(3, 3, figsize=(19, 15), dpi=300)
axes = axes.flatten()

for i, (feat, inter_feat) in enumerate(zip(features_to_plot, interaction_features)):
    ax = axes[i]
    
    col_idx = X_sample_plot.columns.get_loc(feat)
    x_val = X_sample_plot[feat].values
    y_val = shap_values[:, col_idx]
    c_val = X_sample_plot[inter_feat].values 
    
    x_min, x_max = np.percentile(x_val, 1), np.percentile(x_val, 99) 
    y_min, y_max = y_val.min(), y_val.max()
    
    # 0 线底座
    ax.axhline(0, color='gray', linestyle='--', linewidth=1.5, zorder=1)
    
    # LOWESS 平滑曲线 
    smoothed = lowess(y_val, x_val, frac=0.2)
    lowess_x, lowess_y = smoothed[:, 0], smoothed[:, 1]
    
    # 智能象限着色与阈值标注
    signs = np.sign(lowess_y)
    crossings = np.where(np.diff(signs))[0]
    
    if len(crossings) > 0:
        cross_idx = crossings[0]
        threshold_x = lowess_x[cross_idx]
        
        crosses_upward = lowess_y[min(cross_idx + 1, len(lowess_y)-1)] > 0
        
        if crosses_upward:
            ax.fill_between([threshold_x, x_max], 0, y_max * 1.25, facecolor='#E8F4F8', alpha=0.6, zorder=0)
            ax.fill_between([x_min, threshold_x], y_min * 1.25, 0, facecolor='#F8E8E8', alpha=0.6, zorder=0)
        else:
            ax.fill_between([x_min, threshold_x], 0, y_max * 1.25, facecolor='#E8F4F8', alpha=0.6, zorder=0)
            ax.fill_between([threshold_x, x_max], y_min * 1.25, 0, facecolor='#F8E8E8', alpha=0.6, zorder=0)
            
        ax.axvline(threshold_x, color='#D32F2F', linestyle='--', linewidth=1.5, zorder=3)
        ax.scatter(threshold_x, 0, color='#D32F2F', s=60, zorder=6, edgecolors='white', linewidths=1.5)
        
        x_offset = (x_max - x_min) * 0.02
        y_offset = (y_max - y_min) * 0.06
        txt_val = f'{threshold_x:.1f}' if ('Speed' in feat or 'Temperature' in feat) else f'{threshold_x:.2f}'
        
        # 👑 为阈值文字添加纯白描边特效
        txt = ax.text(threshold_x + x_offset, 0 + y_offset, txt_val, 
                      color='#D32F2F', fontsize=14, fontweight='bold', ha='left', va='bottom', zorder=7)
        txt.set_path_effects([pe.withStroke(linewidth=3, foreground='white')])
        
    else:
        if np.mean(lowess_y) > 0:
            ax.axhspan(0, y_max * 1.25, facecolor='#E8F4F8', alpha=0.6, zorder=0)
        else:
            ax.axhspan(y_min * 1.25, 0, facecolor='#F8E8E8', alpha=0.6, zorder=0)

    # 散点和曲线图层
    sc = ax.scatter(x_val, y_val, c=c_val, cmap='coolwarm', s=12, alpha=0.5, zorder=2)
    ax.plot(lowess_x, lowess_y, color='#D32F2F', linewidth=2.5, zorder=4)
    
    # 标题居中
    letters = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)']
    ax.set_title(f"{letters[i]} {feat}", fontsize=17, fontweight='bold', pad=12, loc='center')
    ax.set_xlabel('Feature value', fontsize=13)
    ax.set_ylabel('SHAP value', fontsize=13)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min * 1.2, y_max * 1.2)
    ax.tick_params(axis='both', labelsize=12)
    
    # Colorbar
    cbar = plt.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label(inter_feat, fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    # 透明图例
    lowess_line = plt.Line2D([0], [0], color='#D32F2F', linewidth=2, label='Lowess curve')
    pos_patch = mpatches.Patch(color='#E8F4F8', label='Positive')
    neg_patch = mpatches.Patch(color='#F8E8E8', label='Negative')
    ax.legend(handles=[lowess_line, pos_patch, neg_patch], loc='best', fontsize=11, frameon=False)

plt.tight_layout()
out_fig = r"E:\lunwen3\process\空气质量数据处理\SCI_Figure_SHAP_Matrix_3x3_Quadrant_Glow.png"
# plt.savefig(out_fig, bbox_inches='tight', dpi=600)
plt.show()

print(f"🎉 描边完成！不管散点怎么密集，红色的阈值数值都将极其醒目！")