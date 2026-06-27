import pandas as pd
import geopandas as gpd
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error 
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

print("🚀 正在初始化留一城市验证 (LOCO-CV) 环境，全面同步 FS5 引擎...")

# ==========================================
# 1. 核心配置与特征定义 (严格同步消融实验 FS5)
# ==========================================
TARGET = 'NO2'

# 🌟 修复 1：严格同步抗过拟合引擎！
XGB_CONFIG = {
    'n_estimators': 800,       
    'max_depth': 5,            # 降低树深，防止死记硬背
    'learning_rate': 0.05,     
    'subsample': 0.85,
    'colsample_bytree': 1.0,   
    'reg_lambda': 5.0,         # 强力惩罚项
    'tree_method': 'hist',     
    'device': 'cuda',          
    'random_state': 42
}

# 🌟 修复 2：同步 FS5 纯物理无坐标特征
BASE = [
    'Year_Factor', 'Month', 'DayOfYear', 
    'WS', 'WD_sin', 'WD_cos', 't2m_c', 'surface_pressure', 'ERA5_RH', 'ssr_value', 
    'TROPOMI_NO2_Seamless', 'geos_no2_ppb', 'TROPOMI_BLH_Ratio_Seamless', 'Ventilation_Index',
    'NTL', 'POP', 'DEM' 
]
VAR_3D = [f'LCZ_{i}' for i in range(1, 18)]
VAR_TRAFFIC = ['Dist_to_Road', 'Road_Gaussian_300m', 'Road_Gaussian_1000m', 'Road_Gaussian_3000m']
VAR_AEF = [f'AEF_PC{i}' for i in range(1, 15)]

final_features = BASE + VAR_3D + VAR_TRAFFIC + VAR_AEF 

# ==========================================
# 2. 站点归城：空间打标签
# ==========================================
print("📍 正在给站点打上城市标签...")
df_stations = pd.read_csv(r"E:\lunwen3\process\空气质量数据处理\BTH_125_Stations_Coords.csv")
gdf_stations = gpd.GeoDataFrame(df_stations, geometry=gpd.points_from_xy(df_stations.lon, df_stations.lat), crs="EPSG:4326")

cities = {
    'Beijing': gpd.read_file(r"E:\lunwen3\边界\city\beijing.shp").to_crs("EPSG:4326"),
    'Tianjin': gpd.read_file(r"E:\lunwen3\边界\city\tianjian.shp").to_crs("EPSG:4326"),
    'Shijiazhuang': gpd.read_file(r"E:\lunwen3\边界\city\shijiazhuang.shp").to_crs("EPSG:4326")
}

df_stations['City_Label'] = 'Others'
for city_name, city_gdf in cities.items():
    within_city = gpd.sjoin(gdf_stations, city_gdf, how='inner', predicate='intersects')
    df_stations.loc[df_stations['station'].isin(within_city['station']), 'City_Label'] = city_name

# ==========================================
# 3. 数据加载与合并
# ==========================================
# 🌟 修复 3：使用 v6 无缝融合大表！
df_master = pd.read_csv(r"E:\lunwen3\process\空气质量数据处理\根据站点对应位置准备训练数据\训练数据\融合大表\BTH_Master_Seamless_SpatialRBF_v7.csv")
df_master['date'] = pd.to_datetime(df_master['date'])
# 确保包含 City_Label
df_master = pd.merge(df_master, df_stations[['station', 'City_Label']], on='station', how='left')

# ==========================================
# 4. 留一城市循环测试 (LOCO-CV)
# ==========================================
test_cities = ['Beijing', 'Tianjin', 'Shijiazhuang']
loco_plot_data = {}

print("⚔️ 开始执行极其硬核的留城盲测...")
for t_city in test_cities:
    # 切分数据集并清理空值
    train_df = df_master[df_master['City_Label'] != t_city].dropna(subset=final_features + [TARGET])
    test_df = df_master[df_master['City_Label'] == t_city].dropna(subset=final_features + [TARGET])
    
    # 剔除 <= 0 的异常值
    train_df = train_df[train_df[TARGET] > 0]
    test_df = test_df[test_df[TARGET] > 0]
    
    if len(test_df) == 0: continue
    
    X_train = train_df[final_features].astype(np.float32).values
    y_train_real = train_df[TARGET].values
    X_test = test_df[final_features].astype(np.float32).values
    y_test_real = test_df[TARGET].values
    
    # 🌟 修复 4：执行对数变换和高污染权重！
    y_train_log = np.log1p(y_train_real)
    w_train = np.where(y_train_real > 60, 2.0, 1.0)
    
    model = xgb.XGBRegressor(**XGB_CONFIG)
    model.fit(X_train, y_train_log, sample_weight=w_train)
    
    # 预测并用指数还原
    preds_log = model.predict(X_test)
    preds = np.expm1(preds_log)
    
    r2 = r2_score(y_test_real, preds)
    rmse = np.sqrt(mean_squared_error(y_test_real, preds))
    mae = mean_absolute_error(y_test_real, preds)
    
    print(f"✅ 盲测城市: {t_city:<15} | 站点数: {test_df['station'].nunique():>2} | R²: {r2:.3f} | RMSE: {rmse:.2f} | MAE: {mae:.2f}")
    
    loco_plot_data[t_city] = {
        'y_true': y_test_real,
        'y_pred': preds,
        'r2': r2,
        'rmse': rmse,
        'mae': mae, 
        'n': len(y_test_real)
    }

print("\n🎨 开始绘制 SCI 级别留城盲测密度散点图...")

# ==========================================
# 5. 绘制 1x3 高质感密度散点图
# ==========================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.linewidth'] = 1.0 # 纤细边框

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
VMIN, VMAX = 0, 140 # 全局统一量程

for ax, t_city in zip(axes, test_cities):
    if t_city not in loco_plot_data: continue
        
    data = loco_plot_data[t_city]
    y_t = data['y_true']
    y_p = data['y_pred']
    
    # 物理宽高比 1:1，确保图绝对方正
    ax.set_aspect('equal', adjustable='box')
    
    # 采用 turbo 色带，bins=70 彻底消除竖条纹，vmax 控制视觉热度
    h = ax.hist2d(y_t, y_p, bins=70, range=[[VMIN, VMAX], [VMIN, VMAX]], 
                  cmap='jet', cmin=1, vmax=350, alpha=0.9)
    
    ax.plot([VMIN, VMAX], [VMIN, VMAX], 'k--', linewidth=1.2, alpha=0.7, label='1:1 Line')
    
    m, b = np.polyfit(y_t, y_p, 1)
    ax.plot([VMIN, VMAX], [m * VMIN + b, m * VMAX + b], 
            color='#D62728', linewidth=2, label='Linear Fit')
    
    textstr = '\n'.join((
        f"N = {data['n']:,}",
        f"R² = {data['r2']:.3f}",
        f"RMSE = {data['rmse']:.2f} μg/m³",
        f"MAE = {data['mae']:.2f} μg/m³",
        f"Slope = {m:.2f}"
    ))
    
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='#CCCCCC', linewidth=1.0)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=13,
            verticalalignment='top', bbox=props, fontweight='bold')
    
    ax.set_title(f"{t_city}", fontsize=16, fontweight='bold', pad=12)
    ax.set_xlabel('Observed NO₂ Concentration (μg/m³)', fontsize=14, fontweight='bold') 
    
    if ax == axes[0]:
        ax.set_ylabel('Predicted NO₂ Concentration (μg/m³)', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=12, frameon=False)
        
    ax.set_xlim(VMIN, VMAX)
    ax.set_ylim(VMIN, VMAX)
    ax.tick_params(axis='both', labelsize=12, width=1.0)
    ax.grid(True, linestyle=':', alpha=0.5, linewidth=0.8)

# 精致等高 Colorbar
cbar = fig.colorbar(h[3], ax=axes.ravel().tolist(), pad=0.02, aspect=30, shrink=0.8)
cbar.set_label('Point Density (Count)', fontsize=13, fontweight='bold')
cbar.ax.tick_params(labelsize=12, width=1.0)
cbar.outline.set_linewidth(1.0)

plt.savefig("SCI_Figure_LOCO_DensityScatter.png", bbox_inches='tight')
plt.show()

print("🎉 绘图完成！三大盲测城市高逼格散点图已出炉！")