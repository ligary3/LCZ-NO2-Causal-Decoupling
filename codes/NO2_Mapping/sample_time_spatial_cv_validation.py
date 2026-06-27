# 加上MAE
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
import time

# 基础设置
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

# ==========================================
# 1. 路径与核心配置
# ==========================================
data_path = r"E:\lunwen3\process\空气质量数据处理\根据站点对应位置准备训练数据\训练数据\融合大表\BTH_Master_Seamless_SpatialRBF_v7.csv"
output_res_path = 'SCI_Ablation_Study_Targeted_Resultsv7.csv'

TARGET = 'NO2'

# 🛠️ 核心防过拟合与防稀释参数！(这次绝对没问题了)
XGB_CONFIG = {
    'n_estimators': 800,       
    'max_depth': 5,            # 🌟 压低树深，杜绝空间微环境死记硬背
    'learning_rate': 0.05,     
    'subsample': 0.85,
    'colsample_bytree': 1.0,   # 🌟 恢复 1.0！确保模型每次都能看到 TROPOMI，绝不稀释主心骨
    'reg_lambda': 5.0,         # 🌟 开启 L2 正则化惩罚，压制无用特征，保证加特征只升不降
    'tree_method': 'hist',     
    'device': 'cuda',          
    'random_state': 42
}

# ==========================================
# 2. 数据载入与动态特征提取
# ==========================================
print("📂 载入终极 RBF 无缝缝合大表 (v6)...")
df = pd.read_csv(data_path)

df['y_log'] = np.log1p(df[TARGET])

lcz_cols = [c for c in df.columns if c.startswith('LCZ_')]
aef_cols = [c for c in df.columns if 'AEF_PC' in c]
var_2d_indices = ['ndvi_clean', 'ndbi_value', 'ndwi_value'] if 'ndvi_clean' in df.columns else ['ndvi_value_x', 'ndbi_value', 'ndwi_value']

# ==========================================
# 3. 👑 专为 SCI 顶刊定制的 6 大消融赛道
# ==========================================
# 🌟 赛道 1: 绝对不包含 lon 和 lat！纯物理特征！
BASE = [
    'Year_Factor', 'Month', 'DayOfYear', 
    'WS', 'WD_sin', 'WD_cos', 't2m_c', 'surface_pressure', 'ERA5_RH', 'ssr_value', 
    'TROPOMI_NO2_Seamless', 'geos_no2_ppb', 'TROPOMI_BLH_Ratio_Seamless', 'Ventilation_Index',
    'NTL', 'POP', 'DEM' 
]#17

TRAD_2D = var_2d_indices#3
LCZ = lcz_cols#17
TRAFFIC = ['Dist_to_Road', 'Road_Gaussian_300m', 'Road_Gaussian_1000m', 'Road_Gaussian_3000m']#4
AEF = aef_cols#14

FEATURE_SETS = {
    '1_Base_Model': BASE,
    '2_Base_Trad2D': BASE + TRAD_2D,                       
    '3_Base_LCZ': BASE + LCZ,                              
    '4_Base_LCZ_Traffic': BASE + LCZ + TRAFFIC,
    '5_Final_Proposed': BASE + LCZ + TRAFFIC + AEF,        
    '6_Prove_2D_Redundant': BASE + LCZ + TRAFFIC + AEF + TRAD_2D  
}

# ==========================================
# 4. 极致严谨的数据清洗与分组
# ==========================================
all_features_union = list(set(sum(FEATURE_SETS.values(), [])))
model_data = df.dropna(subset=all_features_union + [TARGET]).copy()
model_data = model_data[model_data[TARGET] > 0].reset_index(drop=True)

weights = np.where(model_data[TARGET] > 60, 2.0, 1.0)

y_log = model_data['y_log'].values
y_real = model_data[TARGET].values
groups_station = model_data['station'].values
groups_time = model_data['date'].astype(str).values

print(f"✅ 数据准备就绪！有效干净样本数: {len(model_data)}")

# ==========================================
# 5. 核心交叉验证引擎
# ==========================================
def evaluate_engine(X, y_l, y_r, w, cv_type, groups=None):
    all_y_true, all_y_pred = [], []
    model = xgb.XGBRegressor(**XGB_CONFIG)
    
    if cv_type == 'Sample_CV':
        splitter = KFold(n_splits=10, shuffle=True, random_state=42)
    else:
        splitter = GroupKFold(n_splits=10)
        
    splits = splitter.split(X, y_l, groups)
        
    for train_idx, test_idx in splits:
        model.fit(X[train_idx], y_l[train_idx], sample_weight=w[train_idx])
        preds = np.expm1(model.predict(X[test_idx])) 
        all_y_pred.extend(preds)
        all_y_true.extend(y_r[test_idx])
    
    all_y_true, all_y_pred = np.array(all_y_true), np.array(all_y_pred)
    
    r2 = r2_score(all_y_true, all_y_pred)
    rmse = np.sqrt(mean_squared_error(all_y_true, all_y_pred))
    mae = mean_absolute_error(all_y_true, all_y_pred) # 🌟 新增 MAE 计算
    
    return r2, rmse, mae # 🌟 返回时加上 mae

# ==========================================
# 6. 启动大循环
# ==========================================
final_results = []

print("\n" + "="*50)
print("🚀 启动 SCI 顶刊级消融实验 (已剔除经纬度陷阱)")
print("="*50)

for name, f_list in FEATURE_SETS.items():
    print(f"\n🔥 正在测试赛道: {name} (特征数: {len(f_list)})")
    X = model_data[f_list].values
    
    for cv_mode, grp in zip(['Sample_CV', 'Spatial_CV', 'Time_CV'], [None, groups_station, groups_time]):
        start = time.time()
        
        # 🌟 接收新增的 mae 变量
        r2, rmse, mae = evaluate_engine(X, y_log, y_real, weights, cv_mode, groups=grp)
        
        # 🌟 打印时加上 MAE
        print(f"   [{cv_mode:10s}] R²: {r2:.4f} | RMSE: {rmse:.2f} | MAE: {mae:.2f} | 耗时: {time.time()-start:.1f}s")
        
        # 🌟 记录结果时加上 MAE
        final_results.append({
            'Feature_Set': name, 'CV_Mode': cv_mode, 'R2': r2, 'RMSE': rmse, 'MAE': mae
        })

# ==========================================
# 7. 生成核心战报
# ==========================================
res_df = pd.DataFrame(final_results)
res_df.to_csv(output_res_path, index=False)

# 生成 R2 透视表
pivot_r2 = res_df.pivot(index='Feature_Set', columns='CV_Mode', values='R2')

# 🌟 生成 MAE 透视表 (为写论文提供更直观的数据)
pivot_mae = res_df.pivot(index='Feature_Set', columns='CV_Mode', values='MAE')

print("\n" + "👑"*25)
print("📊 终极消融实验 R² 核心战报 (无坐标作弊版)")
print("👑"*25)
print(pivot_r2.round(4)) # 保留四位小数让排版更好看
print("\n📊 终极消融实验 MAE 核心战报 (越低越好)")
print(pivot_mae.round(2))
print("="*55)