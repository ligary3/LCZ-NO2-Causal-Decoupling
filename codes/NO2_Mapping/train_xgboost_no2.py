import pandas as pd
import numpy as np
import xgboost as xgb
import joblib

# 载入刚才 RBF 生成的 v6 大表
data_path = r"E:\lunwen3\process\空气质量数据处理\根据站点对应位置准备训练数据\训练数据\融合大表\BTH_Master_Seamless_SpatialRBF_v7.csv"
df_master = pd.read_csv(data_path)
TARGET = 'NO2'

lcz_cols = [c for c in df_master.columns if c.startswith('LCZ_')]
aef_cols = [c for c in df_master.columns if 'AEF_PC' in c]

# 特征清单：绝对纯净，没有任何经纬度！
BASE = [
    'Year_Factor', 'WS', 'WD_sin', 'WD_cos', 't2m_c', 
    'surface_pressure', 'ERA5_RH', 'ERA5_BLH', 'ssr_value', 
    'TROPOMI_NO2_Seamless', 'geos_no2_ppb', 'TROPOMI_BLH_Ratio_Seamless', 'Ventilation_Index', 
    'NTL', 'POP', 'DEM', 'DSM', 'Month', 'DayOfYear' 
]
VAR_TRAFFIC = ['Dist_to_Road', 'Road_Gaussian_300m', 'Road_Gaussian_1000m', 'Road_Gaussian_3000m']
final_features = BASE + lcz_cols + VAR_TRAFFIC + aef_cols 

final_train_data = df_master.dropna(subset=final_features + [TARGET]).copy()
final_train_data = final_train_data[final_train_data[TARGET] > 0].reset_index(drop=True)

X_final = final_train_data[final_features].astype(np.float32).values
y_log_final = np.log1p(final_train_data[TARGET]).astype(np.float32).values
weights_final = np.where(final_train_data[TARGET] > 60, 2.0, 1.0)

print("🚀 正在训练唯一的 100m 极清降尺度引擎...")
# 👑 柔化引擎：彻底锁死这些抗锯齿参数！
final_model = xgb.XGBRegressor(
    n_estimators=800,      
    max_depth=8,               
    learning_rate=0.05,     
    subsample=0.7,             # 👑 锁定为 0.7：增加朦胧感，柔化树模型边缘
    colsample_bytree=0.8,
    min_child_weight=10,       # 👑 锁定为 10：强力砂纸，磨平孤立噪点
    tree_method='hist',     
    device='cuda',          
    random_state=99
)
final_model.fit(X_final, y_log_final, sample_weight=weights_final)

final_model.save_model(r"E:\lunwen3\process\空气质量数据处理\BTH_NO2_Final_Downscale_RBF_v7.json")
joblib.dump(final_features, r"E:\lunwen3\process\空气质量数据处理\BTH_NO2_Final_FeatureList_RBF_v7.joblib")
print("✅ 终极大模型重训完成！")