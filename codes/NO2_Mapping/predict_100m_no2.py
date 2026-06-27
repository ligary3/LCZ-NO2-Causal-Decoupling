import rasterio
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
import os
from tqdm import tqdm
from scipy.interpolate import griddata
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. 路径配置 (锁定 2024-07-18 终局之战)
# ==========================================
work_dir = r"E:\lunwen3\process\预测\20240718" 

# 🚨 只需加载唯一的终极降尺度大模型 (v6)
model_final_path = r"E:\lunwen3\process\空气质量数据处理\BTH_NO2_Final_Downscale_RBF_v7.json"
feat_final_path = r"E:\lunwen3\process\空气质量数据处理\BTH_NO2_Final_FeatureList_RBF_v7.joblib"
output_tif = os.path.join(work_dir, "BTH_NO2_Prediction_100m_20240718_Ultimate_RBF_v7.tif")

print("🚀 正在装载终极降尺度引擎...")
bst_final = xgb.Booster()
bst_final.load_model(model_final_path)
final_features = joblib.load(feat_final_path)

# ==========================================
# 2. TIFF 映射
# ==========================================
base_tif = os.path.join(work_dir, "BTH_BaseFeatures_20240718_100m.tif")
pop_tif = os.path.join(work_dir, "Aligned_POP_100m.tif")
aef_tif = os.path.join(work_dir, "Aligned_AEF_PCA_100m.tif")
lcz_tif = os.path.join(work_dir, "Aligned_LCZ_Prop_100m2.tif")
road_tif = os.path.join(work_dir, "Aligned_Road_Features_100m.tif")
geos_tif = os.path.join(work_dir, "BTH_GEOSCF_NO2_100m_20240718.tif")

base_names = ['longitude', 'latitude', 'WS', 'WD_sin', 'WD_cos', 't2m_c', 'surface_pressure', 
              'ERA5_RH', 'ERA5_BLH', 'ssr_value', 'TROPOMI_NO2', 'NTL', 
              'DEM', 'DSM', 'Month', 'DayOfYear']
pop_names = ['POP']
aef_names = [f'AEF_PC{i}' for i in range(1, 15)]
lcz_names = [f'LCZ_{i}' for i in range(1, 18)]
road_names = ['Dist_to_Road', 'Road_Gaussian_300m', 'Road_Gaussian_1000m', 'Road_Gaussian_3000m']

# ==========================================
# 3. 👑 核心杀招：全域锚点提取 (为 RBF 提供上帝视角)
# ==========================================
print("🌍 正在扫视全图，提取真实物理误差比例锚点...")
with rasterio.open(base_tif) as src_base, rasterio.open(geos_tif) as src_geos:
    lon_global = src_base.read(1).flatten()
    lat_global = src_base.read(2).flatten()
    tropo_global = src_base.read(11).flatten() * 1000000.0  # 量纲对齐
    geos_global = src_geos.read(1).flatten()

    valid_mask = (tropo_global > -9000) & (geos_global > -9000) & ~np.isnan(tropo_global) & (geos_global > 0)
    
    valid_indices = np.where(valid_mask)[0]
    sample_indices = valid_indices[::2500] 

    pts_global = np.column_stack((lon_global[sample_indices], lat_global[sample_indices]))
    vals_global = tropo_global[sample_indices] / (geos_global[sample_indices] + 0.1)

    del lon_global, lat_global, tropo_global, geos_global, valid_mask, valid_indices

print(f"🎯 成功提取 {len(pts_global)} 个有效物理锚点！开始执行分块 RBF 降尺度...")

# ==========================================
# 4. 终极单级预测引擎
# ==========================================
with rasterio.open(base_tif) as src_base:
    meta = src_base.meta.copy()
    meta.update(count=1, dtype='float32', nodata=-9999, compress='lzw')
    
    s_pop = rasterio.open(pop_tif); s_aef = rasterio.open(aef_tif)
    s_lcz = rasterio.open(lcz_tif); s_road = rasterio.open(road_tif)
    s_geos = rasterio.open(geos_tif)
    
    check_done = False # 打印控制开关

    with rasterio.open(output_tif, 'w', **meta) as dst:
        windows = [window for ij, window in src_base.block_windows()]
        
        for window in tqdm(windows, desc="RBF 物理重构与 100m 极清反演中..."):
            d_base = src_base.read(window=window)
            d_pop = s_pop.read(window=window); d_aef = s_aef.read(window=window)
            d_lcz = s_lcz.read(window=window); d_road = s_road.read(window=window)
            d_geos = s_geos.read(window=window)
            
            h, w = d_base.shape[1], d_base.shape[2]
            block_df = pd.DataFrame()
            
            for i, name in enumerate(base_names): block_df[name] = d_base[i].flatten()
            for i, name in enumerate(pop_names): block_df[name] = d_pop[i].flatten()
            for i, name in enumerate(aef_names): block_df[name] = d_aef[i].flatten()
            for i, name in enumerate(lcz_names): block_df[name] = d_lcz[i].flatten()
            for i, name in enumerate(road_names): block_df[name] = d_road[i].flatten()
            block_df['geos_no2_ppb'] = d_geos[0].flatten()
            
            block_df.replace([-9999.0, -9999], np.nan, inplace=True)
            block_df = block_df.rename(columns={'longitude': 'lon', 'latitude': 'lat'})
            
            # ---------------------------------------------------------
            # 🚨 量纲处理区
            # ---------------------------------------------------------
            # 保留你原本的代码不动
            block_df['TROPOMI_NO2'] = block_df['TROPOMI_NO2'] * 1000000.0 
            block_df['ssr_value'] = block_df['ssr_value'] * 1000000.0 
            
            # 加上刚刚发现的 LCZ 乘法
            block_df[lcz_names] = block_df[lcz_names] * 100.0
            
            block_df['Year_Factor'] = 2024
            block_df['Ventilation_Index'] = block_df['WS'] * block_df['ERA5_BLH']
            
            block_df[pop_names] = block_df[pop_names].fillna(0)
            block_df[lcz_names] = block_df[lcz_names].fillna(0)
            
            boundary_mask = ~block_df['DEM'].isna()
            preds = np.full(len(block_df), -9999, dtype=np.float32)
            
            if boundary_mask.any():
                try:
                    # RBF 重构
                    req_pts = block_df.loc[boundary_mask, ['lon', 'lat']].values
                    if len(pts_global) < 4:
                        block_ratio = np.full(len(req_pts), 1.0)
                    else:
                        block_ratio = griddata(pts_global, vals_global, req_pts, method='cubic')
                        if np.isnan(block_ratio).any():
                            block_ratio_near = griddata(pts_global, vals_global, req_pts, method='nearest')
                            block_ratio[np.isnan(block_ratio)] = block_ratio_near[np.isnan(block_ratio)]
                    
                    tropo_simulated = block_df.loc[boundary_mask, 'geos_no2_ppb'].values * block_ratio
                    tropo_real = block_df.loc[boundary_mask, 'TROPOMI_NO2'].values
                    tropo_seamless = np.where(np.isnan(tropo_real), tropo_simulated, tropo_real)
                    
                    block_df.loc[boundary_mask, 'TROPOMI_NO2_Seamless'] = tropo_seamless
                    block_df.loc[boundary_mask, 'TROPOMI_BLH_Ratio_Seamless'] = tropo_seamless / (block_df.loc[boundary_mask, 'ERA5_BLH'] + 1.0)
                    
                    # 取出最终喂给 XGBoost 的特征
                    X_final = block_df.loc[boundary_mask, final_features]
                    
                    # ---------------------------------------------------------
                    # 👀 打印拦截！只在第一个有效块打印前 3 行数据
                    # ---------------------------------------------------------
                    if not check_done:
                        print("\n" + "="*50)
                        print("👀 拦截点：即将进入 XGBoost 模型的数据抽查 (前3行)")
                        # 挑选几个关键变量展示
                        check_cols = ['TROPOMI_NO2_Seamless', 'ssr_value', 'LCZ_1', 'LCZ_2','LCZ_3','LCZ_4','LCZ_5','LCZ_6','LCZ_7','LCZ_8','LCZ_9','LCZ_10','LCZ_11','LCZ_12','LCZ_13','LCZ_14','LCZ_15','LCZ_16','LCZ_17']
                        print(X_final[check_cols].head(3))
                        print("="*50 + "\n")
                        check_done = True
                    # ---------------------------------------------------------

                    dmatrix_final = xgb.DMatrix(X_final)
                    preds[boundary_mask] = np.expm1(bst_final.predict(dmatrix_final))
                    
                except Exception as e:
                    import traceback
                    print(f"❌ 预测出错，请检查: {e}")
                    print(traceback.format_exc())
                    break
            
            dst.write(preds.reshape(h, w), 1, window=window)

    for s in [s_pop, s_aef, s_lcz, s_road, s_geos]: s.close()

print(f"🎉 预测执行完毕！输出文件：{output_tif}")