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
# 1. 路径配置
# ==========================================
work_dir = r"E:\lunwen3\process\预测\20240111" 

model_final_path = r"E:\lunwen3\process\空气质量数据处理\BTH_NO2_Final_Downscale_RBF_v7.json"
feat_final_path = r"E:\lunwen3\process\空气质量数据处理\BTH_NO2_Final_FeatureList_RBF_v7.joblib"

# 👑 核心输出：这不是浓度图，这是一张“浓度下降了多少”的收益图！
output_benefit_tif = os.path.join(work_dir, "BTH_NO2_NetBenefit_LCZ1_to_LCZ4_15pct.tif")

print("🚀 正在装载终极降尺度引擎 (反事实干预模式)...")
bst_final = xgb.Booster()
bst_final.load_model(model_final_path)
final_features = joblib.load(feat_final_path)

# ==========================================
# 2. TIFF 映射 (保持不变)
# ==========================================
base_tif = os.path.join(work_dir, "BTH_BaseFeatures_20240111_100m.tif")
pop_tif = os.path.join(work_dir, "Aligned_POP_100m.tif")
aef_tif = os.path.join(work_dir, "Aligned_AEF_PCA_100m.tif")
lcz_tif = os.path.join(work_dir, "Aligned_LCZ_Prop_100m2.tif")
road_tif = os.path.join(work_dir, "Aligned_Road_Features_100m.tif")
geos_tif = os.path.join(work_dir, "BTH_GEOSCF_NO2_100m_20240111.tif")

base_names = ['longitude', 'latitude', 'WS', 'WD_sin', 'WD_cos', 't2m_c', 'surface_pressure', 
              'ERA5_RH', 'ERA5_BLH', 'ssr_value', 'TROPOMI_NO2', 'NTL', 
              'DEM', 'DSM', 'Month', 'DayOfYear']
pop_names = ['POP']
aef_names = [f'AEF_PC{i}' for i in range(1, 15)]
lcz_names = [f'LCZ_{i}' for i in range(1, 18)]
road_names = ['Dist_to_Road', 'Road_Gaussian_300m', 'Road_Gaussian_1000m', 'Road_Gaussian_3000m']

# ==========================================
# 3. 提取物理锚点 (保持不变)
# ==========================================
print("🌍 正在扫视全图，提取真实物理误差比例锚点...")
with rasterio.open(base_tif) as src_base, rasterio.open(geos_tif) as src_geos:
    lon_global = src_base.read(1).flatten()
    lat_global = src_base.read(2).flatten()
    tropo_global = src_base.read(11).flatten() * 1000000.0  
    geos_global = src_geos.read(1).flatten()

    valid_mask = (tropo_global > -9000) & (geos_global > -9000) & ~np.isnan(tropo_global) & (geos_global > 0)
    valid_indices = np.where(valid_mask)[0]
    sample_indices = valid_indices[::2500] 

    pts_global = np.column_stack((lon_global[sample_indices], lat_global[sample_indices]))
    vals_global = tropo_global[sample_indices] / (geos_global[sample_indices] + 0.1)

    del lon_global, lat_global, tropo_global, geos_global, valid_mask, valid_indices

# ==========================================
# 4. 👑 核心杀招：拦截篡改与双重预测
# ==========================================
with rasterio.open(base_tif) as src_base:
    meta = src_base.meta.copy()
    meta.update(count=1, dtype='float32', nodata=-9999, compress='lzw')
    
    s_pop = rasterio.open(pop_tif); s_aef = rasterio.open(aef_tif)
    s_lcz = rasterio.open(lcz_tif); s_road = rasterio.open(road_tif)
    s_geos = rasterio.open(geos_tif)

    with rasterio.open(output_benefit_tif, 'w', **meta) as dst:
        windows = [window for ij, window in src_base.block_windows()]
        
        for window in tqdm(windows, desc="正在进行城市微更新虚拟模拟..."):
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
            
            # 量纲与组合特征处理
            block_df['TROPOMI_NO2'] = block_df['TROPOMI_NO2'] * 1000000.0 
            block_df['ssr_value'] = block_df['ssr_value'] * 1000000.0 
            block_df[lcz_names] = block_df[lcz_names] * 100.0
            
            block_df['Year_Factor'] = 2024
            block_df['Ventilation_Index'] = block_df['WS'] * block_df['ERA5_BLH']
            
            block_df[pop_names] = block_df[pop_names].fillna(0)
            block_df[lcz_names] = block_df[lcz_names].fillna(0)
            
            boundary_mask = ~block_df['DEM'].isna()
            
            # 这个 preds 阵列用来存“收益值 (Difference)”
            preds_benefit = np.full(len(block_df), -9999, dtype=np.float32)
            
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
                    
                    # 1. 取得真实的 Baseline 特征
                    X_base = block_df.loc[boundary_mask, final_features].copy()
                    
                    # ---------------------------------------------------------
                    # 👑 赛博手术台：反事实干预 (Counterfactual Intervention)
                    # ---------------------------------------------------------
                    X_cf = X_base.copy()
                    
                    # 设定最大转化阈值：每个像元最多把 15% 的区域从 LCZ1 改为 LCZ4
                    MAX_TRANSFER = 15.0 
                    
                    # 计算实际可转化的面积 (如果该地 LCZ1 本来就不到 15%，就全转；超过 15%，只转 15%)
                    transfer_area = np.minimum(X_cf['LCZ_1'], MAX_TRANSFER)
                    
                    # 执行手术：LCZ 1 减少，对应的面积转移给 LCZ 4
                    X_cf['LCZ_1'] = X_cf['LCZ_1'] - transfer_area
                    X_cf['LCZ_4'] = X_cf['LCZ_4'] + transfer_area
                    # ---------------------------------------------------------

                    # 2. 模型平行宇宙双重预测
                    dmatrix_base = xgb.DMatrix(X_base)
                    dmatrix_cf = xgb.DMatrix(X_cf)
                    
                    pred_base_no2 = np.expm1(bst_final.predict(dmatrix_base))
                    pred_cf_no2 = np.expm1(bst_final.predict(dmatrix_cf))
                    
                    # 3. 计算政策净收益：基础浓度 - 改造后浓度 
                    # 正值表示 NO2 降了 (好事)，负值表示涨了，0表示没变(因为没有LCZ1)
                    benefit = pred_base_no2 - pred_cf_no2
                    
                    preds_benefit[boundary_mask] = benefit
                    
                except Exception as e:
                    import traceback
                    print(f"❌ 预测出错，请检查: {e}")
                    print(traceback.format_exc())
                    break
            
            dst.write(preds_benefit.reshape(h, w), 1, window=window)

    for s in [s_pop, s_aef, s_lcz, s_road, s_geos]: s.close()

print(f"🎉 反事实因果收益地图生成完毕！\n输出文件：{output_benefit_tif}")