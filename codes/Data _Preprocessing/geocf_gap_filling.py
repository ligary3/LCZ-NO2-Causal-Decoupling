import pandas as pd
import numpy as np
from scipy.interpolate import griddata
from tqdm import tqdm

print("🌍 启动 RBF 空间插值引擎：纯物理重构卫星底板...")

# 1. 载入带有空值的原始融合大表 (注意路径)
data_path = r"E:\lunwen3\process\空气质量数据处理\根据站点对应位置准备训练数据\训练数据\融合大表\BTH_Master_Enhanced_DualEngine_v3.csv"
df = pd.read_csv(data_path)

# 2. 计算有真实观测时的“物理比例”
# 加上 0.1 防止除以 0
df['Ratio_True'] = df['TROPOMI_NO2'] / (df['geos_no2_ppb'] + 0.1)
df['Ratio_Filled'] = df['Ratio_True']

# 3. 👑 核心杀招：按天进行 RBF 空间插值 (Cubic + Nearest 兜底)
dates = df['date'].unique()
for d in tqdm(dates, desc="逐日空间比例同化中"):
    idx = df['date'] == d
    group = df[idx]
    
    # 找出当天有卫星的站点和没卫星的站点
    valid = group.dropna(subset=['Ratio_True', 'lon', 'lat'])
    missing = group[group['Ratio_True'].isna()]
    
    # 如果当天全晴天，跳过
    if len(missing) == 0:
        continue
        
    # 如果当天全阴天（极其罕见），默认比例为 1.0 (100% 信任 GEOS)
    if len(valid) == 0:
        df.loc[idx, 'Ratio_Filled'] = 1.0
        continue
        
    points = valid[['lon', 'lat']].values
    values = valid['Ratio_True'].values
    req_points = group[['lon', 'lat']].values
    
    # 动态插值：点数>=4用平滑的三次插值(Cubic)，否则用最近邻(Nearest)
    if len(valid) >= 4:
        res = griddata(points, values, req_points, method='cubic')
        # cubic 在外推边缘会产生 NaN，用 nearest 兜底
        if np.isnan(res).any():
            res_near = griddata(points, values, req_points, method='nearest')
            res[np.isnan(res)] = res_near[np.isnan(res)]
    else:
        res = griddata(points, values, req_points, method='nearest')
        
    df.loc[idx, 'Ratio_Filled'] = res

# 4. 物理重构：将无缝比例乘以背景场
df['TROPOMI_NO2_Seamless'] = np.where(
    df['TROPOMI_NO2'].isna(), 
    df['geos_no2_ppb'] * df['Ratio_Filled'], 
    df['TROPOMI_NO2']
)

# 重新计算交互指数
df['TROPOMI_BLH_Ratio_Seamless'] = df['TROPOMI_NO2_Seamless'] / (df['ERA5_BLH'] + 1.0)
df['Ventilation_Index'] = df['WS'] * df['ERA5_BLH']

# 保存这张彻底告别 ML 伪影的终极神表
output_path = r"E:\lunwen3\process\空气质量数据处理\根据站点对应位置准备训练数据\训练数据\融合大表\BTH_Master_Seamless_SpatialRBF_v6.csv"
df.to_csv(output_path, index=False)
print(f"✅ RBF 物理插值完成！完美大表已生成：{output_path}")