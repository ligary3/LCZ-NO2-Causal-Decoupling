import os
import glob
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from scipy.ndimage import gaussian_filter  # 👑 引入高斯滤波核心
import warnings
import tempfile
import gc

warnings.filterwarnings('ignore')

# ==========================================
# 1. 设置路径与全局参数
# ==========================================
lcz_folder = r"E:\lunwen3\BTHLCZ100" 
station_path = r"E:\lunwen3\process\空气质量数据处理\BTH_125_Stations_Coords.csv"
# 输出文件名更新，标记为 Gaussian_Sigma16
output_csv = r"E:\lunwen3\process\空气质量数据处理\根据站点对应位置准备训练数据\训练数据\BTH_LCZ_Gaussian_Sigma16_All_Seasons.csv"

TARGET_CRS = "EPSG:32650"
# 👑 核心参数：根据敏感性分析结果，Sigma 设置为 16 像素
SIGMA_PIXELS = 16.0 

# ==========================================
# 2. 初始化站点坐标 (保持投影后的点，用于采样)
# ==========================================
print(f"⏳ 正在读取站点，并映射至 {TARGET_CRS} 坐标系...")
df_stations = pd.read_csv(station_path)
geometry = [Point(xy) for xy in zip(df_stations['lon'], df_stations['lat'])]
gdf = gpd.GeoDataFrame(df_stations, geometry=geometry, crs="EPSG:4326")
gdf_proj = gdf.to_crs(TARGET_CRS)

# ==========================================
# 3. 智能重投影函数 (保持不变，确保输入数据是“米”)
# ==========================================
def reproject_raster_to_memory(src_path, dst_crs):
    with rasterio.open(src_path) as src:
        if src.crs.to_string() == dst_crs:
            return src_path
        print(f"   ⚠️ 重投影: {os.path.basename(src_path)} -> {dst_crs}")
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({'crs': dst_crs, 'transform': transform, 'width': width, 'height': height})
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.tif')
        temp_path = temp_file.name
        temp_file.close()
        with rasterio.open(temp_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(source=rasterio.band(src, i), destination=rasterio.band(dst, i),
                          src_transform=src.transform, src_crs=src.crs,
                          dst_transform=transform, dst_crs=dst_crs, resampling=Resampling.nearest)
        return temp_path

# ==========================================
# 4. 自动化遍历并执行高斯加权提取 (带严格安检)
# ==========================================
all_tifs = glob.glob(os.path.join(lcz_folder, "*.tif"))
tif_files = []

for f in all_tifs:
    basename = os.path.basename(f)
    # 规则1：必须严格以 "100m.tif" 结尾，把带 EPSG 的中间文件全部踢掉
    if not basename.endswith("100m.tif"):
        continue
    # 规则2：必须以四个季节单词开头
    if not any(basename.startswith(s) for s in ["Spring", "Summer", "Autumn", "Winter"]):
        continue
    tif_files.append(f)

print(f"📂 经过严格安检，最终锁定 {len(tif_files)} 张纯净原始影像！")
print(f"🚀 启动 Sigma=16 高斯加权提取引擎...")

all_results = []

for tif_path in tif_files:
    filename = os.path.basename(tif_path)
    try:
        parts = filename.replace('.tif', '').split('_')
        season, year = parts[0], parts[3]
    except: continue

    print(f"▶ 处理中: {filename}")
    safe_tif_path = reproject_raster_to_memory(tif_path, TARGET_CRS)
    
    with rasterio.open(safe_tif_path) as src:
        lcz_data = src.read(1).astype(np.float32)
        # 1. 确定站点在栅格中的行列号索引
        coords = [(x, y) for x, y in zip(gdf_proj.geometry.x, gdf_proj.geometry.y)]
        rows, cols = [], []
        for x, y in coords:
            r, c = src.index(x, y)
            rows.append(r); cols.append(c)
        
        # 2. 👑 计算“总有效建筑权重图” (分母)
        # 只要类别在 1-17 之间，就是有效陆地建筑像素
        valid_mask = ((lcz_data >= 1) & (lcz_data <= 17)).astype(np.float32)
        denom_map = gaussian_filter(valid_mask, sigma=SIGMA_PIXELS, mode='constant', cval=0.0)
        # 采样站点位置的分母权重值
        denom_sampled = denom_map[rows, cols]
        
        # 3. 👑 循环 1-17 类，计算各自的加权占比 (分子)
        df_block = pd.DataFrame()
        for i in range(1, 18):
            class_mask = (lcz_data == i).astype(np.float32)
            # 对特定类别进行高斯平滑
            num_map = gaussian_filter(class_mask, sigma=SIGMA_PIXELS, mode='constant', cval=0.0)
            num_sampled = num_map[rows, cols]
            
            # 计算占比：(该类加权值 / 总有效加权值) * 100
            # 使用 1e-9 防止分母为 0
            df_block[f'LCZ_{i}'] = (num_sampled / (denom_sampled + 1e-9)) * 100.0
            
            del class_mask, num_map
            gc.collect()

    if safe_tif_path != tif_path: os.remove(safe_tif_path)
    
    df_block['station'] = gdf_proj['station'].values
    df_block['year'] = year
    df_block['season'] = season
    all_results.append(df_block)
    
    del lcz_data, valid_mask, denom_map
    gc.collect()

# ==========================================
# 5. 合并并导出
# ==========================================
print("🔄 正在生成 V7 高斯权重版训练大表...")
final_df = pd.concat(all_results, ignore_index=True)
cols = ['station', 'year', 'season'] + [f'LCZ_{i}' for i in range(1, 18)]
final_df = final_df[cols]
final_df.to_csv(output_csv, index=False)

print(f"🎉 任务完成！")
print(f"👉 物理参数: Gaussian Sigma = 16 (等效 4.8km 衰减边界)")
print(f"👉 结果路径: {output_csv}")