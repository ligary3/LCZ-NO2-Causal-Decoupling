import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from scipy.ndimage import gaussian_filter
import os
import warnings
warnings.filterwarnings('ignore')

# ================= 1. 配置文件路径 =================
master_csv_path = r"E:\lunwen3\process\空气质量数据处理\根据站点对应位置准备训练数据\训练数据\融合大表\BTH_Final_Master_Dataset.csv"
coords_csv_path = r"E:\lunwen3\process\空气质量数据处理\BTH_125_Stations_Coords.csv"
road_dir = r"E:\lunwen3\路网\BTH"
output_csv_path = r"E:\lunwen3\process\空气质量数据处理\根据站点对应位置准备训练数据\训练数据\融合大表\BTH_Final_Master_Dataset_MultiRoads_Gaussian.csv"

# 👑 核心新增：LCZ 模板图路径 (确保你的高斯路网和 LCZ 像素 100% 完美对齐)
template_tif_path = r"E:\lunwen3\result\敏感性分析\Winter_LCZ_BTH_2024_100m_EPSG32650.tif"
# 存放生成的高斯 TIF 的文件夹
output_tif_dir = r"E:\lunwen3\路网\Gaussian_TIFs"
os.makedirs(output_tif_dir, exist_ok=True)

# ================= 2. 加载站点与模板 =================
print("正在加载站点与空间模板...")
df_coords = pd.read_csv(coords_csv_path)
gdf_stations = gpd.GeoDataFrame(df_coords, geometry=gpd.points_from_xy(df_coords.lon, df_coords.lat), crs="EPSG:4326")
gdf_stations = gdf_stations.to_crs("EPSG:32650") # 转为米制

# 读取模板的元数据
with rasterio.open(template_tif_path) as src:
    meta = src.meta.copy()
    transform = src.transform
    width = src.width
    height = src.height
    
# 计算站点在栅格中的行列号 (用于极速提取)
stations_x = gdf_stations.geometry.x.values
stations_y = gdf_stations.geometry.y.values
rows, cols = rasterio.transform.rowcol(transform, stations_x, stations_y)

# 定义高斯核的物理尺度 (米) -> 转化为像素尺度 (除以 100m)
# 例如 300m 对应 sigma=3 个像素
sigmas_m = [300, 1000, 3000]
yearly_road_features = []

# ================= 3. 年际循环生成高斯特征 =================
for year in range(2018, 2025):
    road_shp_path = os.path.join(road_dir, f"{year}road.shp")
    if not os.path.exists(road_shp_path): continue
        
    print(f"\n🚀 正在处理 {year} 年的路网数据...")
    gdf_roads = gpd.read_file(road_shp_path).to_crs("EPSG:32650")
    
    # 基础 DataFrame
    df_year_features = pd.DataFrame({'station': gdf_stations['station'], 'year': year})
    
    # --- 1. 计算距最近道路距离 (保留你的优秀微观特征) ---
    print("   -> 计算 Dist_to_Road...")
    nearest = gpd.sjoin_nearest(gdf_stations, gdf_roads, how='left', distance_col='Dist_to_Road')
    nearest = nearest.drop_duplicates(subset='station')[['station', 'Dist_to_Road']]
    df_year_features = pd.merge(df_year_features, nearest, on='station', how='left')

    # --- 2. 路网栅格化 (Rasterize) ---
    print("   -> 将矢量路网栅格化至 100m 模板...")
    # 把有路的地方烧入 1，没路的地方是 0
    road_shapes = ((geom, 1) for geom in gdf_roads.geometry)
    base_raster = rasterize(
        shapes=road_shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.float32
    )
    
    # --- 3. 高斯卷积平滑与站点提取 ---
    for sigma_m in sigmas_m:
        sigma_pixel = sigma_m / 100.0  # 物理尺度转像素尺度
        col_name = f'Road_Gaussian_{sigma_m}m'
        print(f"   -> 应用高斯平滑核 (Sigma={sigma_m}m)...")
        
        # 👑 核心魔法：高斯滤波模拟污染扩散
        smoothed_raster = gaussian_filter(base_raster, sigma=sigma_pixel)
        
        # 极速提取站点所在像素的值
        extracted_values = smoothed_raster[rows, cols]
        df_year_features[col_name] = extracted_values
        
        # 将生成的 TIF 保存下来，预测出图时直接用！
        tif_out_path = os.path.join(output_tif_dir, f"Road_Gaussian_{year}_{sigma_m}m.tif")
        meta.update(dtype=rasterio.float32)
        with rasterio.open(tif_out_path, 'w', **meta) as dst:
            dst.write(smoothed_raster.astype(rasterio.float32), 1)

    yearly_road_features.append(df_year_features)

# ================= 4. 融合进大表 =================
print("\n🔄 正在融合进主大表...")
df_all_road_features = pd.concat(yearly_road_features, ignore_index=True)
df_master = pd.read_csv(master_csv_path)

# 如果你的大表里有 'date' 没提取出 'year'，在这里安全提取
if 'year' not in df_master.columns:
    df_master['year'] = pd.to_datetime(df_master['date']).dt.year

df_final = pd.merge(df_master, df_all_road_features, on=['station', 'year'], how='left')

# 清理并保存
df_final.to_csv(output_csv_path, index=False)
print(f"🎉 大功告成！包含多尺度高斯核特征的新大表已保存至: {output_csv_path}")
print(f"📁 预测用高斯 TIF 已保存至: {output_tif_dir}")