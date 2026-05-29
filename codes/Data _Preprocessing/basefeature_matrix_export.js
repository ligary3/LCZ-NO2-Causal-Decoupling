// ==========================================
// 1. 定义目标日期和京津冀精确边界
// ==========================================
var targetDate = '2024-07-18'; //0111
var bth_geom = ee.FeatureCollection('projects/ee-l2892786691/assets/JJJ/jjj').geometry();

// 地形降尺度函数
var mean100m = function(img, native_scale) {
  return img.setDefaultProjection({crs: 'EPSG:4326', scale: native_scale})
            .reduceResolution({reducer: ee.Reducer.mean(), maxPixels: 1024})
            .reproject({crs: 'EPSG:32650', scale: 100}); 
};

// ==========================================
// 2. 气象强迫场
// ==========================================
var era5_atm = ee.ImageCollection("ECMWF/ERA5/HOURLY")
  .filterDate(targetDate, ee.Date(targetDate).advance(1, 'day'))
  .filter(ee.Filter.calendarRange(6, 6, 'hour')).first().resample('bilinear'); 

var era5_land = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")
  .filterDate(targetDate, ee.Date(targetDate).advance(1, 'day'))
  .filter(ee.Filter.calendarRange(6, 6, 'hour')).first().resample('bilinear'); 

var u10 = era5_land.select('u_component_of_wind_10m').unmask(era5_atm.select('u_component_of_wind_10m'));
var v10 = era5_land.select('v_component_of_wind_10m').unmask(era5_atm.select('v_component_of_wind_10m'));
var ws = u10.pow(2).add(v10.pow(2)).sqrt().rename('WS');
var wd_sin = u10.divide(ws).rename('WD_sin'); 
var wd_cos = v10.divide(ws).rename('WD_cos');

var t2m_c = era5_land.select('temperature_2m').unmask(era5_atm.select('temperature_2m')).subtract(273.15).rename('t2m_c');
var sp = era5_land.select('surface_pressure').unmask(era5_atm.select('surface_pressure')).rename('surface_pressure');

var td = era5_land.select('dewpoint_temperature_2m').unmask(era5_atm.select('dewpoint_temperature_2m')).subtract(273.15);
var rh = ee.Image().expression(
  '100 * (exp((17.625 * Td)/(243.04 + Td)) / exp((17.625 * T)/(243.04 + T)))', 
  {'T': t2m_c, 'Td': td}
).rename('ERA5_RH');

var blh = era5_atm.select('boundary_layer_height').rename('ERA5_BLH');
var ssr_value = era5_atm.select('surface_solar_radiation_downwards').divide(1e6).rename('ssr_value');

// ==========================================
// 3. 卫星底板：🚀极简纯净版 (专为两阶段机器学习定制)🚀
// ==========================================
var tropomi_raw = ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_NO2")
  .filterDate(targetDate, ee.Date(targetDate).advance(1, 'day'))
  .select('tropospheric_NO2_column_number_density')
  .mean();

// 🚨 核心改动：只用 7km 的极小核，轻微修复 TROPOMI 传感器本身的轨道扫描线缝隙
// 绝对不修补真实的云洞！让厚云区域保持为缺失状态，交由下游 Python 的 Stage 1 引擎缝合。
var g_small = tropomi_raw.focal_mean({radius: 7000, units: 'meters', kernelType: 'gaussian'});

// 仅用小核填补极其细微的缝隙，然后执行双三次重采样平滑像素边缘
var tropomi_no2 = tropomi_raw.unmask(g_small)
  .updateMask(tropomi_raw.mask().or(g_small.mask())) // 确保大块云洞依然是透明/缺失状态
  .resample('bicubic')
  .rename('TROPOMI_NO2');
// ==========================================
// 4. 静态特征 & 时间特征
// ==========================================
var ntl = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG')
  .filterDate('2024-01-01', '2025-01-01').select('avg_rad').mean()
  .resample('bilinear').rename('NTL'); 

var dem = mean100m(ee.Image('USGS/SRTMGL1_003').select('elevation'), 30).rename('DEM');
var dsm = mean100m(ee.ImageCollection('JAXA/ALOS/AW3D30/V4_1').mosaic().select('DSM'), 30).rename('DSM');

var target_ee_date = ee.Date(targetDate);
var monthImg = ee.Image.constant(target_ee_date.get('month')).rename('Month');
var doyImg = ee.Image.constant(target_ee_date.getRelative('day', 'year').add(1)).rename('DayOfYear'); 
var lonLat = ee.Image.pixelLonLat(); 

// ==========================================
// 5. 打包导出 (共 16 波段: 2坐标 + 14特征)
// ==========================================
var inputImage = ee.Image.cat([
  lonLat, ws, wd_sin, wd_cos, t2m_c, sp, rh, blh, ssr_value, 
  tropomi_no2, ntl, dem, dsm, monthImg, doyImg
]).clip(bth_geom).toFloat();

var export_name = 'BTH_BaseFeatures_' + targetDate.replace(/-/g, '') + '_100m';

Export.image.toDrive({
  image: inputImage,
  description: export_name,
  folder: 'GEE_AirQuality_Export',
  fileNamePrefix: export_name,
  region: bth_geom,
  scale: 100, 
  crs: 'EPSG:32650', 
  maxPixels: 1e13
});