# ==========================================
# 0. 环境准备与全局字号控制
# ==========================================
packages <- c("tidyverse", "ggraph", "tidygraph")
installed_packages <- packages %in% rownames(installed.packages())
if (any(installed_packages == FALSE)) {
  install.packages(packages[!installed_packages])
}

library(tidyverse)
library(ggraph)
library(tidygraph)

# 👑 核心控制台：全局字号放大系数 (你可以随时修改这个数字)
num <- 4 

print("🚀 启动机制图谱无字极简引擎：节点动态膨胀 + 宽幅防溢出边界...")

# ==========================================
# 1. 读取数据与预处理
# ==========================================
data_path <- "E:/lunwen3/process/空气质量数据处理/根据站点对应位置准备训练数据/训练数据/融合大表/BTH_Master_Seamless_SpatialRBF_v7.csv"
df <- read.csv(data_path)

features <- c('t2m_c', 'ERA5_RH', 'WS', 'LCZ_1', 'LCZ_4', 'LCZ_7', 'LCZ_9', 'LCZ_10', 'LCZ_11', 'NO2')
df_clean <- df %>% select(all_of(features)) %>% drop_na() %>% sample_n(8000)
df_std <- as.data.frame(scale(df_clean))

# ==========================================
# 2. 构建先验因果矩阵 (SEM 物理机制)
# ==========================================
edges <- tribble(
  ~from, ~to,
  # 1. 气象中介路径 (LCZ -> 微气象)
  'LCZ_1', 'WS',  'LCZ_1', 't2m_c',
  'LCZ_4', 'WS',  'LCZ_4', 't2m_c',
  'LCZ_7', 'WS',  'LCZ_7', 't2m_c',
  'LCZ_9', 'WS',  'LCZ_9', 't2m_c',
  'LCZ_10', 'WS', 'LCZ_10', 't2m_c',
  'LCZ_11', 'WS', 'LCZ_11', 't2m_c',
  
  # 2. 气象内部耦合
  't2m_c', 'ERA5_RH', 'WS', 'ERA5_RH',
  
  # 3. 气象 -> 污染响应
  't2m_c', 'NO2', 'WS', 'NO2', 'ERA5_RH', 'NO2',
  
  # 4. 👑 核心修复：直接排放/吸收的物理路径 (LCZ -> NO2)
  # 必须全部允许，让数据自己决定谁的直接效应强！
  'LCZ_1', 'NO2',
  'LCZ_4', 'NO2',
  'LCZ_7', 'NO2',
  'LCZ_9', 'NO2',
  'LCZ_10', 'NO2',
  'LCZ_11', 'NO2'
)
# ==========================================
# 3. 计算多元线性回归权重
# ==========================================
edges$weight <- 0
targets <- unique(edges$to)

for (tgt in targets) {
  parents <- edges$from[edges$to == tgt]
  formula_str <- paste(tgt, "~", paste(parents, collapse = " + "))
  model <- lm(as.formula(formula_str), data = df_std)
  coefs <- coef(model)
  for (p in parents) {
    edges$weight[edges$from == p & edges$to == tgt] <- coefs[p]
  }
}

edges_active <- edges %>% 
  filter(abs(weight) >= 0.03) %>%
  mutate(edge_col = ifelse(weight > 0, "#D55E00", "#0072B2")) 

# ==========================================
# 4. 节点重命名与绝对坐标布局 (NO₂ 下标保留)
# ==========================================
rename_map <- c(
  't2m_c' = 'T2m', 'ERA5_RH' = 'RH', 'WS' = 'WS',
  'LCZ_1' = 'LCZ 1', 'LCZ_4' = 'LCZ 4', 'LCZ_7' = 'LCZ 7',
  'LCZ_9' = 'LCZ 9', 'LCZ_10' = 'LCZ 10', 'LCZ_11' = 'LCZ A',
  'NO2' = 'NO₂'
)
edges_active$from <- rename_map[edges_active$from]
edges_active$to <- rename_map[edges_active$to]

nodes <- data.frame(name = unique(c(edges_active$from, edges_active$to)))
nodes <- nodes %>%
  mutate(
    layer = case_when(
      grepl("LCZ", name) ~ "I. Urban Morphology",
      name %in% c("T2m", "WS", "RH") ~ "II. Meteorological Mediation",
      name == "NO₂" ~ "III. Air Pollution Response"
    ),
    y = case_when(
      layer == "I. Urban Morphology" ~ 2,
      layer == "II. Meteorological Mediation" ~ 1,
      layer == "III. Air Pollution Response" ~ 0
    ),
    x = case_when(
      name == "LCZ 1" ~ -2.5,
      name == "LCZ 10" ~ -1.5,
      name == "LCZ 4" ~ -0.5,
      name == "LCZ 7" ~ 0.5,
      name == "LCZ 9" ~ 1.5,
      name == "LCZ A" ~ 2.5,
      name == "T2m" ~ -1.5,
      name == "WS" ~ 0,
      name == "RH" ~ 1.5,
      name == "NO₂" ~ 0
    ),
    # 👑 动态节点大小：基础大小 + (num * 1.5) 实现自适应膨胀
    node_size = case_when(
      layer == "I. Urban Morphology" ~ 32 + (num * 1.5),  
      layer == "II. Meteorological Mediation" ~ 22 + (num * 1.5), 
      layer == "III. Air Pollution Response" ~ 26 + (num * 1.5)   
    )
  )

# ==========================================
# 5. 图层物理分身法渲染 (极简无字线条版)
# ==========================================
graph <- tbl_graph(nodes = nodes, edges = edges_active)

p <- ggraph(graph, layout = "manual", x = x, y = y) +
  # 图层 1: 莫兰迪底色背景与侧边栏文字 (👑 右边界 xmax 放宽至 4.2 防截断)
  annotate("rect", xmin = -3.5, xmax = 4.2, ymin = 1.6, ymax = 2.4, alpha = 0.5, fill = "#F1F8E9") +
  annotate("rect", xmin = -3.5, xmax = 4.2, ymin = 0.6, ymax = 1.4, alpha = 0.5, fill = "#E3F2FD") +
  annotate("rect", xmin = -3.5, xmax = 4.2, ymin = -0.5, ymax = 0.4, alpha = 0.5, fill = "#FFEBEE") +
  annotate("text", x = -3.4, y = 2.25, label = "I. Urban Morphology", fontface = "bold", color = "#2E7D32", hjust = 0, size = 6 + num) +
  annotate("text", x = -3.4, y = 1.25, label = "II. Meteorological Mediation", fontface = "bold", color = "#1565C0", hjust = 0, size = 6 + num) +
  annotate("text", x = -3.4, y = 0.25, label = "III. Air Pollution Response", fontface = "bold", color = "#C62828", hjust = 0, size = 6 + num) +

  # 图层 2: 纯彩色线条 (已移除文字注释，靠粗细说话)
  geom_edge_arc(aes(edge_width = abs(weight), color = edge_col),
                strength = 0.2, 
                arrow = arrow(length = unit(5, 'mm'), type = "closed"),
                end_cap = circle(16, 'mm'), 
                start_cap = circle(18, 'mm')) +
  scale_edge_width(range = c(0.8, 3.5)) + 
  scale_edge_color_identity() +               

  # 图层 3 & 4: 节点大圈与内部文字
  geom_node_point(aes(fill = layer, size = node_size), shape = 21, color = "black", stroke = 1.2) +
  scale_size_identity() +                     
  scale_fill_manual(values = c("I. Urban Morphology" = "#6ACC64",
                               "II. Meteorological Mediation" = "#4DBBD5",
                               "III. Air Pollution Response" = "#E64B35")) +
  geom_node_text(aes(label = name), fontface = "bold", size = 5.5 + num) +
  
  # -----------------------------------
  # 图层 5: 完美嵌合至宽版底图的加长图例 (👑 图例框 xmax 拓宽至 4.0)
  # -----------------------------------
  annotate("rect", xmin = 1.15, xmax = 4.3, ymin = -0.48, ymax = 0.38, fill = "white", color = "#333333", linewidth = 0.8) +
  annotate("text", x = 1.25, y = 0.28, label = "Standardized Coefficients (\u03b2)", fontface = "bold", size = 5.5 + num, hjust = 0) +
  
  annotate("segment", x = 1.25, xend = 1.45, y = 0.17, yend = 0.17, color = "#D55E00", linewidth = 2.5) +
  annotate("text", x = 1.55, y = 0.17, label = "Positive Pathway (\u03b2 > 0)", size = 5 + num, hjust = 0) +
  
  annotate("segment", x = 1.25, xend = 1.45, y = 0.06, yend = 0.06, color = "#0072B2", linewidth = 2.5) +
  annotate("text", x = 1.55, y = 0.06, label = "Negative Pathway (\u03b2 < 0)", size = 5 + num, hjust = 0) +
  
  annotate("segment", x = 1.25, xend = 1.45, y = -0.05, yend = -0.05, color = "#7F8C8D", linewidth = 1.5) +
  annotate("text", x = 1.55, y = -0.05, label = "Line width indicates |\u03b2| magnitude", size = 5 + num, hjust = 0) +
  
  annotate("point", x = 1.35, y = -0.16, size = 6 + (num*0.5), shape = 21, fill = "#6ACC64", color = "black") +
  annotate("text", x = 1.55, y = -0.16, label = "Urban Morphology (LCZ)", size = 5 + num, hjust = 0) +
  
  annotate("point", x = 1.35, y = -0.27, size = 6 + (num*0.5), shape = 21, fill = "#4DBBD5", color = "black") +
  annotate("text", x = 1.55, y = -0.27, label = "Meteorological Mediation", size = 5 + num, hjust = 0) +
  
  annotate("point", x = 1.35, y = -0.38, size = 6 + (num*0.5), shape = 21, fill = "#E64B35", color = "black") +
  annotate("text", x = 1.55, y = -0.38, label = "Air Pollution Response", size = 5 + num, hjust = 0) +

  theme_void() +
  theme(legend.position = "none")

# ==========================================
# 6. 导出最终神图
# ==========================================
ggsave("C:/Users/12169/Desktop/论文3/result/因果/SCI_Figure_DAG_Final.png", p, width = 17, height = 11, dpi = 600, bg="white")
print("🎉 出图完毕！全局字号自适应放大已就位！")