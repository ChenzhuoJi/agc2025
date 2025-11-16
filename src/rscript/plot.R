# 安装必要的包（首次运行时需要）
if (!require("tidyverse")) install.packages("tidyverse")
if (!require("patchwork")) install.packages("patchwork")  # 用于组合子图

# 加载包
library(tidyverse)
library(patchwork)  # 方便排列多个子图

# 设置数据目录
data_dir <- "results/comparison"

# 读取目录下所有符合命名规则的CSV文件
csv_files <- list.files(
  path = data_dir,
  pattern = "_comparison.csv$",  # 匹配以"_comparison.csv"结尾的文件
  full.names = TRUE  # 返回完整路径
)

# 批量读取并合并数据
combined_df <- map_dfr(csv_files, function(file) {
  # 读取单个CSV
  df <- read_csv(file, show_col_types = FALSE)
  # 确保第一列名为"metrics"（与你的数据结构匹配）
  colnames(df)[1] <- "metrics"
  # 提取数据集名称（文件名去掉"_comparison.csv"）
  dataname <- str_remove(basename(file), "_comparison.csv")
  # 添加数据集列
  df %>% mutate(dataname = dataname)
})

# 查看合并后的数据
head(combined_df)

# 转换为长格式：方法名作为"method"列，数值作为"value"列
long_df <- combined_df %>%
  pivot_longer(
    cols = -c(metrics, dataname),  # 排除指标和数据集列
    names_to = "method",           # 原列名转为method列
    values_to = "value"            # 数值转为value列
  )

# 查看转换后的数据
head(long_df)

# 获取所有指标（用于生成子图）
metrics <- unique(long_df$metrics)

# 为每个指标创建一个子图，并存储在列表中
plots <- map(metrics, function(metric) {
  # 筛选当前指标的数据
  plot_data <- long_df %>% filter(metrics == metric)
  
  # 绘制柱状图
  ggplot(plot_data, aes(x = dataname, y = value, fill = method)) +
    geom_col(
      position = position_dodge(width = 0.8),  # 并列柱状图（避免重叠）
      width = 0.7  # 柱子宽度
    ) +
    labs(
      y = metric,  # y轴标签为指标名
      x = ""       # x轴标签在总图中统一设置
    ) +
    scale_fill_brewer(palette = "Set3") +  # 配色方案（与Python保持风格一致）
    theme_bw() +  # 白色背景主题
    theme(
      # 网格线设置
      panel.grid.major = element_line(linetype = "dashed", color = "gray80"),
      panel.grid.minor = element_blank(),
      # 去掉顶部和右侧边框
      panel.border = element_rect(color = "black", fill = NA),
      axis.line.x = element_line(color = "black"),
      axis.line.y = element_line(color = "black"),
      # 字体设置（支持中文）
      text = element_text(family = "serif"),  # 英文用衬线字体
      axis.text.x = element_text(angle = 45, hjust = 1, size = 8),  # x轴标签旋转
      axis.text.y = element_text(size = 8),
      axis.title.y = element_text(size = 10),
      # 图例仅在最后一个子图显示
      legend.position = ifelse(metric == last(metrics), "bottom", "none")
    )
})

# 组合所有子图（垂直排列，共享x轴）
combined_plot <- wrap_plots(plots, ncol = 1, guides = "collect") +
  plot_annotation(
    title = "各方法在不同数据集上的性能对比",  # 总标题
    theme = theme(plot.title = element_text(hjust = 0.5, size = 14))  # 标题居中
  ) +
  plot_layout(heights = rep(1, length(metrics)))  # 所有子图高度相同

# 添加底部x轴总标签
combined_plot <- combined_plot & 
  theme(plot.margin = margin(b = 40))  # 底部留空，避免标签被截断
grid::grid.text("数据集", x = 0.5, y = 0.01, gp = grid::gpar(fontsize = 12))  # 总x轴标签

# 保存图片
ggsave(
  filename = "results/comparison_summary_r.png",
  plot = combined_plot,
  width = 12,
  height = 4 * length(metrics),  # 高度随指标数量自适应
  dpi = 300,
  bg = "white"  # 确保背景为白色
)

# 显示图片
print(combined_plot)