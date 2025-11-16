import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
def plot_comparison(
    data_dir="results/comparison",
    legend_loc="upper center",
    legend_bbox_to_anchor=(0.5, -0.3),  # 图例位置参数（配合loc使用）
    metrics_to_plot="all",  # 可指定指标列表，如["ACC", "JC"]
):
    """
    绘制不同方法在各数据集上的性能对比图

    参数:
        data_dir: 存放CSV文件的目录路径
        legend_loc: 图例位置（matplotlib legend的loc参数）
        legend_bbox_to_anchor: 图例偏移位置（元组，如(0.5, 1.2)表示上方居中）
        metrics_to_plot: 要绘制的指标，"all"表示所有指标，或指定列表如["ACC", "JC"]
    """
    # ----------------------
    # 1. 读取并合并数据
    # ----------------------
    all_data = []
    for filename in os.listdir(data_dir):
        if filename.endswith("_comparison.csv"):
            dataname = filename.replace("_comparison.csv", "")
            df = pd.read_csv(os.path.join(data_dir, filename))
            # 确保第一列名为'metrics'
            if df.columns[0] != "metrics":
                df = df.rename(columns={df.columns[0]: "metrics"})
            df["dataname"] = dataname
            all_data.append(df)

    if not all_data:
        raise ValueError(f"在{data_dir}中未找到符合条件的CSV文件")

    combined_df = pd.concat(all_data, ignore_index=True)

    # ----------------------
    # 2. 筛选要绘制的指标
    # ----------------------
    all_metrics = combined_df["metrics"].unique()
    if metrics_to_plot == "all":
        metrics = all_metrics
    else:
        # 检查指定的指标是否存在
        invalid_metrics = [m for m in metrics_to_plot if m not in all_metrics]
        if invalid_metrics:
            raise ValueError(f"无效的指标：{invalid_metrics}，可用指标：{all_metrics}")
        metrics = metrics_to_plot
    n_metrics = len(metrics)

    # ----------------------
    # 3. 配置绘图参数
    # ----------------------
    plt.rcParams["font.family"] = ["Times New Roman", "SimHei"]
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False

    methods = [col for col in combined_df.columns if col not in ["metrics", "dataname"]]
    datasets = combined_df["dataname"].unique()

    # 颜色配置（master单独设置）
    master_color = "#e90303"  # 深红色
    other_methods = [m for m in methods if m != "master"]
    other_colors = plt.cm.Set3(np.linspace(0, 1, len(other_methods)))
    colors = [master_color] + list(other_colors)

    bar_width = 0.15

    # ----------------------
    # 4. 创建子图并绘图
    # ----------------------
    fig, axes = plt.subplots(
        nrows=n_metrics, ncols=1, figsize=(12, 2.5 * n_metrics), sharex=True
    )
    # 处理单指标情况（确保axes是列表格式）
    if n_metrics == 1:
        axes = [axes]

    # 遍历每个指标绘制子图
    for i, metric in enumerate(metrics):
        ax = axes[i]
        metric_data = combined_df[combined_df["metrics"] == metric]
        x = np.arange(len(datasets))

        for j, method in enumerate(methods):
            values = [
                metric_data[metric_data["dataname"] == ds][method].values[0]
                for ds in datasets
            ]
            ax.bar(
                x + j * bar_width,
                values,
                width=bar_width,
                color=colors[j],
                edgecolor="black",
                linewidth=0.7,
                label=method  # 所有子图的柱子都添加 label
            )

        # 子图配置
        ax.set_ylabel(metric, fontsize=10)
        ax.set_xticks(x + bar_width * (len(methods) - 1) / 2)
        ax.set_xticklabels(datasets, rotation=45, ha="right")
        ax.tick_params(axis="y", labelsize=8)

    # ----------------------
    # 5. 添加图例
    # ----------------------
    axes[0].legend(
        loc=legend_loc,
        bbox_to_anchor=legend_bbox_to_anchor,
        ncol=len(methods),
        fontsize=9,
    )

    # ----------------------
    # 6. 保存并显示
    # ----------------------
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.12, hspace=0.4)
    output_path = os.path.join("results", "comparison_summary.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    print(f"汇总图已保存至：{output_path}")
    plt.show()


if __name__ == "__main__":
    # 示例1：默认参数（所有指标，图例在最后一个子图下方）
    # plot_comparison()

    # 示例2：指定图例在上方居中
    plot_comparison(
        legend_loc="upper center",
        legend_bbox_to_anchor=(0.5, 1.2),  # 上方居中偏移
        metrics_to_plot=["ACC","JC", "NMI"],  # ["ACC", "NMI"]  # 仅绘制这两个指标
    )
