import matplotlib.pyplot as plt
import logging
import numpy as np

def print_to_log(*args, **kwargs):
    print(*args, **kwargs)  # 控制台输出
    logging.info(" ".join(map(str, args)))


def Movementfig(transients,Filtered_Position,velocity,is_running,slice_no,output_path,epochs,lick,position):
    # --- 使用 Matplotlib 绘制三Y轴图表 ---

    # 1. 创建图形和左侧的主坐标轴 (ax1)
    fig, ax1 = plt.subplots(figsize=(15, 6)) # 稍微增加高度以容纳更多轴标签

    # --- 轴1 (左侧): 绘制位置 ---
    color_position = 'green'
    ax1.plot(transients.times(), Filtered_Position.values, 
            color=color_position, linewidth=2, label='Location on track') # 加粗一点以便观察
    ax1.set_xlabel('Time (s)', fontsize=16)
    ax1.set_ylabel('Location on track', fontsize=16, color=color_position)
    ax1.tick_params(axis='y', labelcolor=color_position)
    ax1.grid(True, linestyle='--', alpha=0.6) # 主网格线


    # --- 轴2 (右侧): 绘制速度 (Velocity) ---
    ax2 = ax1.twinx() # 创建第一个右侧轴
    color_velocity = 'firebrick'
    ax2.plot(transients.times(), velocity, 
            color=color_velocity, linewidth=0.5, label='Velocity', alpha = 1)
    ax2.set_ylabel('Velocity (cm/s)', fontsize=16, color=color_velocity)
    ax2.tick_params(axis='y', labelcolor=color_velocity)
    ax2.set_ylim(0, 100) # 按要求设置范围 0-40


    # --- 轴3 (最右侧): 绘制跑步时段的填充区域 ---
    ax3 = ax1.twinx() # 再创建一个右侧轴
    # 关键步骤: 将第三个轴的脊柱 (spine) 向右移动，以防止与 ax2 重叠
    # 'outward' 表示向外，60 是移动的点数 (points)，可以根据需要调整
    ax3.spines['right'].set_position(('outward', 100))

    # 准备填充数据 (和上次一样)
    y_fill_height = 400 # 假设位置Y轴最大为400
    fill_values = np.zeros_like(transients.times(), dtype=float)
    fill_values[is_running] = y_fill_height

    # 在第三个轴上绘制填充区域
    fill_color_running = (0/255, 100/255, 200/255, 0.25)
    ax3.fill_between(transients.times(), fill_values, 0,
                    facecolor=fill_color_running, label='Running period', zorder=0) # zorder=0 确保它在最底层
    ax3.set_ylim(0, y_fill_height) # 确保填充区域的轴范围正确
    ax3.set_yticks([]) # 这个轴只是背景，不需要刻度


    # --- 统一管理三个图例 ---
    # 从三个轴分别获取 "handles" 和 "labels"
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    h3, l3 = ax3.get_legend_handles_labels()

    # 合并并创建图例
    fig.legend(handles=h1 + h2 + h3, labels=l1 + l2 + l3, 
            loc='upper center', bbox_to_anchor=(0.5, 1.0),
            ncol=3, fancybox=True, fontsize=16)

    # 调整布局以防止图例或标签被裁剪
    fig.tight_layout(rect=[0, 0.05, 1, 0.9]) # 调整rect为图例和底部留出空间
    plt.show()


    fig, ax = plt.subplots(figsize=(8, 6))
    for n in range(len(epochs)-1, 0, -1):
        lick_times = lick.restrict(epochs[n]).times()
        position_times = position.restrict(epochs[n]).times()
        position_values = position.restrict(epochs[n]).values
        lick_positions = np.interp(lick_times, position_times, position_values)
        ax.plot(lick_positions, np.full_like(lick_positions, n),
                '.', markersize=5, color='#6FA3C6', alpha=0.3)
    ax.invert_yaxis()
    ax.set_xlabel('Position')
    ax.set_ylabel('Trial')

    plt.show()
    
    
def compute_distribution(data,xlabel,savepath,xlim_left=-1.1,xlim_right=1.1,ylim_up=1.1,ylim_down=0,write_sigma=True,accumulate = True):
    data:np.ndarray
    xlabel:str
    # 确保data是一维ndarray（如果是二维，先展平）
    data = data.ravel()  # 展平为一维
    sorted_data = np.sort(data)  # 排序
    cdf = np.linspace(1/len(sorted_data), 1, len(sorted_data))  # 累计概率
    mu = np.mean(data)
    sigma = np.std(data)

    plt.figure(figsize=(8, 5))
    # 核心：绘制直方图（bins=30是经验值，可调整；density=True显示概率密度）
    plt.hist(
        data, 
        bins=30,                # 分箱数，越大越精细
        color='skyblue',        # 主体颜色
        alpha=0.7,              # 透明度
        edgecolor='white',      # 边框色，提升清晰度
    )
    if write_sigma:
        plt.axvline(x=mu - 3*sigma, color='red', linestyle='--', alpha=0.6, label='μ - 3σ')
        plt.axvline(x=mu - 2*sigma, color='orange', linestyle='--', alpha=0.7, label='μ - 2σ')
        plt.axvline(x=mu - 1*sigma, color='green', linestyle='--', alpha=0.8, label='μ - 1σ')
        plt.legend(loc='upper right', fontsize=9)  # 图例位置
    plt.xlim(xlim_left,xlim_right)       # x轴限定-1~1（相关系数范围）
    plt.xlabel(xlabel)  # x轴标签（更精准的英文）
    plt.ylabel('Cell number')       # y轴标签（概率密度）
    plt.title(f'Histogram of {xlabel}')  # 标题

    plt.grid(axis='y', alpha=0.2)  # 仅显示y轴网格，更清晰
    plt.savefig(
    savepath / f"{xlabel}.png",  # 保存路径/名字
    dpi=300,             # 高清 300dpi
    pad_inches=0.1       # 内边距
    )
    plt.show()
    if accumulate:
        # --------------- 绘制累计分布图（确保坐标轴适配）---------------
        plt.figure(figsize=(8, 5))
        plt.plot(sorted_data, cdf, color='blue', linewidth=1.5)
        if write_sigma:
            plt.axvline(x=mu - 3*sigma, color='red', linestyle='--', alpha=0.3, label='3σ')
            plt.axvline(x=mu - 2*sigma, color='orange', linestyle='--', alpha=0.5, label='2σ')
            plt.axvline(x=mu - 1*sigma, color='green', linestyle='--', alpha=0.7, label='1σ')
        plt.xlim(xlim_left,xlim_right)
        plt.ylim(0, 1)
        plt.xlabel(xlabel)
        plt.ylabel('Probability')
        plt.legend(loc='lower right', frameon=True, fontsize=10)  # 指定位置+显示边框
        plt.grid(alpha=0.2)  # 可选：添加网格更易读
        plt.savefig(
        savepath / f"{xlabel}_accumulate.png",  # 保存路径/名字
        dpi=300,             # 高清 300dpi
        pad_inches=0.1       # 内边距
        )
        plt.show()