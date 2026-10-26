import numpy as np
from scipy.ndimage import label

def place_field_identification(
    neuron_idx,
    tuning_curves,
    transients,
    Filtered_Position,
    FilterField_thresholdRatio,
    FilterMean_threshold,
    FilterTime_thresholdRatio,
    epochs,
    bin_indices,
    frame_rate=30.0,  # 默认帧率，可根据实际情况调整
    num_frames=36030
):
    """
    潜在位置细胞识别

    参数：
    - neuron_idx (int): 神经元的索引。
    - tuning_curves (pd.DataFrame 或 np.ndarray): 所有神经元的平滑调谐曲线，形状为 (神经元数, 位置数)。
    - transients (pd.DataFrame 或 np.ndarray): 瞬时响应数据，维度为 (时间, 神经元)。
    - epochs (dict): epoch 字典，每个 epoch 包含 'start' 和 'end' 时间（以秒为单位）。
    - Filtered_Position (object): 具有 `values` 属性的过滤位置对象，形状为 (时间,)。
    - identify_place_fields (function): 用于识别位置场的函数，接受 `tuning_curves` 和 `neuron_idx` 作为参数，返回布尔数组。
    - snr_ratio (float): 信噪比阈值比例，默认 7。
    - track_traversed_threshold_ratio (float): 遍历次数的阈值比例，默认 0.33。
    - frame_rate (float): 帧率，用于将时间转换为帧索引，默认 1.0。
    - num_frames (int, 可选): 总帧数。如果未提供，将从 `Filtered_Position` 中推断。

    返回：
    - bool: 如果神经元通过所有过滤步骤，返回 `True`；否则返回 `False`。
    """
    result = []
    
    # 第一步：识别潜在位置场
    tc = tuning_curves[neuron_idx]
    potential_place_fields = identify_place_fields(tuning_curves, neuron_idx=neuron_idx, threshold_ratio=FilterField_thresholdRatio)
    
    # 如果没有符合条件的位置场，返回 False
    if not np.any(potential_place_fields):
        return False


    masks = potential_place_fields 
    # extract each continuous field-range from the boolean mask
    field_ranges = []
    in_field = False
    for pos, val in enumerate(masks):
        if val and not in_field:
            start = pos; in_field = True
        elif not val and in_field:
            end = pos; in_field = False
            field_ranges.append((start, end))
    if in_field:
        field_ranges.append((start, len(masks)))

    all_field_mask = np.zeros_like(tc, dtype=bool)
    for start_bin, end_bin in field_ranges:
        all_field_mask[start_bin:end_bin] = True
    out_data = tc.iloc[~all_field_mask]


    epochs_frames = [
    (
        max(int(np.floor(start * frame_rate)), 0),
        min(int(np.ceil(end * frame_rate)), num_frames)
    )
    # 直接通过 .start 和 .end 属性获取时间区间
    for start, end in zip(epochs.start, epochs.end)
    ]
    
    threshold = len(epochs_frames) * FilterTime_thresholdRatio #FilterTime Threshold ratio
    neuron_activity = transients[:, neuron_idx] > 0  # shape: (num_frames,)

    # 3) check mean_inside > 7 * mean_outside for _any_ field
    for start_bin, end_bin in field_ranges:
        valid_mean = False
        valid_time = False
        # positional slice with .iloc
        in_data  = tc.iloc[start_bin:end_bin]   # positional slice, OK
        n_bins = len(tc)
        mask_this_field = np.zeros(n_bins, dtype=bool)
        mask_this_field[start_bin:end_bin] = True
        # out_data = tc.iloc[~mask_this_field]

        mean_in  = np.nanmean(in_data)
        mean_out = np.nanmean(out_data)
        if mean_in >= mean_out * FilterMean_threshold:
            valid_mean = True
        else:
            continue

        traversals = 0
        for start, end in epochs_frames:
            positions_epoch = bin_indices.values[start:end]
            activity_epoch = neuron_activity[start:end]

            # 判断是否落在当前 field 范围内
            in_field = (positions_epoch >= start_bin) & (positions_epoch <= end_bin)
            significant_activity = activity_epoch & in_field
            
            if np.any(significant_activity):
                traversals += 1
            

        if traversals > threshold:
            valid_time = True

        if (valid_mean and valid_time): 
            result.append([end_bin-start_bin+1])
    
    return(result)
    # 第四步：检查跨 epoch 的遍历次数是否超过阈值
    # 预处理 epochs，转换为帧索引并确保在有效范围内
    # epochs_frames = [
    #     (
    #         max(int(np.floor(epoch['start'] * frame_rate)), 0),
    #         min(int(np.ceil(epoch['end'] * frame_rate)), num_frames)
    #     )
    #     for epoch in epochs.values()      



def identify_place_fields(smoothcurves, neuron_idx, threshold_ratio=0.25, min_true_len=3, max_true_len=30):
    """
    计算指定神经元的place fields，并返回布尔类型数组。
    
    参数：
    smoothcurves (np.array): 所有神经元的调谐曲线数组，形状为 (num_neurons, num_bins)。
    neuron_idx (int): 要计算place fields的神经元索引。
    threshold_ratio (float): 阈值比例，用于确定place fields的阈值。
    min_true_len (int): 最小的连续True长度，用于识别有效的place fields。
    max_true_len (int): 最大的连续True长度，用于识别有效的place fields。
    
    返回值：
    np.array: 布尔类型数组，标记出指定神经元的place fields。
    """
    
    neuron_tuning_curves = smoothcurves[neuron_idx]
    
    # 计算峰值和基线值
    peak = np.max(neuron_tuning_curves)
    sorted_values = np.sort(neuron_tuning_curves)
    low_20_percent = sorted_values[:int(0.2 * len(sorted_values))]
    baseline = np.mean(low_20_percent)
    
    # 计算阈值（基线 + 0.25 * (峰值 - 基线)）
    threshold = baseline + (peak - baseline) * threshold_ratio
    
    # 找到超过阈值的点
    potential = neuron_tuning_curves > threshold
    
    # 使用 Scipy 识别连续的True区域
    labeled, num_features = label(potential)
    
    # 创建布尔数组存储位置场
    place_fields = np.zeros_like(potential, dtype=bool)
    for feature in range(1, num_features + 1):
        indices = np.where(labeled == feature)[0]
        field_length = len(indices)
        if min_true_len <= field_length < max_true_len:
            place_fields[indices] = True
    
    return place_fields


def compute_spatial_information(neuron_idx,activity,tuning_curves,bin_indices):
    # 计算每个空间区间的时间占比 p_i
    p = np.array([np.sum(bin_indices == i) / len(bin_indices) for i in range(max(bin_indices)+1)])  # 位置分布的概率
    a = np.mean(activity)  # 总体平均活动

    # 计算每个区间的平均活动 a_i，处理空区间
    a_i = tuning_curves[neuron_idx]
    
    # 防止 a_i 为零，添加一个小的 epsilon
    epsilon = 1e-10
    a_i = np.where(a_i == 0, epsilon, a_i)

    
    # 计算互信息量 I = sum(p_i * (a_i / a) * log2(a_i / a))
    spatial_info_real = np.sum(p * (a_i / a) * np.log2(a_i / a))
    
    return spatial_info_real