import numpy as np
import pandas as pd
import pynapple as nap
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d

def frameinsert(chunk_size,n_chunks,insert_frames,total_inserts,transients):
    original_data = transients.values  # (36000, cellnum)
    original_times = transients.times()
    dt = np.diff(original_times).mean()  # 帧间隔 (约 0.0333 秒
    # 预分配内存（精确尺寸）
    new_data = np.zeros((original_data.shape[0] + total_inserts, original_data.shape[1]), dtype=original_data.dtype)
    new_data = np.zeros((original_data.shape[0] + total_inserts, original_data.shape[1]), dtype=original_data.dtype)
    new_times = np.zeros(original_data.shape[0] + total_inserts)

    # 索引指针初始化
    current_new_idx = 0
    for chunk in range(n_chunks):
        # === 插入3帧 ===
        if chunk == 0:
            # 首块前：复制首帧
            insert_values = np.tile(original_data[0], (insert_frames, 1))
            insert_times = np.linspace(0, insert_frames*dt, insert_frames, endpoint=False)
        else:
            # 中间块前：线性插值
            prev_last = original_data[chunk*chunk_size - 1]
            curr_first = original_data[chunk*chunk_size]
            insert_values = np.linspace(prev_last, curr_first, insert_frames+2)[1:-1]
            
            # 关键修复：精确生成3个时间戳
            insert_times = np.linspace(
                start=current_new_time + dt,
                stop=current_new_time + (insert_frames+1)*dt,
                num=insert_frames,
                endpoint=False
            )
        
        # 严格验证时间戳数量
        assert len(insert_times) == insert_frames, f"插入时间戳数量错误：{len(insert_times)} 应等于 {insert_frames}"
        
        # 填充插入数据
        new_data[current_new_idx:current_new_idx+insert_frames] = insert_values
        new_times[current_new_idx:current_new_idx+insert_frames] = insert_times
        current_new_idx += insert_frames

        # === 写入当前块数据 ===
        chunk_start = chunk * chunk_size
        chunk_end = chunk_start + chunk_size
        
        # 生成精确3600个时间戳（使用相对时间计算）
        chunk_times = new_times[current_new_idx-1] + dt + np.arange(chunk_size)*dt
        
        new_data[current_new_idx:current_new_idx+chunk_size] = original_data[chunk_start:chunk_end]
        new_times[current_new_idx:current_new_idx+chunk_size] = chunk_times
        current_new_idx += chunk_size
        current_new_time = new_times[current_new_idx-1]  # 记录当前时间指针

        # 最终维度验证
        n_frames = total_inserts + chunk_size*n_chunks
        assert len(new_times) == n_frames, f"时间戳总数应为{n_frames}，实际为{len(new_times)}"
        assert new_data.shape[0] == n_frames, f"数据维度应为{n_frames}，实际为{new_data.shape[0],}"
        
    new_transients = nap.TsdFrame(
        t=new_times,
        d=new_data,
        time_units='s',
        columns=transients.columns
        )

    print(f"时间连续性检查：{np.allclose(np.diff(new_times), dt, atol=1e-6)}")  # 应返回True
    print(f"首块前插入时间示例：{new_times[:3]} → 原始首时间：{original_times[0]}")
    return new_transients

def smoothAngularTuningCurves(tuning_curves, sigma=1):
    smoothed_data = np.zeros_like(tuning_curves.values)
    for i in range(tuning_curves.shape[1]):
        # 使用 mode='wrap' 处理环形数据
        smoothed_data[:, i] = gaussian_filter1d(
            tuning_curves.iloc[:, i].values, 
            sigma=sigma, 
            mode='wrap'  # 关键修改：环形卷积
        )
    return pd.DataFrame(
        data=smoothed_data, 
        index=tuning_curves.index, 
        columns=tuning_curves.columns
    )

def binning(Filtered_Position,bin_num):
    position_bins = np.linspace(np.min(Filtered_Position.values),
                            np.max(Filtered_Position.values),
                            bin_num + 1)
    bin_indices = np.digitize(Filtered_Position.values,
                            position_bins,
                            right=True) - 1
    bin_indices = np.clip(bin_indices, 0, bin_num - 1)

    bin_indices = pd.Series(bin_indices)
    return(bin_indices)

def compute_velocity(epochs,position,track_length,filtered_transients,min_speed_threshold=5):
    #计算累计距离
    cumulative_position_tmp = []  
    for i in range(0, len(epochs)): 
        temp = position.restrict(epochs[i]) + track_length * i
        cumulative_position_tmp.append(temp)  
    cumulative_position_array = np.concatenate(cumulative_position_tmp)
    #构建另一个插值函数并插值
    interp_func = interp1d(
        cumulative_position_array.times(),
        cumulative_position_array.values,
        fill_value="extrapolate"
    )
    tmp = interp_func(filtered_transients.times())   # use interpolation function returned by `interp1d`
    cumulative_position = nap.Tsd(
        t=filtered_transients.times(),
        d=tmp,
        time_units='s'
    )

    #用累计距离计算速度
    velocity = np.array(0)
    for i in range(1, len(cumulative_position.times())):
        velocity = np.append(
            velocity,
            abs(cumulative_position.values[i] - cumulative_position.values[i-1]) /
            (cumulative_position.times()[i] - cumulative_position.times()[i-1])
        )
    #进行卷积
    smoothWindowSize = 29
    smoothed_middle = np.convolve(
        velocity,
        np.ones(smoothWindowSize, dtype=int),
        'valid'
    ) / smoothWindowSize   
    r = np.arange(1, smoothWindowSize-1, 2)
    start = np.cumsum(velocity[:smoothWindowSize-1])[::2] / r
    stop = (np.cumsum(velocity[:-smoothWindowSize:-1])[::2] / r)[::-1]
    velocity = np.concatenate((start, smoothed_middle, stop))

    is_running = (velocity > min_speed_threshold)#跑动帧

    return velocity,is_running

def get_parts(tsd,split_time,pre):
    if pre:
        # 第一段：时间戳 ≤ split_time（包含分界点）
        tsd_part = tsd[tsd.times() <= split_time]
    else:
        # 第二段：时间戳 > split_time（不包含分界点）
        tsd_part = tsd[tsd.times() > split_time]
    return tsd_part

def compute_tuningcurves(transients,position,bin_num):
    initial_tuning_curves = nap.compute_1d_tuning_curves_continuous(
        transients,
        feature=position,
        nb_bins=bin_num,
    )
    smoothcurves = smoothAngularTuningCurves(initial_tuning_curves, sigma=1) #PandasDF,80bin*cells   
    return smoothcurves

def pearson_correlation(x, y):
    # 检查向量维度是否一致
    if len(x) != len(y):
        raise ValueError("两个向量的维度必须相同")
    x[np.isnan(x)]=0
    y[np.isnan(y)]=0
    n = len(x)
    
    # 计算各项求和值
    sum_x = sum(x)
    sum_y = sum(y)
    sum_x2 = sum(i*i for i in x)
    sum_y2 = sum(i*i for i in y)
    sum_xy = sum(x[i]*y[i] for i in range(n))
    
    # 计算皮尔森相关系数分子
    numerator = n * sum_xy - sum_x * sum_y
    
    # 计算皮尔森相关系数分母
    denominator = ((n * sum_x2 - sum_x**2) * (n * sum_y2 - sum_y**2)) ** 0.5
    
    # 避免分母为0的情况
    if denominator == 0:
        return 0.0
    
    # 计算最终结果
    return numerator / denominator