#TODO: 缓存空间似乎给的太大了，可以再测测更小的缓存
# TODO: 为简化处理，假设所有的工作负载都是正方形矩阵，排除了IS数据流
#TODO：似乎没有写出总能耗和edp
import os
import csv
import time
from typing import List, Dict, Any
from tqdm import tqdm

# -----------------------------
# 参数类定义（同你原本代码）
# -----------------------------
class ArchitectureParam:
    def __init__(self, bandwidth, data_precision, compute_pipeline, Array_i, Array_j, sram_size):
        self.bandwidth = bandwidth
        self.data_precision = data_precision
        self.compute_pipeline = compute_pipeline
        self.Array_i = Array_i
        self.Array_j = Array_j
        self.sram_size = sram_size  # Bytes

    def to_dict(self):
        return vars(self)

class WorkloadParam:
    def __init__(self, M, K, N):
        self.M = M
        self.K = K
        self.N = N

    def to_dict(self):
        return vars(self)

class DataflowStrategy:
    def __init__(self, m, k, n, dataflow, loops):
        self.m = m
        self.k = k
        self.n = n
        self.dataflow = dataflow
        self.loops = loops

    def to_dict(self):
        return vars(self)

# -----------------------------
# 其它通用函数
# -----------------------------
def roundup(X, x):
    return (X + x - 1) // x

def generate_all_strategies(workload: WorkloadParam, arch: ArchitectureParam) -> List[DataflowStrategy]:
    M, K, N = workload.M, workload.K, workload.N
    array_step = arch.Array_i
    dataflows = ["ws", "os"]    
    loops_list = ["nkm", "nmk", "kmn", "knm", "mnk", "mkn"]
    sram_limit = arch.sram_size // 2  # 由于双缓冲，只能使用其一半的容量
    strategies = []
    for m in range(array_step, M+1, array_step):
        if M % m != 0:
            continue
        for k in range(array_step, K+1, array_step):
            if K % k != 0:
                continue
            for n in range(array_step, N+1, array_step):
                if N % n != 0:
                    continue
                if (m * k >= sram_limit) or (k * n >= sram_limit) or (m * n >= sram_limit):
                    continue
                for dataflow in dataflows:
                    for loops in loops_list:
                        strategies.append(DataflowStrategy(m, k, n, dataflow, loops))
    return strategies

def save_map_space(strategies: List[DataflowStrategy], out_csv: str):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    if not strategies:
        print("No mapping strategies to save.")
        return
    with open(out_csv, 'w', newline='') as f:
        fieldnames = list(strategies[0].to_dict().keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for strategy in strategies:
            writer.writerow(strategy.to_dict())



# -----------------------------
# 运行一次仿真（每次覆盖写参数CSV到data/目录）
def append_architecture_csv(arch: ArchitectureParam):
    file_path = 'data/architecture_parameters.csv'
    header = ['bandwidth(B)','data_precision(bit)','compute_pipeline','Array_i(height)','Array_j(width)','sram_size']
    need_header = not os.path.exists(file_path) or os.stat(file_path).st_size == 0
    with open(file_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if need_header:
            writer.writerow(header)
        writer.writerow([arch.bandwidth, arch.data_precision, arch.compute_pipeline, arch.Array_i, arch.Array_j, arch.sram_size])

def append_workload_csv(wl: WorkloadParam):
    file_path = 'data/workload_parameters.csv'
    header = ['level_total','M','K','N']
    need_header = not os.path.exists(file_path) or os.stat(file_path).st_size == 0
    with open(file_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if need_header:
            writer.writerow(header)
        writer.writerow([1, wl.M, wl.K, wl.N])

def append_dataflow_csv(dfs: DataflowStrategy):
    file_path = 'data/dataflow_parameters.csv'
    header = ['m','k','n','dataflow','loops']
    need_header = not os.path.exists(file_path) or os.stat(file_path).st_size == 0
    with open(file_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if need_header:
            writer.writerow(header)
        writer.writerow([dfs.m, dfs.k, dfs.n, dfs.dataflow, dfs.loops])

def clear_all_csv():
    for fname in [
        'data/architecture_parameters.csv',
        'data/workload_parameters.csv',
        'data/dataflow_parameters.csv'
    ]:
        if os.path.exists(fname):
            os.remove(fname)


# -----------------------------
def run_simulation(arch, wl, dfs, idx):
    # 1. 覆盖写入参数文件
    clear_all_csv()
    append_architecture_csv(arch)
    append_workload_csv(wl)
    append_dataflow_csv(dfs)
    # 2. 写中间变量文件（也放在data/，如有需要可放入output_dir）
    intermediate_variables_path = 'data/intermediate_variables.txt'
    with open(intermediate_variables_path, 'w') as f:
        f.write('1\n')
        f.write('0\n')
    # 3. 判定仿真类型和路径
    m_loop_times = roundup(wl.M, dfs.m)
    k_loop_times = roundup(wl.K, dfs.k)
    n_loop_times = roundup(wl.N, dfs.n)
    if m_loop_times == 1 and k_loop_times == 1 and n_loop_times == 1:
        cmd = f"python ./os/once/start_once.py"
    else:
        cmd = f"python ./os/{dfs.loops}/start_repeat.py"
        

    # print(f"[#{idx}] CMD: {cmd}")
    ret = os.system(cmd)
    # 4. 读取仿真结果并删除
    result_file = f'./data/output/output_1.csv'
    if os.path.exists(result_file):
        with open(result_file, 'r', newline='') as rf:
            reader = csv.DictReader(rf)
            rows = list(reader)
            if rows:
                sim_result = rows[0]  # 只取一行（通常就是唯一结果）
            else:
                sim_result = {}        
        try:
            os.remove(result_file)
        except Exception as e:
            print(f"Warning: could not delete {result_file}: {e}")
    else:
        sim_result = {}
    # 5. 合并统计
    result = {**arch.to_dict(), **wl.to_dict(), **dfs.to_dict()}
    result.update({
        'm_loop_times': m_loop_times,
        'k_loop_times': k_loop_times,
        'n_loop_times': n_loop_times,
        'cmd': cmd,
        'ret_code': ret
    })
    result.update(sim_result)
    return result

def save_results(results: List[Dict[str, Any]], out_csv: str):
    if not results:
        print("No results to save.")
        return
    with open(out_csv, 'w', newline='') as f:
        fieldnames = list(results[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
# -----------------------------
# 主流程
# -----------------------------
def main():
    time_start = time.time()
    array_sizes = [(64, 64), (128, 128), (256, 256)] # 缩小了映射空间范围
    sram_sizes = [64*1024,128*1024,256*1024, 512*1024,1024*1024,2048*1024,4096*1024]
    bitwidths = [512, 1024, 2048,4096,8192 ]  # 注意：这里的bitwidths是以bit为单位的
    workload_shapes = [(128, 128, 128), (256, 256, 256), (512, 512, 512), (1024, 1024, 1024),(2048, 2048, 2048)]
    
    # test
    # array_sizes = [(32, 32), (64, 64)]
    # sram_sizes_total = [256*1024]
    # bitwidths = [128]
    # workload_shapes = [(128, 128, 128)]
    
    byte_bandwidths = [b // 8 for b in bitwidths]
    data_precision = 8
    compute_pipeline = 1

    architectures = []
    for (ai, aj) in array_sizes:
        for sram in sram_sizes:
            for bw in byte_bandwidths:
                architectures.append(
                    ArchitectureParam(
                        bandwidth=bw,
                        data_precision=data_precision,
                        compute_pipeline=compute_pipeline,
                        Array_i=ai,
                        Array_j=aj,
                        sram_size=sram
                    )
                )

    
    workloads = [WorkloadParam(M=m, K=k, N=n) for (m, k, n) in workload_shapes]

    
    for arch in architectures:
        for wl in workloads:
            folder_name = (
                f"data/output/array{arch.Array_i}x{arch.Array_j}_"
                f"sram{arch.sram_size//1024}KB_"
                f"bw{arch.bandwidth}B_"
                f"M{wl.M}_K{wl.K}_N{wl.N}"
            )
            map_space_path = os.path.join(folder_name, "map_space.csv")
            result_csv_path = os.path.join(folder_name, "result.csv")
            
            print("folder_name:", folder_name)
            # ----------- 检查是否已存在，若已完成则跳过 -----------
            if os.path.exists(result_csv_path) and os.path.exists(map_space_path):
                print(f"【跳过】{folder_name} 已存在 result.csv 和 map_space.csv，跳过本配置。")
                continue
            
            idx = 1 # 用于标识每种组合配置下各个映射策略的仿真序号
            os.makedirs(folder_name, exist_ok=True)
            strategies = generate_all_strategies(wl, arch)
            save_map_space(strategies, map_space_path)
            config_results = []
            print(f"\n==== Running Simulation {idx} ====")
            print("Architecture:", arch.to_dict())
            print("Workload    :", wl.to_dict())
            for dfs in tqdm(strategies, desc=f"Simulating {folder_name}"):
                result = run_simulation(arch, wl, dfs, idx)
                config_results.append(result)
                idx += 1
            save_results(config_results, result_csv_path)

    print('\nSimulation complete.')
    print('Total cost time: {:.2f} seconds'.format(time.time() - time_start))

    # 原运行配置
    time_start = time.time()
    array_sizes = [(32, 32), (64, 64), (128, 128), (256, 256)] # 缩小了映射空间范围
    sram_sizes_total = [256*1024, 512*1024,1024*1024]
    bitwidths = [128, 256, 512, 1024, 2048 ]  # 注意：这里的bitwidths是以bit为单位的
    workload_shapes = [(128, 128, 128), (256, 256, 256), (512, 512, 512), (1024, 1024, 1024)]
    
    # test
    # array_sizes = [(32, 32), (64, 64)]
    # sram_sizes_total = [256*1024]
    # bitwidths = [128]
    # workload_shapes = [(128, 128, 128)]
    
    sram_sizes = [s // 2 for s in sram_sizes_total]  # 由于双缓冲，只能使用其一半的容量
    byte_bandwidths = [b // 8 for b in bitwidths]
    data_precision = 8
    compute_pipeline = 1

    architectures = []
    for (ai, aj) in array_sizes:
        for sram in sram_sizes:
            for bw in byte_bandwidths:
                architectures.append(
                    ArchitectureParam(
                        bandwidth=bw,
                        data_precision=data_precision,
                        compute_pipeline=compute_pipeline,
                        Array_i=ai,
                        Array_j=aj,
                        sram_size=sram
                    )
                )

    
    workloads = [WorkloadParam(M=m, K=k, N=n) for (m, k, n) in workload_shapes]

    
    for arch in architectures:
        for wl in workloads:
            folder_name = (
                f"data/output/array{arch.Array_i}x{arch.Array_j}_"
                f"sram{arch.sram_size//1024}KB_"
                f"bw{arch.bandwidth}B_"
                f"M{wl.M}_K{wl.K}_N{wl.N}"
            )
            map_space_path = os.path.join(folder_name, "map_space.csv")
            result_csv_path = os.path.join(folder_name, "result.csv")
            
            print("folder_name:", folder_name)
            # ----------- 检查是否已存在，若已完成则跳过 -----------
            if os.path.exists(result_csv_path) and os.path.exists(map_space_path):
                print(f"【跳过】{folder_name} 已存在 result.csv 和 map_space.csv，跳过本配置。")
                continue
            
            idx = 1 # 用于标识每种组合配置下各个映射策略的仿真序号
            os.makedirs(folder_name, exist_ok=True)
            strategies = generate_all_strategies(wl, arch)
            save_map_space(strategies, map_space_path)
            config_results = []
            print(f"\n==== Running Simulation {idx} ====")
            print("Architecture:", arch.to_dict())
            print("Workload    :", wl.to_dict())
            for dfs in tqdm(strategies, desc=f"Simulating {folder_name}"):
                result = run_simulation(arch, wl, dfs, idx)
                config_results.append(result)
                idx += 1
            save_results(config_results, result_csv_path)

    print('\nSimulation complete.')
    print('Total cost time: {:.2f} seconds'.format(time.time() - time_start))

if __name__ == '__main__':
    main()
