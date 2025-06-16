import os
import csv
import time
from typing import List, Dict, Any

# -----------------------------
# 架构、负载、映射参数类定义
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
# 辅助函数
# -----------------------------
def roundup(X, x):
    return (X + x - 1) // x

def generate_all_strategies(workload: WorkloadParam, arch: ArchitectureParam) -> List[DataflowStrategy]:
    """为给定workload/arch，生成所有合法的DataflowStrategy（m/k/n都要是array_i倍数且能整除M/K/N，且 m*k, k*n, m*n < sram_size/2）"""
    M, K, N = workload.M, workload.K, workload.N
    array_step = arch.Array_i  # array_i == array_j
    dataflows = ["is", "ws", "os"]
    loops_list = ["nkm", "nmk", "kmn", "knm", "mnk", "mkn"]
    sram_limit = arch.sram_size // 2  # 半容量
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
                # SRAM约束
                if (m * k >= sram_limit) or (k * n >= sram_limit) or (m * n >= sram_limit):
                    continue
                for dataflow in dataflows:
                    for loops in loops_list:
                        strategies.append(DataflowStrategy(m, k, n, dataflow, loops))
    return strategies

def save_map_space(strategies: List[DataflowStrategy], out_csv: str):
    if not strategies:
        print("No mapping strategies to save.")
        return
    with open(out_csv, 'w', newline='') as f:
        fieldnames = list(strategies[0].to_dict().keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for strategy in strategies:
            writer.writerow(strategy.to_dict())

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

def run_simulation(arch: ArchitectureParam, wl: WorkloadParam, dfs: DataflowStrategy, idx: int) -> Dict[str, Any]:
    os.makedirs('data/output', exist_ok=True)
    # 追加参数到csv，三者都可能重复
    append_architecture_csv(arch)
    append_workload_csv(wl)
    append_dataflow_csv(dfs)
    intermediate_variables_path = 'data/intermediate_variables.txt'
    with open(intermediate_variables_path, 'w') as f:
        f.write(str(idx) + '\n')
        f.write('0\n')  # tracking参数

    m_loop_times = roundup(wl.M, dfs.m)
    k_loop_times = roundup(wl.K, dfs.k)
    n_loop_times = roundup(wl.N, dfs.n)

    if m_loop_times == 1 and k_loop_times == 1 and n_loop_times == 1:
        cmd = f"python ./os/once/start_once.py"
    else:
        cmd = f"python ./os/{dfs.loops}/start_repeat.py"

    print(f"[#{idx}] CMD: {cmd}")
    ret = os.system(cmd)

    # 改为读取真实模拟结果 output_{idx}.csv
    result_file = f'./data/output/output_{idx}.csv'
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
    
    

    # 合并所有参数与仿真统计结果
    result = {**arch.to_dict(), **wl.to_dict(), **dfs.to_dict()}
    result.update({
        'm_loop_times': m_loop_times,
        'k_loop_times': k_loop_times,
        'n_loop_times': n_loop_times,
        'cmd': cmd,
        'ret_code': ret
    })
    result.update(sim_result)  # 添加所有仿真统计字段（如cycle, throughput, energy...）
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
    clear_all_csv()

    architectures = [
        ArchitectureParam(
            bandwidth=64, 
            data_precision=8, 
            compute_pipeline=1, 
            Array_i=32, 
            Array_j=32, 
            sram_size=256*1024  # 单位: Bytes
        ),
        # 可以继续添加其他架构
    ]

    workloads = [
        WorkloadParam(
            M=64, 
            K=64, 
            N=64
        ),
        # 可以继续添加其他 workload
    ]

    results = []
    idx = 1
    for arch in architectures:
        for wl in workloads:
            strategies = generate_all_strategies(wl, arch)
            save_map_space(strategies, "data/output/map_space.csv")
            for dfs in strategies:
                print(f"\n==== Running Simulation {idx} ====")
                print("Architecture:", arch.to_dict())
                print("Workload    :", wl.to_dict())
                print("Dataflow    :", dfs.to_dict())
                result = run_simulation(arch, wl, dfs, idx)
                results.append(result)
                idx += 1

    output_csv = 'data/output/simulation_summary.csv'
    save_results(results, output_csv)

    print(f"\nSimulation complete. Results saved to {output_csv}")
    print('Total cost time: {:.2f} seconds'.format(time.time() - time_start))

if __name__ == '__main__':
    main()
