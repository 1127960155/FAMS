import csv
import os

def analyze_result_csv(input_csv, output_csv):
    with open(input_csv, 'r', newline='') as fin:
        reader = csv.DictReader(fin)
        rows = list(reader)
        # 只保留以下列：
        keep_fields = [
            # 保留映射策略相关
            'm','k','n','dataflow','loops',
            # 保留实验结果相关
            'cycle',#'computation_cycle','communication_cycle','communication_A_cycle',
            #'communication_B_cycle','communication_C_cycle','communication_Y_cycle',
            'PE_utility','throughput','communication_off_chip','communication_on_chip',
            'communication_internal',
            'energy_cost_communication','energy_cost_computation'
        ]

        # 新增字段
        keep_fields += ['total_energy', 'edp']

        new_rows = []
        for row in rows:
            # 计算新字段
            try:
                comm_e = float(row['energy_cost_communication'])
                comp_e = float(row['energy_cost_computation'])
                cycle = float(row['cycle'])
            except Exception as e:
                comm_e = comp_e = cycle = 0
            total_energy = comm_e + comp_e
            edp = total_energy * cycle

            # 保留指定列并添加新字段
            new_row = {k: row[k] for k in keep_fields if k in row}
            new_row['total_energy'] = total_energy
            new_row['edp'] = edp
            # 若保留列里没有的字段则补齐
            for k in ['total_energy', 'edp']:
                if k not in new_row:
                    new_row[k] = locals()[k]
            new_rows.append(new_row)

        with open(output_csv, 'w', newline='') as fout:
            writer = csv.DictWriter(fout, fieldnames=keep_fields)
            writer.writeheader()
            for r in new_rows:
                writer.writerow(r)

if __name__ == "__main__":
    base_dir = os.path.join("data", "output")
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path):
            input_csv = os.path.join(folder_path, "result.csv")
            output_csv = os.path.join(folder_path, "result_analysis.csv")
            if os.path.exists(input_csv):
                print(f"Processing {input_csv} ...")
                analyze_result_csv(input_csv, output_csv)
                print(f"  => Done: {output_csv}")
            else:
                print(f"Skipped {folder_path}, result.csv not found.")
    print("全部处理完成！")