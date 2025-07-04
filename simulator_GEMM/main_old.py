# encoding: utf-8
from __future__ import division
from __future__ import print_function
import csv
import os
import argparse
import time
time_start = time.time()
def roundup(X,x):
    temp = X // x
    reminder = X % x
    if reminder != 0:
        temp += 1
    return temp

parser = argparse.ArgumentParser(description='tracking file')
parser.add_argument('--tracking', type=int, default=0)
args = parser.parse_args()


stop_flag = 0
level_num = 0
architecture_parameters = open('data/architecture_parameters.csv', 'r')
dataflow_parameters = open('data/dataflow_parameters.csv', 'r')
workload_parameters = open('data/workload_parameters.csv', 'r')
freader_architecture_parameters = csv.reader(architecture_parameters, delimiter=',')
freader_dataflow_parameters = csv.reader(dataflow_parameters, delimiter=',')
freader_workload_parameters = csv.reader(workload_parameters, delimiter=',')
variable_header_architecture_parameters = next(freader_architecture_parameters)
variable_header_dataflow_parameters = next(freader_dataflow_parameters)
variable_header_workload_parameters = next(freader_workload_parameters)
variable_architecture_parameters = next(freader_architecture_parameters)
variable_dataflow_parameters = next(freader_dataflow_parameters)
variable_workload_parameters = next(freader_workload_parameters)
level_total = int(variable_workload_parameters[0])
os.makedirs('./data/output', exist_ok = True)

while(stop_flag != 1):
    level_num += 1
    print('simulation level%d computing......'%(level_num))
    intermediate_variables = open('data/intermediate_variables.txt',mode='w')
    intermediate_variables.write(str(level_num))
    intermediate_variables.write('\n')
    intermediate_variables.write(str(args.tracking))
    intermediate_variables.close()
    M, K, N = int(variable_workload_parameters[1]), int(variable_workload_parameters[2]), int(variable_workload_parameters[3])
    m, k, n = int(variable_dataflow_parameters[0]), int(variable_dataflow_parameters[1]), int(variable_dataflow_parameters[2])
    dataflow, loops = str(variable_dataflow_parameters[3]), str(variable_dataflow_parameters[4])
    print('M, K, N = %d, %d, %d' %(M, K, N))
    m_loop_times, n_loop_times, k_loop_times = roundup(M,m), roundup(N,n), roundup(K,k)
    print('m_loop_times, k_loop_times, n_loop_times = %d, %d, %d' %(m_loop_times, k_loop_times, n_loop_times))

    if m_loop_times == 1 and k_loop_times == 1 and n_loop_times == 1:
        os.system('python ./%s/once/start_once.py' %(dataflow))
    else:
        os.system('python ./%s/%s/start_repeat.py' %(dataflow, loops))

    print('simulation level%d complete'%(level_num))
    if level_num == level_total:
        stop_flag = 1
    else:
        variable_architecture_parameters = next(freader_architecture_parameters)
        variable_dataflow_parameters = next(freader_dataflow_parameters)
        variable_workload_parameters = next(freader_workload_parameters)

architecture_parameters.close()
dataflow_parameters.close()
workload_parameters.close()
#os.remove("data/intermediate_variables.txt")
print('simulation complete')
time_end = time.time()
print('cost time {}'.format(time_end - time_start))
