# encoding: utf-8
from __future__ import division
from __future__ import print_function
import transform_unit_repeat as t
import transform_unit_repeat_last as tlast
import functions as f
import csv
import numpy as np
import sys


# 超参数
architecture_parameters = open('data/architecture_parameters.csv', 'r')
dataflow_parameters = open('data/dataflow_parameters.csv', 'r')
workload_parameters = open('data/workload_parameters.csv', 'r')
freader_architecture_parameters = csv.reader(architecture_parameters, delimiter=',')
freader_dataflow_parameters = csv.reader(dataflow_parameters, delimiter=',')
freader_workload_parameters = csv.reader(workload_parameters, delimiter=',')
variable_header_architecture_parameters = next(freader_architecture_parameters)
variable_header_dataflow_parameters = next(freader_dataflow_parameters)
variable_header_workload_parameters = next(freader_workload_parameters)
intermediate_variables = open('data/intermediate_variables.txt',mode='r')
level_num_tracking = intermediate_variables.readlines()
intermediate_variables.close()
level_num = int(level_num_tracking[0])
tracking = int(level_num_tracking[1])
for i in range(level_num):
    variable_architecture_parameters = next(freader_architecture_parameters)
    variable_dataflow_parameters = next(freader_dataflow_parameters)
    variable_workload_parameters = next(freader_workload_parameters)
architecture_parameters.close()
dataflow_parameters.close()
workload_parameters.close()
if tracking == 1:
    output_file = open('./data/output/track_file_%d.csv'%(level_num), 'w')
    fieldnames = ['cycle','G_B0_ptw', 'G_B1_ptw','G_B0_ptr','G_B1_ptr','G_A0_ptw','G_A1_ptw','G_A0_ptr',\
    'G_A1_ptr','G_C0_ptr','G_C1_ptr','G_C0_ptw','G_C1_ptw']
    writer = csv.DictWriter(output_file, fieldnames=fieldnames)
    writer.writeheader()
IX, IY, FW, FH = int(variable_workload_parameters[1]), int(variable_workload_parameters[2]), int(variable_workload_parameters[3]), int(variable_workload_parameters[4])
C_channel, K_channel, Batchsize = int(variable_workload_parameters[5]), int(variable_workload_parameters[6]), int(variable_workload_parameters[7])
stride, padding = int(variable_workload_parameters[8]), str(variable_workload_parameters[9])
m, k, n = int(variable_dataflow_parameters[0]), int(variable_dataflow_parameters[1]), int(variable_dataflow_parameters[2])
dataflow, loops = str(variable_dataflow_parameters[3]), str(variable_dataflow_parameters[4])
dataflow_M, dataflow_K = str(variable_dataflow_parameters[5]), str(variable_dataflow_parameters[6])
bandwidth_B, data_precision, compute_pipeline = int(variable_architecture_parameters[0]), int(variable_architecture_parameters[1]), int(variable_architecture_parameters[2])
Array_i, Array_j = int(variable_architecture_parameters[3]), int(variable_architecture_parameters[4])
bandwidth = bandwidth_B * 8 / data_precision

if padding == 'valid':
    OX, OY = (IX - FW) // stride + 1, (IY - FH) // stride + 1
else: # padding == 'same'
    OX, OY = (IX - 1) // stride + 1, (IY - 1) // stride + 1
M, K, N = K_channel, FH * FW * C_channel, OX * OY * Batchsize
m, k, n = min(M, m), min(K, k), min(N, n)


# 参数初始化
cycle = 0
communication_cycle = 0
communication_A_cycle, communication_B_cycle = 0, 0
communication_C_cycle, communication_Y_cycle = 0, 0
communication = 0
on_chip_alter = 0
G_A_in_use, G_B_in_use = 0, 0
G_A0_ready, G_A1_ready = 0, 0
G_A0_ptr, G_A0_ptw, G_A1_ptr, G_A1_ptw = 0, 0, 0, 0
a_loops, b_loops = 1, 1
G_B0_ready, G_B1_ready = 0, 0
G_B0_ptr, G_B0_ptw, G_B1_ptr, G_B1_ptw = 0, 0, 0, 0
B_SA_load = 0
G_C0_ready, G_C1_ready = 0, 0
G_C0_ptw, G_C0_ptr, G_C1_ptw, G_C1_ptr = 0, 0, 0, 0
G_Cw_in_use, G_Cr_in_use = 0, 0
cw_loops, cr_loops = 1, 1
bw_in_use = 'B'
DR_list = [0 for i in range(Array_j-1)]
SA_list = [0 for i in range(Array_i*compute_pipeline)]
A_interval_work, A_interval_spare = 0, 0
G_B_ptr_add = 0
m_loop_times, n_loop_times, k_loop_times = f.roundup(M,m), f.roundup(N,n), f.roundup(K,k)
m_coordinate, k_coordinate, n_coordinate = 0, 0, 0
partial_sum = [[0 for i in range(n_loop_times)] for j in range(m_loop_times)]
communication_off_chip = 0
communication_on_chip = 0
communication_interval = 0
communication_internal = 0
# 初始化，将A与B的数据从HBM读到GSM
B_buf_tile_trans = f.ceil(k * n, bandwidth)
while(G_B0_ptw!=B_buf_tile_trans):
    G_B0_ptw += bandwidth
    cycle += 1
    communication_B_cycle += 1
    if tracking == 1:
        writer.writerow({'cycle':str(cycle),'G_B0_ptw':str(G_B0_ptw),'G_B1_ptw':str(G_B1_ptw),'G_B0_ptr':str(G_B0_ptr),'G_B1_ptr':str(G_B1_ptr),\
        'G_A0_ptw':str(G_A0_ptw),'G_A1_ptw':str(G_A1_ptw),'G_A0_ptr':str(G_A0_ptr),'G_A1_ptr':str(G_A1_ptr),\
        'G_C0_ptr':str(G_C0_ptr),'G_C1_ptr':str(G_C1_ptr),'G_C0_ptw':str(G_C0_ptw),'G_C1_ptw':str(G_C1_ptw)})
communication_off_chip += f.ceil(k * n, bandwidth)


A_buf_tile_trans = f.ceil(m * k, bandwidth)
while(G_A0_ptw!=A_buf_tile_trans):
    G_A0_ptw += bandwidth
    cycle += 1
    communication_A_cycle += 1
    if tracking == 1:
        writer.writerow({'cycle':str(cycle),'G_B0_ptw':str(G_B0_ptw),'G_B1_ptw':str(G_B1_ptw),'G_B0_ptr':str(G_B0_ptr),'G_B1_ptr':str(G_B1_ptr),\
        'G_A0_ptw':str(G_A0_ptw),'G_A1_ptw':str(G_A1_ptw),'G_A0_ptr':str(G_A0_ptr),'G_A1_ptr':str(G_A1_ptr),\
        'G_C0_ptr':str(G_C0_ptr),'G_C1_ptr':str(G_C1_ptr),'G_C0_ptw':str(G_C0_ptw),'G_C1_ptw':str(G_C1_ptw)})
communication_off_chip += f.ceil(m * k, bandwidth)
partial_sum[0][0] = 1
G_A0_ptw, G_B0_ptw, G_C0_ptw = 0, 0, 0
G_A0_ready, G_B0_ready, G_C0_ready = 1, 1, 1
# 初始化，将B_buf中的第一块读到SA中
for i in range(Array_i):
    G_B0_ptr += Array_j
    # b在读最后一列
    if b_loops % f.roundup(n, Array_j) == 0 and b_loops != f.roundup(n, Array_j) * f.roundup(k, Array_i):
        # Array_j个数据从B_buf读出
        communication_on_chip += n + Array_j - f.expansion(n, Array_j)
        # Array_j个数在PE间传递的次数不同，从1到Array_i-1，为了简化计算同时不失正确性，这里用平均数
        communication_interval += (Array_i - 1) * (n + Array_j - f.expansion(n, Array_j))
    # b在读最后一行
    elif b_loops - (f.roundup(k, Array_i) - 1) * f.roundup(n, Array_j) > 0 and b_loops != f.roundup(n, Array_j) * f.roundup(k, Array_i):
        # Array_j个数据从B_buf读出
        communication_on_chip += Array_j * (k + Array_i - f.expansion(k, Array_i)) / Array_i
        # Array_j个数在PE间传递的次数不同，从1到Array_i-1，为了简化计算同时不失正确性，这里用平均数
        communication_interval += (Array_i - 1) * (Array_j * (k + Array_i - f.expansion(k, Array_i)) / Array_i)
    # b在读最后一块
    elif b_loops == f.roundup(n, Array_j) * f.roundup(k, Array_i):
        # Array_j个数据从B_buf读出
        communication_on_chip += (n + Array_j - f.expansion(n, Array_j)) * (k + Array_i - f.expansion(k, Array_i)) / Array_i
        # Array_j个数在PE间传递的次数不同，从1到Array_i-1，为了简化计算同时不失正确性，这里用平均数
        communication_interval += (Array_i - 1) * ((n + Array_j - f.expansion(n, Array_j)) * (k + Array_i - f.expansion(k, Array_i)) / Array_i)
    else:
        # Array_j个数据从B_buf读出
        communication_on_chip += Array_j
        # Array_j个数在PE间传递的次数不同，从1到Array_i-1，为了简化计算同时不失正确性，这里用平均数
        communication_interval += (Array_i - 1) * Array_j
    cycle += 1
    if tracking == 1:
        writer.writerow({'cycle':str(cycle),'G_B0_ptw':str(G_B0_ptw),'G_B1_ptw':str(G_B1_ptw),'G_B0_ptr':str(G_B0_ptr),'G_B1_ptr':str(G_B1_ptr),\
        'G_A0_ptw':str(G_A0_ptw),'G_A1_ptw':str(G_A1_ptw),'G_A0_ptr':str(G_A0_ptr),'G_A1_ptr':str(G_A1_ptr),\
        'G_C0_ptr':str(G_C0_ptr),'G_C1_ptr':str(G_C1_ptr),'G_C0_ptw':str(G_C0_ptw),'G_C1_ptw':str(G_C1_ptw)})
b_loops += 1
B_SA_load += 1
A_interval_work, A_interval_spare = Array_i, (Array_i-1)*compute_pipeline + m + Array_j + 1
counter, counter_total = 0, m_loop_times * k_loop_times * n_loop_times
#################################################################计算部分开始#################################################
#############################################################################################################################
#############################################################################################################################
if m_loop_times == 1 and k_loop_times == 1 and n_loop_times >= 2:
    for n_serial in range(0,n_loop_times):
        for m_serial in range(0,m_loop_times):
            for k_serial in range(0,k_loop_times):
                counter += 1
                # 计算本次片上的mkn规模大小
                m_num, k_num, n_num = f.mkn_num(m_serial, k_serial, n_serial, m_loop_times, k_loop_times, n_loop_times, M, K, N, m, k, n)
                m_serial_next, k_serial_next, n_serial_next = f.next_num_nmk(m_serial, k_serial, n_serial, m_loop_times, k_loop_times, n_loop_times)
                m_serial_before, k_serial_before, n_serial_before = f.before_num_nmk(m_serial, k_serial, n_serial, m_loop_times, k_loop_times, n_loop_times)
                m_num_next, k_num_next, n_num_next = f.mkn_num(m_serial_next, k_serial_next, n_serial_next, m_loop_times, k_loop_times, n_loop_times, M, K, N, m, k, n)
                m_num_before, k_num_before, n_num_before = f.mkn_num(m_serial_before, k_serial_before, n_serial_before, m_loop_times, k_loop_times, n_loop_times, M, K, N, m, k, n)
                m_coordinate, k_coordinate, n_coordinate = m_serial_next * m, k_serial_next * k, n_serial_next * n

                A_buf_tile_trans = f.ceil(m_num_next * k_num_next, bandwidth)
                B_buf_tile_trans = f.ceil(k_num_next * n_num_next, bandwidth)
                # 判断C是否要从片外读入，第一次是不需要读入的
                if partial_sum[m_serial_next][n_serial_next] == 0:
                    C_buf_tile_trans = 0
                else:
                    C_buf_tile_trans = f.ceil(m_num_next * n_num_next, bandwidth)
                Y_buf_tile_trans = f.ceil(m_num_before * n_num_before, bandwidth)
                on_chip_alter = 0

                # 第一个
                if m_serial==0 and k_serial==0 and n_serial==0:
                    A_load, B_load, C_load_in, C_load_out = 0, 1, 1, 0
                    A_load0, B_load0, C_load_in0, C_load_out0 = 0, 1, 1, 0
                    while(on_chip_alter == 0):
                        G_A_in_use, G_A0_ready, G_A1_ready, G_A0_ptr, G_A0_ptw, G_A1_ptr, G_A1_ptw, a_loops, b_loops,\
                        G_B0_ready, G_B1_ready, G_B0_ptr, G_B0_ptw, G_B1_ptr, G_B1_ptw, G_B_in_use, B_SA_load,\
                        A_interval_work, A_interval_spare, G_C0_ready, G_C0_ptw, G_C0_ptr, G_C1_ready, G_C1_ptw,\
                        G_C1_ptr, G_Cw_in_use, G_Cr_in_use, cw_loops, cr_loops,\
                        on_chip_alter, bw_in_use, communication_A_cycle, communication_B_cycle, communication_C_cycle,\
                        communication_Y_cycle, B_load, G_B_ptr_add, communication_on_chip,\
                        communication_interval, communication_internal = t.buf_process(G_A_in_use,
                                                        G_A0_ready,
                                                        G_A1_ready,
                                                        G_A0_ptr,
                                                        G_A0_ptw,
                                                        G_A1_ptr,
                                                        G_A1_ptw,
                                                        a_loops,
                                                        b_loops,
                                                        G_B0_ready,
                                                        G_B1_ready,
                                                        G_B0_ptr,
                                                        G_B0_ptw,
                                                        G_B1_ptr,
                                                        G_B1_ptw,
                                                        G_B_in_use,
                                                        B_SA_load,
                                                        A_interval_work,
                                                        A_interval_spare,
                                                        G_C0_ready,
                                                        G_C0_ptw,
                                                        G_C0_ptr,
                                                        G_C1_ready,
                                                        G_C1_ptw,
                                                        G_C1_ptr,
                                                        G_Cw_in_use,
                                                        G_Cr_in_use,
                                                        cw_loops,
                                                        cr_loops,
                                                        Array_i,
                                                        Array_j,
                                                        DR_list,
                                                        SA_list,
                                                        m_num,
                                                        n_num,
                                                        k_num,
                                                        m_num_before,
                                                        n_num_before,
                                                        k_num_before,
                                                        n_num_next,
                                                        k_num_next,
                                                        bandwidth,
                                                        compute_pipeline,
                                                        on_chip_alter,
                                                        A_buf_tile_trans,
                                                        B_buf_tile_trans,
                                                        C_buf_tile_trans,
                                                        Y_buf_tile_trans,
                                                        bw_in_use,
                                                        communication_A_cycle,
                                                        communication_B_cycle,
                                                        communication_C_cycle,
                                                        communication_Y_cycle,
                                                        A_load,
                                                        B_load,
                                                        C_load_in,
                                                        C_load_out,
                                                        G_B_ptr_add,
                                                        communication_on_chip,
                                                        communication_interval,
                                                        communication_internal
                                                        )
                        cycle += 1
                        if tracking == 1:
                            writer.writerow({'cycle':str(cycle),'G_B0_ptw':str(G_B0_ptw),'G_B1_ptw':str(G_B1_ptw),'G_B0_ptr':str(G_B0_ptr),'G_B1_ptr':str(G_B1_ptr),\
                            'G_A0_ptw':str(G_A0_ptw),'G_A1_ptw':str(G_A1_ptw),'G_A0_ptr':str(G_A0_ptr),'G_A1_ptr':str(G_A1_ptr),\
                            'G_C0_ptr':str(G_C0_ptr),'G_C1_ptr':str(G_C1_ptr),'G_C0_ptw':str(G_C0_ptw),'G_C1_ptw':str(G_C1_ptw)})
                # 最后一个
                elif m_serial==m_loop_times-1 and k_serial==k_loop_times-1 and n_serial==n_loop_times-1:
                    A_load, B_load, C_load_in, C_load_out = 0, 0, 0, 1
                    A_load0, B_load0, C_load_in0, C_load_out0 = 0, 0, 0, 1
                    while(on_chip_alter == 0):
                        G_A_in_use, G_A0_ready, G_A1_ready, G_A0_ptr, G_A0_ptw, G_A1_ptr, G_A1_ptw, a_loops, b_loops,\
                        G_B0_ready, G_B1_ready, G_B0_ptr, G_B0_ptw, G_B1_ptr, G_B1_ptw, G_B_in_use, B_SA_load,\
                        A_interval_work, A_interval_spare, G_C0_ready, G_C0_ptw, G_C0_ptr, G_C1_ready, G_C1_ptw,\
                        G_C1_ptr, G_Cw_in_use, G_Cr_in_use, cw_loops, cr_loops,\
                        on_chip_alter, bw_in_use, communication_Y_cycle, G_B_ptr_add, communication_on_chip,\
                        communication_interval, communication_internal = tlast.buf_process(G_A_in_use,
                                                        G_A0_ready,
                                                        G_A1_ready,
                                                        G_A0_ptr,
                                                        G_A0_ptw,
                                                        G_A1_ptr,
                                                        G_A1_ptw,
                                                        a_loops,
                                                        b_loops,
                                                        G_B0_ready,
                                                        G_B1_ready,
                                                        G_B0_ptr,
                                                        G_B0_ptw,
                                                        G_B1_ptr,
                                                        G_B1_ptw,
                                                        G_B_in_use,
                                                        B_SA_load,
                                                        A_interval_work,
                                                        A_interval_spare,
                                                        G_C0_ready,
                                                        G_C0_ptw,
                                                        G_C0_ptr,
                                                        G_C1_ready,
                                                        G_C1_ptw,
                                                        G_C1_ptr,
                                                        G_Cw_in_use,
                                                        G_Cr_in_use,
                                                        cw_loops,
                                                        cr_loops,
                                                        Array_i,
                                                        Array_j,
                                                        DR_list,
                                                        SA_list,
                                                        m_num,
                                                        n_num,
                                                        k_num,
                                                        m_num_before,
                                                        n_num_before,
                                                        k_num_before,
                                                        n_num_next,
                                                        k_num_next,
                                                        bandwidth,
                                                        compute_pipeline,
                                                        on_chip_alter,
                                                        A_buf_tile_trans,
                                                        B_buf_tile_trans,
                                                        C_buf_tile_trans,
                                                        Y_buf_tile_trans,
                                                        bw_in_use,
                                                        communication_Y_cycle,
                                                        C_load_out,
                                                        G_B_ptr_add,
                                                        communication_on_chip,
                                                        communication_interval,
                                                        communication_internal
                                                        )
                        cycle += 1
                        if tracking == 1:
                            writer.writerow({'cycle':str(cycle),'G_B0_ptw':str(G_B0_ptw),'G_B1_ptw':str(G_B1_ptw),'G_B0_ptr':str(G_B0_ptr),'G_B1_ptr':str(G_B1_ptr),\
                            'G_A0_ptw':str(G_A0_ptw),'G_A1_ptw':str(G_A1_ptw),'G_A0_ptr':str(G_A0_ptr),'G_A1_ptr':str(G_A1_ptr),\
                            'G_C0_ptr':str(G_C0_ptr),'G_C1_ptr':str(G_C1_ptr),'G_C0_ptw':str(G_C0_ptw),'G_C1_ptw':str(G_C1_ptw)})
                else:
                    A_load, B_load, C_load_in, C_load_out = 0, 1, 1, 1
                    A_load0, B_load0, C_load_in0, C_load_out0 = 0, 1, 1, 1
                    bw_in_use = 'B'
                    if G_B_in_use == 0:
                        G_B1_ready = 0
                    else:
                        G_B0_ready = 0
                    while(on_chip_alter == 0):
                        G_A_in_use, G_A0_ready, G_A1_ready, G_A0_ptr, G_A0_ptw, G_A1_ptr, G_A1_ptw, a_loops, b_loops,\
                        G_B0_ready, G_B1_ready, G_B0_ptr, G_B0_ptw, G_B1_ptr, G_B1_ptw, G_B_in_use, B_SA_load,\
                        A_interval_work, A_interval_spare, G_C0_ready, G_C0_ptw, G_C0_ptr, G_C1_ready, G_C1_ptw,\
                        G_C1_ptr, G_Cw_in_use, G_Cr_in_use, cw_loops, cr_loops,\
                        on_chip_alter, bw_in_use, communication_A_cycle, communication_B_cycle, communication_C_cycle,\
                        communication_Y_cycle, B_load, G_B_ptr_add, communication_on_chip,\
                        communication_interval, communication_internal = t.buf_process(G_A_in_use,
                                                        G_A0_ready,
                                                        G_A1_ready,
                                                        G_A0_ptr,
                                                        G_A0_ptw,
                                                        G_A1_ptr,
                                                        G_A1_ptw,
                                                        a_loops,
                                                        b_loops,
                                                        G_B0_ready,
                                                        G_B1_ready,
                                                        G_B0_ptr,
                                                        G_B0_ptw,
                                                        G_B1_ptr,
                                                        G_B1_ptw,
                                                        G_B_in_use,
                                                        B_SA_load,
                                                        A_interval_work,
                                                        A_interval_spare,
                                                        G_C0_ready,
                                                        G_C0_ptw,
                                                        G_C0_ptr,
                                                        G_C1_ready,
                                                        G_C1_ptw,
                                                        G_C1_ptr,
                                                        G_Cw_in_use,
                                                        G_Cr_in_use,
                                                        cw_loops,
                                                        cr_loops,
                                                        Array_i,
                                                        Array_j,
                                                        DR_list,
                                                        SA_list,
                                                        m_num,
                                                        n_num,
                                                        k_num,
                                                        m_num_before,
                                                        n_num_before,
                                                        k_num_before,
                                                        n_num_next,
                                                        k_num_next,
                                                        bandwidth,
                                                        compute_pipeline,
                                                        on_chip_alter,
                                                        A_buf_tile_trans,
                                                        B_buf_tile_trans,
                                                        C_buf_tile_trans,
                                                        Y_buf_tile_trans,
                                                        bw_in_use,
                                                        communication_A_cycle,
                                                        communication_B_cycle,
                                                        communication_C_cycle,
                                                        communication_Y_cycle,
                                                        A_load,
                                                        B_load,
                                                        C_load_in,
                                                        C_load_out,
                                                        G_B_ptr_add,
                                                        communication_on_chip,
                                                        communication_interval,
                                                        communication_internal
                                                        )
                        cycle += 1
                        if tracking == 1:
                            writer.writerow({'cycle':str(cycle),'G_B0_ptw':str(G_B0_ptw),'G_B1_ptw':str(G_B1_ptw),'G_B0_ptr':str(G_B0_ptr),'G_B1_ptr':str(G_B1_ptr),\
                            'G_A0_ptw':str(G_A0_ptw),'G_A1_ptw':str(G_A1_ptw),'G_A0_ptr':str(G_A0_ptr),'G_A1_ptr':str(G_A1_ptr),\
                            'G_C0_ptr':str(G_C0_ptr),'G_C1_ptr':str(G_C1_ptr),'G_C0_ptw':str(G_C0_ptw),'G_C1_ptw':str(G_C1_ptw)})

                communication_off_chip += A_load0 * f.ceil(m_num_next * k_num_next, bandwidth)\
                + B_load0 * f.ceil(k_num_next * n_num_next, bandwidth) + C_load_in0 * partial_sum[m_serial_next][n_serial_next] * f.ceil(m_num_next * n_num_next, bandwidth)\
                + C_load_out0 * f.ceil(m_num_before * n_num_before, bandwidth)
                # 记录片外的部分和状态，“读入”到片上一次后，下次读取处于存在部分和的状态
                if C_load_in == 1:
                    partial_sum[m_serial_next][n_serial_next] = 1
                print("\r", end="")
                print("Simulation progress: {}%: ".format(int(counter/counter_total * 100)), end="")
                sys.stdout.flush()
if m_loop_times == 1 and k_loop_times >= 2 and n_loop_times == 1:
    for n_serial in range(0,n_loop_times):
        for m_serial in range(0,m_loop_times):
            for k_serial in range(0,k_loop_times):
                counter += 1
                # 计算本次片上的mkn规模大小
                m_num, k_num, n_num = f.mkn_num(m_serial, k_serial, n_serial, m_loop_times, k_loop_times, n_loop_times, M, K, N, m, k, n)
                m_serial_next, k_serial_next, n_serial_next = f.next_num_nmk(m_serial, k_serial, n_serial, m_loop_times, k_loop_times, n_loop_times)
                m_serial_before, k_serial_before, n_serial_before = f.before_num_nmk(m_serial, k_serial, n_serial, m_loop_times, k_loop_times, n_loop_times)
                m_num_next, k_num_next, n_num_next = f.mkn_num(m_serial_next, k_serial_next, n_serial_next, m_loop_times, k_loop_times, n_loop_times, M, K, N, m, k, n)
                m_num_before, k_num_before, n_num_before = f.mkn_num(m_serial_before, k_serial_before, n_serial_before, m_loop_times, k_loop_times, n_loop_times, M, K, N, m, k, n)

                m_coordinate, k_coordinate, n_coordinate = m_serial_next * m, k_serial_next * k, n_serial_next * n

                A_buf_tile_trans = f.ceil(m_num_next * k_num_next, bandwidth)
                B_buf_tile_trans = f.ceil(k_num_next * n_num_next, bandwidth)
                # 判断C是否要从片外读入，第一次是不需要读入的
                if partial_sum[m_serial_next][n_serial_next] == 0:
                    C_buf_tile_trans = 0
                else:
                    C_buf_tile_trans = f.ceil(m_num_next * n_num_next, bandwidth)
                Y_buf_tile_trans = f.ceil(m_num_before * n_num_before, bandwidth)
                on_chip_alter = 0
                # 最后一个
                if m_serial==m_loop_times-1 and k_serial==k_loop_times-1 and n_serial==n_loop_times-1:
                    A_load, B_load, C_load_in, C_load_out = 0, 0, 0, 0
                    A_load0, B_load0, C_load_in0, C_load_out0 = 0, 0, 0, 0
                    while(on_chip_alter == 0):
                        G_A_in_use, G_A0_ready, G_A1_ready, G_A0_ptr, G_A0_ptw, G_A1_ptr, G_A1_ptw, a_loops, b_loops,\
                        G_B0_ready, G_B1_ready, G_B0_ptr, G_B0_ptw, G_B1_ptr, G_B1_ptw, G_B_in_use, B_SA_load,\
                        A_interval_work, A_interval_spare, G_C0_ready, G_C0_ptw, G_C0_ptr, G_C1_ready, G_C1_ptw,\
                        G_C1_ptr, G_Cw_in_use, G_Cr_in_use, cw_loops, cr_loops,\
                        on_chip_alter, bw_in_use, communication_Y_cycle, G_B_ptr_add, communication_on_chip,\
                        communication_interval, communication_internal = tlast.buf_process(G_A_in_use,
                                                        G_A0_ready,
                                                        G_A1_ready,
                                                        G_A0_ptr,
                                                        G_A0_ptw,
                                                        G_A1_ptr,
                                                        G_A1_ptw,
                                                        a_loops,
                                                        b_loops,
                                                        G_B0_ready,
                                                        G_B1_ready,
                                                        G_B0_ptr,
                                                        G_B0_ptw,
                                                        G_B1_ptr,
                                                        G_B1_ptw,
                                                        G_B_in_use,
                                                        B_SA_load,
                                                        A_interval_work,
                                                        A_interval_spare,
                                                        G_C0_ready,
                                                        G_C0_ptw,
                                                        G_C0_ptr,
                                                        G_C1_ready,
                                                        G_C1_ptw,
                                                        G_C1_ptr,
                                                        G_Cw_in_use,
                                                        G_Cr_in_use,
                                                        cw_loops,
                                                        cr_loops,
                                                        Array_i,
                                                        Array_j,
                                                        DR_list,
                                                        SA_list,
                                                        m_num,
                                                        n_num,
                                                        k_num,
                                                        m_num_before,
                                                        n_num_before,
                                                        k_num_before,
                                                        n_num_next,
                                                        k_num_next,
                                                        bandwidth,
                                                        compute_pipeline,
                                                        on_chip_alter,
                                                        A_buf_tile_trans,
                                                        B_buf_tile_trans,
                                                        C_buf_tile_trans,
                                                        Y_buf_tile_trans,
                                                        bw_in_use,
                                                        communication_Y_cycle,
                                                        C_load_out,
                                                        G_B_ptr_add,
                                                        communication_on_chip,
                                                        communication_interval,
                                                        communication_internal
                                                        )
                        cycle += 1
                        if tracking == 1:
                            writer.writerow({'cycle':str(cycle),'G_B0_ptw':str(G_B0_ptw),'G_B1_ptw':str(G_B1_ptw),'G_B0_ptr':str(G_B0_ptr),'G_B1_ptr':str(G_B1_ptr),\
                            'G_A0_ptw':str(G_A0_ptw),'G_A1_ptw':str(G_A1_ptw),'G_A0_ptr':str(G_A0_ptr),'G_A1_ptr':str(G_A1_ptr),\
                            'G_C0_ptr':str(G_C0_ptr),'G_C1_ptr':str(G_C1_ptr),'G_C0_ptw':str(G_C0_ptw),'G_C1_ptw':str(G_C1_ptw)})
                else:
                    A_load, B_load, C_load_in, C_load_out = 1, 1, 0, 0
                    A_load0, B_load0, C_load_in0, C_load_out0 = 1, 1, 0, 0
                    bw_in_use = 'B'
                    if G_B_in_use == 0:
                        G_B1_ready = 0
                    else:
                        G_B0_ready = 0
                    while(on_chip_alter == 0):
                        G_A_in_use, G_A0_ready, G_A1_ready, G_A0_ptr, G_A0_ptw, G_A1_ptr, G_A1_ptw, a_loops, b_loops,\
                        G_B0_ready, G_B1_ready, G_B0_ptr, G_B0_ptw, G_B1_ptr, G_B1_ptw, G_B_in_use, B_SA_load,\
                        A_interval_work, A_interval_spare, G_C0_ready, G_C0_ptw, G_C0_ptr, G_C1_ready, G_C1_ptw,\
                        G_C1_ptr, G_Cw_in_use, G_Cr_in_use, cw_loops, cr_loops,\
                        on_chip_alter, bw_in_use, communication_A_cycle, communication_B_cycle, communication_C_cycle,\
                        communication_Y_cycle, B_load, G_B_ptr_add, communication_on_chip,\
                        communication_interval, communication_internal = t.buf_process(G_A_in_use,
                                                        G_A0_ready,
                                                        G_A1_ready,
                                                        G_A0_ptr,
                                                        G_A0_ptw,
                                                        G_A1_ptr,
                                                        G_A1_ptw,
                                                        a_loops,
                                                        b_loops,
                                                        G_B0_ready,
                                                        G_B1_ready,
                                                        G_B0_ptr,
                                                        G_B0_ptw,
                                                        G_B1_ptr,
                                                        G_B1_ptw,
                                                        G_B_in_use,
                                                        B_SA_load,
                                                        A_interval_work,
                                                        A_interval_spare,
                                                        G_C0_ready,
                                                        G_C0_ptw,
                                                        G_C0_ptr,
                                                        G_C1_ready,
                                                        G_C1_ptw,
                                                        G_C1_ptr,
                                                        G_Cw_in_use,
                                                        G_Cr_in_use,
                                                        cw_loops,
                                                        cr_loops,
                                                        Array_i,
                                                        Array_j,
                                                        DR_list,
                                                        SA_list,
                                                        m_num,
                                                        n_num,
                                                        k_num,
                                                        m_num_before,
                                                        n_num_before,
                                                        k_num_before,
                                                        n_num_next,
                                                        k_num_next,
                                                        bandwidth,
                                                        compute_pipeline,
                                                        on_chip_alter,
                                                        A_buf_tile_trans,
                                                        B_buf_tile_trans,
                                                        C_buf_tile_trans,
                                                        Y_buf_tile_trans,
                                                        bw_in_use,
                                                        communication_A_cycle,
                                                        communication_B_cycle,
                                                        communication_C_cycle,
                                                        communication_Y_cycle,
                                                        A_load,
                                                        B_load,
                                                        C_load_in,
                                                        C_load_out,
                                                        G_B_ptr_add,
                                                        communication_on_chip,
                                                        communication_interval,
                                                        communication_internal
                                                        )
                        cycle += 1
                        if tracking == 1:
                            writer.writerow({'cycle':str(cycle),'G_B0_ptw':str(G_B0_ptw),'G_B1_ptw':str(G_B1_ptw),'G_B0_ptr':str(G_B0_ptr),'G_B1_ptr':str(G_B1_ptr),\
                            'G_A0_ptw':str(G_A0_ptw),'G_A1_ptw':str(G_A1_ptw),'G_A0_ptr':str(G_A0_ptr),'G_A1_ptr':str(G_A1_ptr),\
                            'G_C0_ptr':str(G_C0_ptr),'G_C1_ptr':str(G_C1_ptr),'G_C0_ptw':str(G_C0_ptw),'G_C1_ptw':str(G_C1_ptw)})

                communication_off_chip += A_load0 * f.ceil(m_num_next * k_num_next, bandwidth)\
                + B_load0 * f.ceil(k_num_next * n_num_next, bandwidth) + C_load_in0 * partial_sum[m_serial_next][n_serial_next] * f.ceil(m_num_next * n_num_next, bandwidth)\
                + C_load_out0 * f.ceil(m_num_before * n_num_before, bandwidth)
                # 记录片外的部分和状态，“读入”到片上一次后，下次读取处于存在部分和的状态
                if C_load_in == 1:
                    partial_sum[m_serial_next][n_serial_next] = 1
                print("\r", end="")
                print("Simulation progress: {}%: ".format(int(counter/counter_total * 100)), end="")
                sys.stdout.flush()
if m_loop_times == 1 and k_loop_times == 2 and n_loop_times >= 2:
    for n_serial in range(0,n_loop_times):
        for m_serial in range(0,m_loop_times):
            for k_serial in range(0,k_loop_times):
                counter += 1
                # 计算本次片上的mkn规模大小
                m_num, k_num, n_num = f.mkn_num(m_serial, k_serial, n_serial, m_loop_times, k_loop_times, n_loop_times, M, K, N, m, k, n)
                m_serial_next, k_serial_next, n_serial_next = f.next_num_nmk(m_serial, k_serial, n_serial, m_loop_times, k_loop_times, n_loop_times)
                m_serial_before, k_serial_before, n_serial_before = f.before_num_nmk(m_serial, k_serial, n_serial, m_loop_times, k_loop_times, n_loop_times)
                m_num_next, k_num_next, n_num_next = f.mkn_num(m_serial_next, k_serial_next, n_serial_next, m_loop_times, k_loop_times, n_loop_times, M, K, N, m, k, n)
                m_num_before, k_num_before, n_num_before = f.mkn_num(m_serial_before, k_serial_before, n_serial_before, m_loop_times, k_loop_times, n_loop_times, M, K, N, m, k, n)

                m_coordinate, k_coordinate, n_coordinate = m_serial_next * m, k_serial_next * k, n_serial_next * n

                A_buf_tile_trans = f.ceil(m_num_next * k_num_next, bandwidth)
                B_buf_tile_trans = f.ceil(k_num_next * n_num_next, bandwidth)
                # 判断C是否要从片外读入，第一次是不需要读入的
                if partial_sum[m_serial_next][n_serial_next] == 0:
                    C_buf_tile_trans = 0
                else:
                    C_buf_tile_trans = f.ceil(m_num_next * n_num_next, bandwidth)
                Y_buf_tile_trans = f.ceil(m_num_before * n_num_before, bandwidth)
                on_chip_alter = 0
                # 第一个
                if m_serial==0 and k_serial==0 and n_serial==0:
                    A_load, B_load, C_load_in, C_load_out = 1, 1, 0, 0
                    A_load0, B_load0, C_load_in0, C_load_out0 = 1, 1, 0, 0
                    while(on_chip_alter == 0):
                        G_A_in_use, G_A0_ready, G_A1_ready, G_A0_ptr, G_A0_ptw, G_A1_ptr, G_A1_ptw, a_loops, b_loops,\
                        G_B0_ready, G_B1_ready, G_B0_ptr, G_B0_ptw, G_B1_ptr, G_B1_ptw, G_B_in_use, B_SA_load,\
                        A_interval_work, A_interval_spare, G_C0_ready, G_C0_ptw, G_C0_ptr, G_C1_ready, G_C1_ptw,\
                        G_C1_ptr, G_Cw_in_use, G_Cr_in_use, cw_loops, cr_loops,\
                        on_chip_alter, bw_in_use, communication_A_cycle, communication_B_cycle, communication_C_cycle,\
                        communication_Y_cycle, B_load, G_B_ptr_add, communication_on_chip,\
                        communication_interval, communication_internal = t.buf_process(G_A_in_use,
                                                        G_A0_ready,
                                                        G_A1_ready,
                                                        G_A0_ptr,
                                                        G_A0_ptw,
                                                        G_A1_ptr,
                                                        G_A1_ptw,
                                                        a_loops,
                                                        b_loops,
                                                        G_B0_ready,
                                                        G_B1_ready,
                                                        G_B0_ptr,
                                                        G_B0_ptw,
                                                        G_B1_ptr,
                                                        G_B1_ptw,
                                                        G_B_in_use,
                                                        B_SA_load,
                                                        A_interval_work,
                                                        A_interval_spare,
                                                        G_C0_ready,
                                                        G_C0_ptw,
                                                        G_C0_ptr,
                                                        G_C1_ready,
                                                        G_C1_ptw,
                                                        G_C1_ptr,
                                                        G_Cw_in_use,
                                                        G_Cr_in_use,
                                                        cw_loops,
                                                        cr_loops,
                                                        Array_i,
                                                        Array_j,
                                                        DR_list,
                                                        SA_list,
                                                        m_num,
                                                        n_num,
                                                        k_num,
                                                        m_num_before,
                                                        n_num_before,
                                                        k_num_before,
                                                        n_num_next,
                                                        k_num_next,
                                                        bandwidth,
                                                        compute_pipeline,
                                                        on_chip_alter,
                                                        A_buf_tile_trans,
                                                        B_buf_tile_trans,
                                                        C_buf_tile_trans,
                                                        Y_buf_tile_trans,
                                                        bw_in_use,
                                                        communication_A_cycle,
                                                        communication_B_cycle,
                                                        communication_C_cycle,
                                                        communication_Y_cycle,
                                                        A_load,
                                                        B_load,
                                                        C_load_in,
                                                        C_load_out,
                                                        G_B_ptr_add,
                                                        communication_on_chip,
                                                        communication_interval,
                                                        communication_internal
                                                        )
                        cycle += 1
                        if tracking == 1:
                            writer.writerow({'cycle':str(cycle),'G_B0_ptw':str(G_B0_ptw),'G_B1_ptw':str(G_B1_ptw),'G_B0_ptr':str(G_B0_ptr),'G_B1_ptr':str(G_B1_ptr),\
                            'G_A0_ptw':str(G_A0_ptw),'G_A1_ptw':str(G_A1_ptw),'G_A0_ptr':str(G_A0_ptr),'G_A1_ptr':str(G_A1_ptr),\
                            'G_C0_ptr':str(G_C0_ptr),'G_C1_ptr':str(G_C1_ptr),'G_C0_ptw':str(G_C0_ptw),'G_C1_ptw':str(G_C1_ptw)})
                # 最后一个
                elif m_serial==m_loop_times-1 and k_serial==k_loop_times-1 and n_serial==n_loop_times-1:
                    A_load, B_load, C_load_in, C_load_out = 0, 0, 0, 1
                    A_load0, B_load0, C_load_in0, C_load_out0 = 0, 0, 0, 1
                    while(on_chip_alter == 0):
                        G_A_in_use, G_A0_ready, G_A1_ready, G_A0_ptr, G_A0_ptw, G_A1_ptr, G_A1_ptw, a_loops, b_loops,\
                        G_B0_ready, G_B1_ready, G_B0_ptr, G_B0_ptw, G_B1_ptr, G_B1_ptw, G_B_in_use, B_SA_load,\
                        A_interval_work, A_interval_spare, G_C0_ready, G_C0_ptw, G_C0_ptr, G_C1_ready, G_C1_ptw,\
                        G_C1_ptr, G_Cw_in_use, G_Cr_in_use, cw_loops, cr_loops,\
                        on_chip_alter, bw_in_use, communication_Y_cycle, G_B_ptr_add, communication_on_chip,\
                        communication_interval, communication_internal = tlast.buf_process(G_A_in_use,
                                                        G_A0_ready,
                                                        G_A1_ready,
                                                        G_A0_ptr,
                                                        G_A0_ptw,
                                                        G_A1_ptr,
                                                        G_A1_ptw,
                                                        a_loops,
                                                        b_loops,
                                                        G_B0_ready,
                                                        G_B1_ready,
                                                        G_B0_ptr,
                                                        G_B0_ptw,
                                                        G_B1_ptr,
                                                        G_B1_ptw,
                                                        G_B_in_use,
                                                        B_SA_load,
                                                        A_interval_work,
                                                        A_interval_spare,
                                                        G_C0_ready,
                                                        G_C0_ptw,
                                                        G_C0_ptr,
                                                        G_C1_ready,
                                                        G_C1_ptw,
                                                        G_C1_ptr,
                                                        G_Cw_in_use,
                                                        G_Cr_in_use,
                                                        cw_loops,
                                                        cr_loops,
                                                        Array_i,
                                                        Array_j,
                                                        DR_list,
                                                        SA_list,
                                                        m_num,
                                                        n_num,
                                                        k_num,
                                                        m_num_before,
                                                        n_num_before,
                                                        k_num_before,
                                                        n_num_next,
                                                        k_num_next,
                                                        bandwidth,
                                                        compute_pipeline,
                                                        on_chip_alter,
                                                        A_buf_tile_trans,
                                                        B_buf_tile_trans,
                                                        C_buf_tile_trans,
                                                        Y_buf_tile_trans,
                                                        bw_in_use,
                                                        communication_Y_cycle,
                                                        C_load_out,
                                                        G_B_ptr_add,
                                                        communication_on_chip,
                                                        communication_interval,
                                                        communication_internal
                                                        )
                        cycle += 1
                        if tracking == 1:
                            writer.writerow({'cycle':str(cycle),'G_B0_ptw':str(G_B0_ptw),'G_B1_ptw':str(G_B1_ptw),'G_B0_ptr':str(G_B0_ptr),'G_B1_ptr':str(G_B1_ptr),\
                            'G_A0_ptw':str(G_A0_ptw),'G_A1_ptw':str(G_A1_ptw),'G_A0_ptr':str(G_A0_ptr),'G_A1_ptr':str(G_A1_ptr),\
                            'G_C0_ptr':str(G_C0_ptr),'G_C1_ptr':str(G_C1_ptr),'G_C0_ptw':str(G_C0_ptw),'G_C1_ptw':str(G_C1_ptw)})
                elif m_serial==0 and k_serial==0 and n_serial!=0:
                    A_load, B_load, C_load_in, C_load_out = 0, 1, 0, 0
                    A_load0, B_load0, C_load_in0, C_load_out0 = 0, 1, 0, 0
                    bw_in_use = 'B'
                    if G_B_in_use == 0:
                        G_B1_ready = 0
                    else:
                        G_B0_ready = 0
                    while(on_chip_alter == 0):
                        G_A_in_use, G_A0_ready, G_A1_ready, G_A0_ptr, G_A0_ptw, G_A1_ptr, G_A1_ptw, a_loops, b_loops,\
                        G_B0_ready, G_B1_ready, G_B0_ptr, G_B0_ptw, G_B1_ptr, G_B1_ptw, G_B_in_use, B_SA_load,\
                        A_interval_work, A_interval_spare, G_C0_ready, G_C0_ptw, G_C0_ptr, G_C1_ready, G_C1_ptw,\
                        G_C1_ptr, G_Cw_in_use, G_Cr_in_use, cw_loops, cr_loops,\
                        on_chip_alter, bw_in_use, communication_A_cycle, communication_B_cycle, communication_C_cycle,\
                        communication_Y_cycle, B_load, G_B_ptr_add, communication_on_chip,\
                        communication_interval, communication_internal = t.buf_process(G_A_in_use,
                                                        G_A0_ready,
                                                        G_A1_ready,
                                                        G_A0_ptr,
                                                        G_A0_ptw,
                                                        G_A1_ptr,
                                                        G_A1_ptw,
                                                        a_loops,
                                                        b_loops,
                                                        G_B0_ready,
                                                        G_B1_ready,
                                                        G_B0_ptr,
                                                        G_B0_ptw,
                                                        G_B1_ptr,
                                                        G_B1_ptw,
                                                        G_B_in_use,
                                                        B_SA_load,
                                                        A_interval_work,
                                                        A_interval_spare,
                                                        G_C0_ready,
                                                        G_C0_ptw,
                                                        G_C0_ptr,
                                                        G_C1_ready,
                                                        G_C1_ptw,
                                                        G_C1_ptr,
                                                        G_Cw_in_use,
                                                        G_Cr_in_use,
                                                        cw_loops,
                                                        cr_loops,
                                                        Array_i,
                                                        Array_j,
                                                        DR_list,
                                                        SA_list,
                                                        m_num,
                                                        n_num,
                                                        k_num,
                                                        m_num_before,
                                                        n_num_before,
                                                        k_num_before,
                                                        n_num_next,
                                                        k_num_next,
                                                        bandwidth,
                                                        compute_pipeline,
                                                        on_chip_alter,
                                                        A_buf_tile_trans,
                                                        B_buf_tile_trans,
                                                        C_buf_tile_trans,
                                                        Y_buf_tile_trans,
                                                        bw_in_use,
                                                        communication_A_cycle,
                                                        communication_B_cycle,
                                                        communication_C_cycle,
                                                        communication_Y_cycle,
                                                        A_load,
                                                        B_load,
                                                        C_load_in,
                                                        C_load_out,
                                                        G_B_ptr_add,
                                                        communication_on_chip,
                                                        communication_interval,
                                                        communication_internal
                                                        )
                        cycle += 1
                        if tracking == 1:
                            writer.writerow({'cycle':str(cycle),'G_B0_ptw':str(G_B0_ptw),'G_B1_ptw':str(G_B1_ptw),'G_B0_ptr':str(G_B0_ptr),'G_B1_ptr':str(G_B1_ptr),\
                            'G_A0_ptw':str(G_A0_ptw),'G_A1_ptw':str(G_A1_ptw),'G_A0_ptr':str(G_A0_ptr),'G_A1_ptr':str(G_A1_ptr),\
                            'G_C0_ptr':str(G_C0_ptr),'G_C1_ptr':str(G_C1_ptr),'G_C0_ptw':str(G_C0_ptw),'G_C1_ptw':str(G_C1_ptw)})
                elif m_serial==0 and k_serial==k_loop_times-1 and n_serial==0:
                    A_load, B_load, C_load_in, C_load_out = 0, 1, 1, 0
                    A_load0, B_load0, C_load_in0, C_load_out0 = 0, 1, 1, 0
                    bw_in_use = 'B'
                    if G_B_in_use == 0:
                        G_B1_ready = 0
                    else:
                        G_B0_ready = 0
                    while(on_chip_alter == 0):
                        G_A_in_use, G_A0_ready, G_A1_ready, G_A0_ptr, G_A0_ptw, G_A1_ptr, G_A1_ptw, a_loops, b_loops,\
                        G_B0_ready, G_B1_ready, G_B0_ptr, G_B0_ptw, G_B1_ptr, G_B1_ptw, G_B_in_use, B_SA_load,\
                        A_interval_work, A_interval_spare, G_C0_ready, G_C0_ptw, G_C0_ptr, G_C1_ready, G_C1_ptw,\
                        G_C1_ptr, G_Cw_in_use, G_Cr_in_use, cw_loops, cr_loops,\
                        on_chip_alter, bw_in_use, communication_A_cycle, communication_B_cycle, communication_C_cycle,\
                        communication_Y_cycle, B_load, G_B_ptr_add, communication_on_chip,\
                        communication_interval, communication_internal = t.buf_process(G_A_in_use,
                                                        G_A0_ready,
                                                        G_A1_ready,
                                                        G_A0_ptr,
                                                        G_A0_ptw,
                                                        G_A1_ptr,
                                                        G_A1_ptw,
                                                        a_loops,
                                                        b_loops,
                                                        G_B0_ready,
                                                        G_B1_ready,
                                                        G_B0_ptr,
                                                        G_B0_ptw,
                                                        G_B1_ptr,
                                                        G_B1_ptw,
                                                        G_B_in_use,
                                                        B_SA_load,
                                                        A_interval_work,
                                                        A_interval_spare,
                                                        G_C0_ready,
                                                        G_C0_ptw,
                                                        G_C0_ptr,
                                                        G_C1_ready,
                                                        G_C1_ptw,
                                                        G_C1_ptr,
                                                        G_Cw_in_use,
                                                        G_Cr_in_use,
                                                        cw_loops,
                                                        cr_loops,
                                                        Array_i,
                                                        Array_j,
                                                        DR_list,
                                                        SA_list,
                                                        m_num,
                                                        n_num,
                                                        k_num,
                                                        m_num_before,
                                                        n_num_before,
                                                        k_num_before,
                                                        n_num_next,
                                                        k_num_next,
                                                        bandwidth,
                                                        compute_pipeline,
                                                        on_chip_alter,
                                                        A_buf_tile_trans,
                                                        B_buf_tile_trans,
                                                        C_buf_tile_trans,
                                                        Y_buf_tile_trans,
                                                        bw_in_use,
                                                        communication_A_cycle,
                                                        communication_B_cycle,
                                                        communication_C_cycle,
                                                        communication_Y_cycle,
                                                        A_load,
                                                        B_load,
                                                        C_load_in,
                                                        C_load_out,
                                                        G_B_ptr_add,
                                                        communication_on_chip,
                                                        communication_interval,
                                                        communication_internal
                                                        )
                        cycle += 1
                        if tracking == 1:
                            writer.writerow({'cycle':str(cycle),'G_B0_ptw':str(G_B0_ptw),'G_B1_ptw':str(G_B1_ptw),'G_B0_ptr':str(G_B0_ptr),'G_B1_ptr':str(G_B1_ptr),\
                            'G_A0_ptw':str(G_A0_ptw),'G_A1_ptw':str(G_A1_ptw),'G_A0_ptr':str(G_A0_ptr),'G_A1_ptr':str(G_A1_ptr),\
                            'G_C0_ptr':str(G_C0_ptr),'G_C1_ptr':str(G_C1_ptr),'G_C0_ptw':str(G_C0_ptw),'G_C1_ptw':str(G_C1_ptw)})
                else:
                    A_load, B_load, C_load_in, C_load_out = 0, 1, 1, 1
                    A_load0, B_load0, C_load_in0, C_load_out0 = 0, 1, 1, 1
                    bw_in_use = 'B'
                    if G_B_in_use == 0:
                        G_B1_ready = 0
                    else:
                        G_B0_ready = 0
                    while(on_chip_alter == 0):
                        G_A_in_use, G_A0_ready, G_A1_ready, G_A0_ptr, G_A0_ptw, G_A1_ptr, G_A1_ptw, a_loops, b_loops,\
                        G_B0_ready, G_B1_ready, G_B0_ptr, G_B0_ptw, G_B1_ptr, G_B1_ptw, G_B_in_use, B_SA_load,\
                        A_interval_work, A_interval_spare, G_C0_ready, G_C0_ptw, G_C0_ptr, G_C1_ready, G_C1_ptw,\
                        G_C1_ptr, G_Cw_in_use, G_Cr_in_use, cw_loops, cr_loops,\
                        on_chip_alter, bw_in_use, communication_A_cycle, communication_B_cycle, communication_C_cycle,\
                        communication_Y_cycle, B_load, G_B_ptr_add, communication_on_chip,\
                        communication_interval, communication_internal = t.buf_process(G_A_in_use,
                                                        G_A0_ready,
                                                        G_A1_ready,
                                                        G_A0_ptr,
                                                        G_A0_ptw,
                                                        G_A1_ptr,
                                                        G_A1_ptw,
                                                        a_loops,
                                                        b_loops,
                                                        G_B0_ready,
                                                        G_B1_ready,
                                                        G_B0_ptr,
                                                        G_B0_ptw,
                                                        G_B1_ptr,
                                                        G_B1_ptw,
                                                        G_B_in_use,
                                                        B_SA_load,
                                                        A_interval_work,
                                                        A_interval_spare,
                                                        G_C0_ready,
                                                        G_C0_ptw,
                                                        G_C0_ptr,
                                                        G_C1_ready,
                                                        G_C1_ptw,
                                                        G_C1_ptr,
                                                        G_Cw_in_use,
                                                        G_Cr_in_use,
                                                        cw_loops,
                                                        cr_loops,
                                                        Array_i,
                                                        Array_j,
                                                        DR_list,
                                                        SA_list,
                                                        m_num,
                                                        n_num,
                                                        k_num,
                                                        m_num_before,
                                                        n_num_before,
                                                        k_num_before,
                                                        n_num_next,
                                                        k_num_next,
                                                        bandwidth,
                                                        compute_pipeline,
                                                        on_chip_alter,
                                                        A_buf_tile_trans,
                                                        B_buf_tile_trans,
                                                        C_buf_tile_trans,
                                                        Y_buf_tile_trans,
                                                        bw_in_use,
                                                        communication_A_cycle,
                                                        communication_B_cycle,
                                                        communication_C_cycle,
                                                        communication_Y_cycle,
                                                        A_load,
                                                        B_load,
                                                        C_load_in,
                                                        C_load_out,
                                                        G_B_ptr_add,
                                                        communication_on_chip,
                                                        communication_interval,
                                                        communication_internal
                                                        )
                        cycle += 1
                        if tracking == 1:
                            writer.writerow({'cycle':str(cycle),'G_B0_ptw':str(G_B0_ptw),'G_B1_ptw':str(G_B1_ptw),'G_B0_ptr':str(G_B0_ptr),'G_B1_ptr':str(G_B1_ptr),\
                            'G_A0_ptw':str(G_A0_ptw),'G_A1_ptw':str(G_A1_ptw),'G_A0_ptr':str(G_A0_ptr),'G_A1_ptr':str(G_A1_ptr),\
                            'G_C0_ptr':str(G_C0_ptr),'G_C1_ptr':str(G_C1_ptr),'G_C0_ptw':str(G_C0_ptw),'G_C1_ptw':str(G_C1_ptw)})

                communication_off_chip += A_load0 * f.ceil(m_num_next * k_num_next, bandwidth)\
                + B_load0 * f.ceil(k_num_next * n_num_next, bandwidth) + C_load_in0 * partial_sum[m_serial_next][n_serial_next] * f.ceil(m_num_next * n_num_next, bandwidth)\
                + C_load_out0 * f.ceil(m_num_before * n_num_before, bandwidth)
                # 记录片外的部分和状态，“读入”到片上一次后，下次读取处于存在部分和的状态
                if C_load_in == 1:
                    partial_sum[m_serial_next][n_serial_next] = 1
                print("\r", end="")
                print("Simulation progress: {}%: ".format(int(counter/counter_total * 100)), end="")
                sys.stdout.flush()
if m_loop_times == 1 and k_loop_times >= 3 and n_loop_times == 2:
    for n_serial in range(0,n_loop_times):
        for m_serial in range(0,m_loop_times):
            for k_serial in range(0,k_loop_times):
                counter += 1
                # 计算本次片上的mkn规模大小
                m_num, k_num, n_num = f.mkn_num(m_serial, k_serial, n_serial, m_loop_times, k_loop_times, n_loop_times, M, K, N, m, k, n)
                m_serial_next, k_serial_next, n_serial_next = f.next_num_nmk(m_serial, k_serial, n_serial, m_loop_times, k_loop_times, n_loop_times)
                m_serial_before, k_serial_before, n_serial_before = f.before_num_nmk(m_serial, k_serial, n_serial, m_loop_times, k_loop_times, n_loop_times)
                m_num_next, k_num_next, n_num_next = f.mkn_num(m_serial_next, k_serial_next, n_serial_next, m_loop_times, k_loop_times, n_loop_times, M, K, N, m, k, n)
                m_num_before, k_num_before, n_num_before = f.mkn_num(m_serial_before, k_serial_before, n_serial_before, m_loop_times, k_loop_times, n_loop_times, M, K, N, m, k, n)

                m_coordinate, k_coordinate, n_coordinate = m_serial_next * m, k_serial_next * k, n_serial_next * n

                A_buf_tile_trans = f.ceil(m_num_next * k_num_next, bandwidth)
                B_buf_tile_trans = f.ceil(k_num_next * n_num_next, bandwidth)
                # 判断C是否要从片外读入，第一次是不需要读入的
                if partial_sum[m_serial_next][n_serial_next] == 0:
                    C_buf_tile_trans = 0
                else:
                    C_buf_tile_trans = f.ceil(m_num_next * n_num_next, bandwidth)
                Y_buf_tile_trans = f.ceil(m_num_before * n_num_before, bandwidth)
                on_chip_alter = 0
                # 第一个
                if m_serial==0 and k_serial==k_loop_times-1 and n_serial==0:
                    A_load, B_load, C_load_in, C_load_out = 1, 1, 1, 0
                    A_load0, B_load0, C_load_in0, C_load_out0 = 1, 1, 1, 0
                    bw_in_use = 'B'
                    if G_B_in_use == 0:
                        G_B1_ready = 0
                    else:
                        G_B0_ready = 0
                    while(on_chip_alter == 0):
                        G_A_in_use, G_A0_ready, G_A1_ready, G_A0_ptr, G_A0_ptw, G_A1_ptr, G_A1_ptw, a_loops, b_loops,\
                        G_B0_ready, G_B1_ready, G_B0_ptr, G_B0_ptw, G_B1_ptr, G_B1_ptw, G_B_in_use, B_SA_load,\
                        A_interval_work, A_interval_spare, G_C0_ready, G_C0_ptw, G_C0_ptr, G_C1_ready, G_C1_ptw,\
                        G_C1_ptr, G_Cw_in_use, G_Cr_in_use, cw_loops, cr_loops,\
                        on_chip_alter, bw_in_use, communication_A_cycle, communication_B_cycle, communication_C_cycle,\
                        communication_Y_cycle, B_load, G_B_ptr_add, communication_on_chip,\
                        communication_interval, communication_internal = t.buf_process(G_A_in_use,
                                                        G_A0_ready,
                                                        G_A1_ready,
                                                        G_A0_ptr,
                                                        G_A0_ptw,
                                                        G_A1_ptr,
                                                        G_A1_ptw,
                                                        a_loops,
                                                        b_loops,
                                                        G_B0_ready,
                                                        G_B1_ready,
                                                        G_B0_ptr,
                                                        G_B0_ptw,
                                                        G_B1_ptr,
                                                        G_B1_ptw,
                                                        G_B_in_use,
                                                        B_SA_load,
                                                        A_interval_work,
                                                        A_interval_spare,
                                                        G_C0_ready,
                                                        G_C0_ptw,
                                                        G_C0_ptr,
                                                        G_C1_ready,
                                                        G_C1_ptw,
                                                        G_C1_ptr,
                                                        G_Cw_in_use,
                                                        G_Cr_in_use,
                                                        cw_loops,
                                                        cr_loops,
                                                        Array_i,
                                                        Array_j,
                                                        DR_list,
                                                        SA_list,
                                                        m_num,
                                                        n_num,
                                                        k_num,
                                                        m_num_before,
                                                        n_num_before,
                                                        k_num_before,
                                                        n_num_next,
                                                        k_num_next,
                                                        bandwidth,
                                                        compute_pipeline,
                                                        on_chip_alter,
                                                        A_buf_tile_trans,
                                                        B_buf_tile_trans,
                                                        C_buf_tile_trans,
                                                        Y_buf_tile_trans,
                                                        bw_in_use,
                                                        communication_A_cycle,
                                                        communication_B_cycle,
                                                        communication_C_cycle,
                                                        communication_Y_cycle,
                                                        A_load,
                                                        B_load,
                                                        C_load_in,
                                                        C_load_out,
                                                        G_B_ptr_add,
                                                        communication_on_chip,
                                                        communication_interval,
                                                        communication_internal
                                                        )
                        cycle += 1
                        if tracking == 1:
                            writer.writerow({'cycle':str(cycle),'G_B0_ptw':str(G_B0_ptw),'G_B1_ptw':str(G_B1_ptw),'G_B0_ptr':str(G_B0_ptr),'G_B1_ptr':str(G_B1_ptr),\
                            'G_A0_ptw':str(G_A0_ptw),'G_A1_ptw':str(G_A1_ptw),'G_A0_ptr':str(G_A0_ptr),'G_A1_ptr':str(G_A1_ptr),\
                            'G_C0_ptr':str(G_C0_ptr),'G_C1_ptr':str(G_C1_ptr),'G_C0_ptw':str(G_C0_ptw),'G_C1_ptw':str(G_C1_ptw)})
                # 最后一个
                elif m_serial==m_loop_times-1 and k_serial==k_loop_times-1 and n_serial==n_loop_times-1:
                    A_load, B_load, C_load_in, C_load_out = 0, 0, 0, 1
                    A_load0, B_load0, C_load_in0, C_load_out0 = 0, 0, 0, 1
                    while(on_chip_alter == 0):
                        G_A_in_use, G_A0_ready, G_A1_ready, G_A0_ptr, G_A0_ptw, G_A1_ptr, G_A1_ptw, a_loops, b_loops,\
                        G_B0_ready, G_B1_ready, G_B0_ptr, G_B0_ptw, G_B1_ptr, G_B1_ptw, G_B_in_use, B_SA_load,\
                        A_interval_work, A_interval_spare, G_C0_ready, G_C0_ptw, G_C0_ptr, G_C1_ready, G_C1_ptw,\
                        G_C1_ptr, G_Cw_in_use, G_Cr_in_use, cw_loops, cr_loops,\
                        on_chip_alter, bw_in_use, communication_Y_cycle, G_B_ptr_add, communication_on_chip,\
                        communication_interval, communication_internal = tlast.buf_process(G_A_in_use,
                                                        G_A0_ready,
                                                        G_A1_ready,
                                                        G_A0_ptr,
                                                        G_A0_ptw,
                                                        G_A1_ptr,
                                                        G_A1_ptw,
                                                        a_loops,
                                                        b_loops,
                                                        G_B0_ready,
                                                        G_B1_ready,
                                                        G_B0_ptr,
                                                        G_B0_ptw,
                                                        G_B1_ptr,
                                                        G_B1_ptw,
                                                        G_B_in_use,
                                                        B_SA_load,
                                                        A_interval_work,
                                                        A_interval_spare,
                                                        G_C0_ready,
                                                        G_C0_ptw,
                                                        G_C0_ptr,
                                                        G_C1_ready,
                                                        G_C1_ptw,
                                                        G_C1_ptr,
                                                        G_Cw_in_use,
                                                        G_Cr_in_use,
                                                        cw_loops,
                                                        cr_loops,
                                                        Array_i,
                                                        Array_j,
                                                        DR_list,
                                                        SA_list,
                                                        m_num,
                                                        n_num,
                                                        k_num,
                                                        m_num_before,
                                                        n_num_before,
                                                        k_num_before,
                                                        n_num_next,
                                                        k_num_next,
                                                        bandwidth,
                                                        compute_pipeline,
                                                        on_chip_alter,
                                                        A_buf_tile_trans,
                                                        B_buf_tile_trans,
                                                        C_buf_tile_trans,
                                                        Y_buf_tile_trans,
                                                        bw_in_use,
                                                        communication_Y_cycle,
                                                        C_load_out,
                                                        G_B_ptr_add,
                                                        communication_on_chip,
                                                        communication_interval,
                                                        communication_internal
                                                        )
                        cycle += 1
                        if tracking == 1:
                            writer.writerow({'cycle':str(cycle),'G_B0_ptw':str(G_B0_ptw),'G_B1_ptw':str(G_B1_ptw),'G_B0_ptr':str(G_B0_ptr),'G_B1_ptr':str(G_B1_ptr),\
                            'G_A0_ptw':str(G_A0_ptw),'G_A1_ptw':str(G_A1_ptw),'G_A0_ptr':str(G_A0_ptr),'G_A1_ptr':str(G_A1_ptr),\
                            'G_C0_ptr':str(G_C0_ptr),'G_C1_ptr':str(G_C1_ptr),'G_C0_ptw':str(G_C0_ptw),'G_C1_ptw':str(G_C1_ptw)})
                else:
                    A_load, B_load, C_load_in, C_load_out = 1, 1, 0, 0
                    A_load0, B_load0, C_load_in0, C_load_out0 = 1, 1, 0, 0
                    bw_in_use = 'B'
                    if G_B_in_use == 0:
                        G_B1_ready = 0
                    else:
                        G_B0_ready = 0
                    while(on_chip_alter == 0):
                        G_A_in_use, G_A0_ready, G_A1_ready, G_A0_ptr, G_A0_ptw, G_A1_ptr, G_A1_ptw, a_loops, b_loops,\
                        G_B0_ready, G_B1_ready, G_B0_ptr, G_B0_ptw, G_B1_ptr, G_B1_ptw, G_B_in_use, B_SA_load,\
                        A_interval_work, A_interval_spare, G_C0_ready, G_C0_ptw, G_C0_ptr, G_C1_ready, G_C1_ptw,\
                        G_C1_ptr, G_Cw_in_use, G_Cr_in_use, cw_loops, cr_loops,\
                        on_chip_alter, bw_in_use, communication_A_cycle, communication_B_cycle, communication_C_cycle,\
                        communication_Y_cycle, B_load, G_B_ptr_add, communication_on_chip,\
                        communication_interval, communication_internal = t.buf_process(G_A_in_use,
                                                        G_A0_ready,
                                                        G_A1_ready,
                                                        G_A0_ptr,
                                                        G_A0_ptw,
                                                        G_A1_ptr,
                                                        G_A1_ptw,
                                                        a_loops,
                                                        b_loops,
                                                        G_B0_ready,
                                                        G_B1_ready,
                                                        G_B0_ptr,
                                                        G_B0_ptw,
                                                        G_B1_ptr,
                                                        G_B1_ptw,
                                                        G_B_in_use,
                                                        B_SA_load,
                                                        A_interval_work,
                                                        A_interval_spare,
                                                        G_C0_ready,
                                                        G_C0_ptw,
                                                        G_C0_ptr,
                                                        G_C1_ready,
                                                        G_C1_ptw,
                                                        G_C1_ptr,
                                                        G_Cw_in_use,
                                                        G_Cr_in_use,
                                                        cw_loops,
                                                        cr_loops,
                                                        Array_i,
                                                        Array_j,
                                                        DR_list,
                                                        SA_list,
                                                        m_num,
                                                        n_num,
                                                        k_num,
                                                        m_num_before,
                                                        n_num_before,
                                                        k_num_before,
                                                        n_num_next,
                                                        k_num_next,
                                                        bandwidth,
                                                        compute_pipeline,
                                                        on_chip_alter,
                                                        A_buf_tile_trans,
                                                        B_buf_tile_trans,
                                                        C_buf_tile_trans,
                                                        Y_buf_tile_trans,
                                                        bw_in_use,
                                                        communication_A_cycle,
                                                        communication_B_cycle,
                                                        communication_C_cycle,
                                                        communication_Y_cycle,
                                                        A_load,
                                                        B_load,
                                                        C_load_in,
                                                        C_load_out,
                                                        G_B_ptr_add,
                                                        communication_on_chip,
                                                        communication_interval,
                                                        communication_internal
                                                        )
                        cycle += 1
                        if tracking == 1:
                            writer.writerow({'cycle':str(cycle),'G_B0_ptw':str(G_B0_ptw),'G_B1_ptw':str(G_B1_ptw),'G_B0_ptr':str(G_B0_ptr),'G_B1_ptr':str(G_B1_ptr),\
                            'G_A0_ptw':str(G_A0_ptw),'G_A1_ptw':str(G_A1_ptw),'G_A0_ptr':str(G_A0_ptr),'G_A1_ptr':str(G_A1_ptr),\
                            'G_C0_ptr':str(G_C0_ptr),'G_C1_ptr':str(G_C1_ptr),'G_C0_ptw':str(G_C0_ptw),'G_C1_ptw':str(G_C1_ptw)})

                communication_off_chip += A_load0 * f.ceil(m_num_next * k_num_next, bandwidth)\
                + B_load0 * f.ceil(k_num_next * n_num_next, bandwidth) + C_load_in0 * partial_sum[m_serial_next][n_serial_next] * f.ceil(m_num_next * n_num_next, bandwidth)\
                + C_load_out0 * f.ceil(m_num_before * n_num_before, bandwidth)
                # 记录片外的部分和状态，“读入”到片上一次后，下次读取处于存在部分和的状态
                if C_load_in == 1:
                    partial_sum[m_serial_next][n_serial_next] = 1
                print("\r", end="")
                print("Simulation progress: {}%: ".format(int(counter/counter_total * 100)), end="")
                sys.stdout.flush()
if m_loop_times == 1 and k_loop_times >= 3 and n_loop_times >= 3:
    for n_serial in range(0,n_loop_times):
        for m_serial in range(0,m_loop_times):
            for k_serial in range(0,k_loop_times):
                counter += 1
                # 计算本次片上的mkn规模大小
                m_num, k_num, n_num = f.mkn_num(m_serial, k_serial, n_serial, m_loop_times, k_loop_times, n_loop_times, M, K, N, m, k, n)
                m_serial_next, k_serial_next, n_serial_next = f.next_num_nmk(m_serial, k_serial, n_serial, m_loop_times, k_loop_times, n_loop_times)
                m_serial_before, k_serial_before, n_serial_before = f.before_num_nmk(m_serial, k_serial, n_serial, m_loop_times, k_loop_times, n_loop_times)
                m_num_next, k_num_next, n_num_next = f.mkn_num(m_serial_next, k_serial_next, n_serial_next, m_loop_times, k_loop_times, n_loop_times, M, K, N, m, k, n)
                m_num_before, k_num_before, n_num_before = f.mkn_num(m_serial_before, k_serial_before, n_serial_before, m_loop_times, k_loop_times, n_loop_times, M, K, N, m, k, n)
                m_coordinate, k_coordinate, n_coordinate = m_serial_next * m, k_serial_next * k, n_serial_next * n

                A_buf_tile_trans = f.ceil(m_num_next * k_num_next, bandwidth)
                B_buf_tile_trans = f.ceil(k_num_next * n_num_next, bandwidth)
                # 判断C是否要从片外读入，第一次是不需要读入的
                if partial_sum[m_serial_next][n_serial_next] == 0:
                    C_buf_tile_trans = 0
                else:
                    C_buf_tile_trans = f.ceil(m_num_next * n_num_next, bandwidth)
                Y_buf_tile_trans = f.ceil(m_num_before * n_num_before, bandwidth)
                on_chip_alter = 0
                # 第一个
                if m_serial==0 and k_serial==k_loop_times-1 and n_serial==0:
                    A_load, B_load, C_load_in, C_load_out = 1, 1, 1, 0
                    A_load0, B_load0, C_load_in0, C_load_out0 = 1, 1, 1, 0
                    bw_in_use = 'B'
                    if G_B_in_use == 0:
                        G_B1_ready = 0
                    else:
                        G_B0_ready = 0
                    while(on_chip_alter == 0):
                        G_A_in_use, G_A0_ready, G_A1_ready, G_A0_ptr, G_A0_ptw, G_A1_ptr, G_A1_ptw, a_loops, b_loops,\
                        G_B0_ready, G_B1_ready, G_B0_ptr, G_B0_ptw, G_B1_ptr, G_B1_ptw, G_B_in_use, B_SA_load,\
                        A_interval_work, A_interval_spare, G_C0_ready, G_C0_ptw, G_C0_ptr, G_C1_ready, G_C1_ptw,\
                        G_C1_ptr, G_Cw_in_use, G_Cr_in_use, cw_loops, cr_loops,\
                        on_chip_alter, bw_in_use, communication_A_cycle, communication_B_cycle, communication_C_cycle,\
                        communication_Y_cycle, B_load, G_B_ptr_add, communication_on_chip,\
                        communication_interval, communication_internal = t.buf_process(G_A_in_use,
                                                        G_A0_ready,
                                                        G_A1_ready,
                                                        G_A0_ptr,
                                                        G_A0_ptw,
                                                        G_A1_ptr,
                                                        G_A1_ptw,
                                                        a_loops,
                                                        b_loops,
                                                        G_B0_ready,
                                                        G_B1_ready,
                                                        G_B0_ptr,
                                                        G_B0_ptw,
                                                        G_B1_ptr,
                                                        G_B1_ptw,
                                                        G_B_in_use,
                                                        B_SA_load,
                                                        A_interval_work,
                                                        A_interval_spare,
                                                        G_C0_ready,
                                                        G_C0_ptw,
                                                        G_C0_ptr,
                                                        G_C1_ready,
                                                        G_C1_ptw,
                                                        G_C1_ptr,
                                                        G_Cw_in_use,
                                                        G_Cr_in_use,
                                                        cw_loops,
                                                        cr_loops,
                                                        Array_i,
                                                        Array_j,
                                                        DR_list,
                                                        SA_list,
                                                        m_num,
                                                        n_num,
                                                        k_num,
                                                        m_num_before,
                                                        n_num_before,
                                                        k_num_before,
                                                        n_num_next,
                                                        k_num_next,
                                                        bandwidth,
                                                        compute_pipeline,
                                                        on_chip_alter,
                                                        A_buf_tile_trans,
                                                        B_buf_tile_trans,
                                                        C_buf_tile_trans,
                                                        Y_buf_tile_trans,
                                                        bw_in_use,
                                                        communication_A_cycle,
                                                        communication_B_cycle,
                                                        communication_C_cycle,
                                                        communication_Y_cycle,
                                                        A_load,
                                                        B_load,
                                                        C_load_in,
                                                        C_load_out,
                                                        G_B_ptr_add,
                                                        communication_on_chip,
                                                        communication_interval,
                                                        communication_internal
                                                        )
                        cycle += 1
                        if tracking == 1:
                            writer.writerow({'cycle':str(cycle),'G_B0_ptw':str(G_B0_ptw),'G_B1_ptw':str(G_B1_ptw),'G_B0_ptr':str(G_B0_ptr),'G_B1_ptr':str(G_B1_ptr),\
                            'G_A0_ptw':str(G_A0_ptw),'G_A1_ptw':str(G_A1_ptw),'G_A0_ptr':str(G_A0_ptr),'G_A1_ptr':str(G_A1_ptr),\
                            'G_C0_ptr':str(G_C0_ptr),'G_C1_ptr':str(G_C1_ptr),'G_C0_ptw':str(G_C0_ptw),'G_C1_ptw':str(G_C1_ptw)})
                # 最后一个
                elif m_serial==m_loop_times-1 and k_serial==k_loop_times-1 and n_serial==n_loop_times-1:
                    A_load, B_load, C_load_in, C_load_out = 0, 0, 0, 1
                    A_load0, B_load0, C_load_in0, C_load_out0 = 0, 0, 0, 1
                    while(on_chip_alter == 0):
                        G_A_in_use, G_A0_ready, G_A1_ready, G_A0_ptr, G_A0_ptw, G_A1_ptr, G_A1_ptw, a_loops, b_loops,\
                        G_B0_ready, G_B1_ready, G_B0_ptr, G_B0_ptw, G_B1_ptr, G_B1_ptw, G_B_in_use, B_SA_load,\
                        A_interval_work, A_interval_spare, G_C0_ready, G_C0_ptw, G_C0_ptr, G_C1_ready, G_C1_ptw,\
                        G_C1_ptr, G_Cw_in_use, G_Cr_in_use, cw_loops, cr_loops,\
                        on_chip_alter, bw_in_use, communication_Y_cycle, G_B_ptr_add, communication_on_chip,\
                        communication_interval, communication_internal = tlast.buf_process(G_A_in_use,
                                                        G_A0_ready,
                                                        G_A1_ready,
                                                        G_A0_ptr,
                                                        G_A0_ptw,
                                                        G_A1_ptr,
                                                        G_A1_ptw,
                                                        a_loops,
                                                        b_loops,
                                                        G_B0_ready,
                                                        G_B1_ready,
                                                        G_B0_ptr,
                                                        G_B0_ptw,
                                                        G_B1_ptr,
                                                        G_B1_ptw,
                                                        G_B_in_use,
                                                        B_SA_load,
                                                        A_interval_work,
                                                        A_interval_spare,
                                                        G_C0_ready,
                                                        G_C0_ptw,
                                                        G_C0_ptr,
                                                        G_C1_ready,
                                                        G_C1_ptw,
                                                        G_C1_ptr,
                                                        G_Cw_in_use,
                                                        G_Cr_in_use,
                                                        cw_loops,
                                                        cr_loops,
                                                        Array_i,
                                                        Array_j,
                                                        DR_list,
                                                        SA_list,
                                                        m_num,
                                                        n_num,
                                                        k_num,
                                                        m_num_before,
                                                        n_num_before,
                                                        k_num_before,
                                                        n_num_next,
                                                        k_num_next,
                                                        bandwidth,
                                                        compute_pipeline,
                                                        on_chip_alter,
                                                        A_buf_tile_trans,
                                                        B_buf_tile_trans,
                                                        C_buf_tile_trans,
                                                        Y_buf_tile_trans,
                                                        bw_in_use,
                                                        communication_Y_cycle,
                                                        C_load_out,
                                                        G_B_ptr_add,
                                                        communication_on_chip,
                                                        communication_interval,
                                                        communication_internal
                                                        )
                        cycle += 1
                        if tracking == 1:
                            writer.writerow({'cycle':str(cycle),'G_B0_ptw':str(G_B0_ptw),'G_B1_ptw':str(G_B1_ptw),'G_B0_ptr':str(G_B0_ptr),'G_B1_ptr':str(G_B1_ptr),\
                            'G_A0_ptw':str(G_A0_ptw),'G_A1_ptw':str(G_A1_ptw),'G_A0_ptr':str(G_A0_ptr),'G_A1_ptr':str(G_A1_ptr),\
                            'G_C0_ptr':str(G_C0_ptr),'G_C1_ptr':str(G_C1_ptr),'G_C0_ptw':str(G_C0_ptw),'G_C1_ptw':str(G_C1_ptw)})
                elif m_serial==0 and k_serial!=k_loop_times-1:
                    A_load, B_load, C_load_in, C_load_out = 1, 1, 0, 0
                    A_load0, B_load0, C_load_in0, C_load_out0 = 1, 1, 0, 0
                    bw_in_use = 'B'
                    if G_B_in_use == 0:
                        G_B1_ready = 0
                    else:
                        G_B0_ready = 0
                    while(on_chip_alter == 0):
                        G_A_in_use, G_A0_ready, G_A1_ready, G_A0_ptr, G_A0_ptw, G_A1_ptr, G_A1_ptw, a_loops, b_loops,\
                        G_B0_ready, G_B1_ready, G_B0_ptr, G_B0_ptw, G_B1_ptr, G_B1_ptw, G_B_in_use, B_SA_load,\
                        A_interval_work, A_interval_spare, G_C0_ready, G_C0_ptw, G_C0_ptr, G_C1_ready, G_C1_ptw,\
                        G_C1_ptr, G_Cw_in_use, G_Cr_in_use, cw_loops, cr_loops,\
                        on_chip_alter, bw_in_use, communication_A_cycle, communication_B_cycle, communication_C_cycle,\
                        communication_Y_cycle, B_load, G_B_ptr_add, communication_on_chip,\
                        communication_interval, communication_internal = t.buf_process(G_A_in_use,
                                                        G_A0_ready,
                                                        G_A1_ready,
                                                        G_A0_ptr,
                                                        G_A0_ptw,
                                                        G_A1_ptr,
                                                        G_A1_ptw,
                                                        a_loops,
                                                        b_loops,
                                                        G_B0_ready,
                                                        G_B1_ready,
                                                        G_B0_ptr,
                                                        G_B0_ptw,
                                                        G_B1_ptr,
                                                        G_B1_ptw,
                                                        G_B_in_use,
                                                        B_SA_load,
                                                        A_interval_work,
                                                        A_interval_spare,
                                                        G_C0_ready,
                                                        G_C0_ptw,
                                                        G_C0_ptr,
                                                        G_C1_ready,
                                                        G_C1_ptw,
                                                        G_C1_ptr,
                                                        G_Cw_in_use,
                                                        G_Cr_in_use,
                                                        cw_loops,
                                                        cr_loops,
                                                        Array_i,
                                                        Array_j,
                                                        DR_list,
                                                        SA_list,
                                                        m_num,
                                                        n_num,
                                                        k_num,
                                                        m_num_before,
                                                        n_num_before,
                                                        k_num_before,
                                                        n_num_next,
                                                        k_num_next,
                                                        bandwidth,
                                                        compute_pipeline,
                                                        on_chip_alter,
                                                        A_buf_tile_trans,
                                                        B_buf_tile_trans,
                                                        C_buf_tile_trans,
                                                        Y_buf_tile_trans,
                                                        bw_in_use,
                                                        communication_A_cycle,
                                                        communication_B_cycle,
                                                        communication_C_cycle,
                                                        communication_Y_cycle,
                                                        A_load,
                                                        B_load,
                                                        C_load_in,
                                                        C_load_out,
                                                        G_B_ptr_add,
                                                        communication_on_chip,
                                                        communication_interval,
                                                        communication_internal
                                                        )
                        cycle += 1
                        if tracking == 1:
                            writer.writerow({'cycle':str(cycle),'G_B0_ptw':str(G_B0_ptw),'G_B1_ptw':str(G_B1_ptw),'G_B0_ptr':str(G_B0_ptr),'G_B1_ptr':str(G_B1_ptr),\
                            'G_A0_ptw':str(G_A0_ptw),'G_A1_ptw':str(G_A1_ptw),'G_A0_ptr':str(G_A0_ptr),'G_A1_ptr':str(G_A1_ptr),\
                            'G_C0_ptr':str(G_C0_ptr),'G_C1_ptr':str(G_C1_ptr),'G_C0_ptw':str(G_C0_ptw),'G_C1_ptw':str(G_C1_ptw)})
                else:
                    A_load, B_load, C_load_in, C_load_out = 1, 1, 1, 1
                    A_load0, B_load0, C_load_in0, C_load_out0 = 1, 1, 1, 1
                    bw_in_use = 'B'
                    if G_B_in_use == 0:
                        G_B1_ready = 0
                    else:
                        G_B0_ready = 0
                    while(on_chip_alter == 0):
                        G_A_in_use, G_A0_ready, G_A1_ready, G_A0_ptr, G_A0_ptw, G_A1_ptr, G_A1_ptw, a_loops, b_loops,\
                        G_B0_ready, G_B1_ready, G_B0_ptr, G_B0_ptw, G_B1_ptr, G_B1_ptw, G_B_in_use, B_SA_load,\
                        A_interval_work, A_interval_spare, G_C0_ready, G_C0_ptw, G_C0_ptr, G_C1_ready, G_C1_ptw,\
                        G_C1_ptr, G_Cw_in_use, G_Cr_in_use, cw_loops, cr_loops,\
                        on_chip_alter, bw_in_use, communication_A_cycle, communication_B_cycle, communication_C_cycle,\
                        communication_Y_cycle, B_load, G_B_ptr_add, communication_on_chip,\
                        communication_interval, communication_internal = t.buf_process(G_A_in_use,
                                                        G_A0_ready,
                                                        G_A1_ready,
                                                        G_A0_ptr,
                                                        G_A0_ptw,
                                                        G_A1_ptr,
                                                        G_A1_ptw,
                                                        a_loops,
                                                        b_loops,
                                                        G_B0_ready,
                                                        G_B1_ready,
                                                        G_B0_ptr,
                                                        G_B0_ptw,
                                                        G_B1_ptr,
                                                        G_B1_ptw,
                                                        G_B_in_use,
                                                        B_SA_load,
                                                        A_interval_work,
                                                        A_interval_spare,
                                                        G_C0_ready,
                                                        G_C0_ptw,
                                                        G_C0_ptr,
                                                        G_C1_ready,
                                                        G_C1_ptw,
                                                        G_C1_ptr,
                                                        G_Cw_in_use,
                                                        G_Cr_in_use,
                                                        cw_loops,
                                                        cr_loops,
                                                        Array_i,
                                                        Array_j,
                                                        DR_list,
                                                        SA_list,
                                                        m_num,
                                                        n_num,
                                                        k_num,
                                                        m_num_before,
                                                        n_num_before,
                                                        k_num_before,
                                                        n_num_next,
                                                        k_num_next,
                                                        bandwidth,
                                                        compute_pipeline,
                                                        on_chip_alter,
                                                        A_buf_tile_trans,
                                                        B_buf_tile_trans,
                                                        C_buf_tile_trans,
                                                        Y_buf_tile_trans,
                                                        bw_in_use,
                                                        communication_A_cycle,
                                                        communication_B_cycle,
                                                        communication_C_cycle,
                                                        communication_Y_cycle,
                                                        A_load,
                                                        B_load,
                                                        C_load_in,
                                                        C_load_out,
                                                        G_B_ptr_add,
                                                        communication_on_chip,
                                                        communication_interval,
                                                        communication_internal
                                                        )
                        cycle += 1
                        if tracking == 1:
                            writer.writerow({'cycle':str(cycle),'G_B0_ptw':str(G_B0_ptw),'G_B1_ptw':str(G_B1_ptw),'G_B0_ptr':str(G_B0_ptr),'G_B1_ptr':str(G_B1_ptr),\
                            'G_A0_ptw':str(G_A0_ptw),'G_A1_ptw':str(G_A1_ptw),'G_A0_ptr':str(G_A0_ptr),'G_A1_ptr':str(G_A1_ptr),\
                            'G_C0_ptr':str(G_C0_ptr),'G_C1_ptr':str(G_C1_ptr),'G_C0_ptw':str(G_C0_ptw),'G_C1_ptw':str(G_C1_ptw)})

                communication_off_chip += A_load0 * f.ceil(m_num_next * k_num_next, bandwidth)\
                + B_load0 * f.ceil(k_num_next * n_num_next, bandwidth) + C_load_in0 * partial_sum[m_serial_next][n_serial_next] * f.ceil(m_num_next * n_num_next, bandwidth)\
                + C_load_out0 * f.ceil(m_num_before * n_num_before, bandwidth)
                # 记录片外的部分和状态，“读入”到片上一次后，下次读取处于存在部分和的状态
                if C_load_in == 1:
                    partial_sum[m_serial_next][n_serial_next] = 1
                print("\r", end="")
                print("Simulation progress: {}%: ".format(int(counter/counter_total * 100)), end="")
                sys.stdout.flush()
if m_loop_times >= 2 and k_loop_times == 1 and n_loop_times == 1:
    for n_serial in range(0,n_loop_times):
        for m_serial in range(0,m_loop_times):
            for k_serial in range(0,k_loop_times):
                counter += 1
                # 计算本次片上的mkn规模大小
                m_num, k_num, n_num = f.mkn_num(m_serial, k_serial, n_serial, m_loop_times, k_loop_times, n_loop_times, M, K, N, m, k, n)
                m_serial_next, k_serial_next, n_serial_next = f.next_num_nmk(m_serial, k_serial, n_serial, m_loop_times, k_loop_times, n_loop_times)
                m_serial_before, k_serial_before, n_serial_before = f.before_num_nmk(m_serial, k_serial, n_serial, m_loop_times, k_loop_times, n_loop_times)
                m_num_next, k_num_next, n_num_next = f.mkn_num(m_serial_next, k_serial_next, n_serial_next, m_loop_times, k_loop_times, n_loop_times, M, K, N, m, k, n)
                m_num_before, k_num_before, n_num_before = f.mkn_num(m_serial_before, k_serial_before, n_serial_before, m_loop_times, k_loop_times, n_loop_times, M, K, N, m, k, n)

                m_coordinate, k_coordinate, n_coordinate = m_serial_next * m, k_serial_next * k, n_serial_next * n

                A_buf_tile_trans = f.ceil(m_num_next * k_num_next, bandwidth)
                B_buf_tile_trans = f.ceil(k_num_next * n_num_next, bandwidth)
                # 判断C是否要从片外读入，第一次是不需要读入的
                if partial_sum[m_serial_next][n_serial_next] == 0:
                    C_buf_tile_trans = 0
                else:
                    C_buf_tile_trans = f.ceil(m_num_next * n_num_next, bandwidth)
                Y_buf_tile_trans = f.ceil(m_num_before * n_num_before, bandwidth)
                on_chip_alter = 0
                # 第一个
                if m_serial==0 and k_serial==0 and n_serial==0:
                    A_load, B_load, C_load_in, C_load_out = 1, 0, 1, 0
                    A_load0, B_load0, C_load_in0, C_load_out0 = 1, 0, 1, 0
                    while(on_chip_alter == 0):
                        G_A_in_use, G_A0_ready, G_A1_ready, G_A0_ptr, G_A0_ptw, G_A1_ptr, G_A1_ptw, a_loops, b_loops,\
                        G_B0_ready, G_B1_ready, G_B0_ptr, G_B0_ptw, G_B1_ptr, G_B1_ptw, G_B_in_use, B_SA_load,\
                        A_interval_work, A_interval_spare, G_C0_ready, G_C0_ptw, G_C0_ptr, G_C1_ready, G_C1_ptw,\
                        G_C1_ptr, G_Cw_in_use, G_Cr_in_use, cw_loops, cr_loops,\
                        on_chip_alter, bw_in_use, communication_A_cycle, communication_B_cycle, communication_C_cycle,\
                        communication_Y_cycle, B_load, G_B_ptr_add, communication_on_chip,\
                        communication_interval, communication_internal = t.buf_process(G_A_in_use,
                                                        G_A0_ready,
                                                        G_A1_ready,
                                                        G_A0_ptr,
                                                        G_A0_ptw,
                                                        G_A1_ptr,
                                                        G_A1_ptw,
                                                        a_loops,
                                                        b_loops,
                                                        G_B0_ready,
                                                        G_B1_ready,
                                                        G_B0_ptr,
                                                        G_B0_ptw,
                                                        G_B1_ptr,
                                                        G_B1_ptw,
                                                        G_B_in_use,
                                                        B_SA_load,
                                                        A_interval_work,
                                                        A_interval_spare,
                                                        G_C0_ready,
                                                        G_C0_ptw,
                                                        G_C0_ptr,
                                                        G_C1_ready,
                                                        G_C1_ptw,
                                                        G_C1_ptr,
                                                        G_Cw_in_use,
                                                        G_Cr_in_use,
                                                        cw_loops,
                                                        cr_loops,
                                                        Array_i,
                                                        Array_j,
                                                        DR_list,
                                                        SA_list,
                                                        m_num,
                                                        n_num,
                                                        k_num,
                                                        m_num_before,
                                                        n_num_before,
                                                        k_num_before,
                                                        n_num_next,
                                                        k_num_next,
                                                        bandwidth,
                                                        compute_pipeline,
                                                        on_chip_alter,
                                                        A_buf_tile_trans,
                                                        B_buf_tile_trans,
                                                        C_buf_tile_trans,
                                                        Y_buf_tile_trans,
                                                        bw_in_use,
                                                        communication_A_cycle,
                                                        communication_B_cycle,
                                                        communication_C_cycle,
                                                        communication_Y_cycle,
                                                        A_load,
                                                        B_load,
                                                        C_load_in,
                                                        C_load_out,
                                                        G_B_ptr_add,
                                                        communication_on_chip,
                                                        communication_interval,
                                                        communication_internal
                                                        )
                        cycle += 1
                        if tracking == 1:
                            writer.writerow({'cycle':str(cycle),'G_B0_ptw':str(G_B0_ptw),'G_B1_ptw':str(G_B1_ptw),'G_B0_ptr':str(G_B0_ptr),'G_B1_ptr':str(G_B1_ptr),\
                            'G_A0_ptw':str(G_A0_ptw),'G_A1_ptw':str(G_A1_ptw),'G_A0_ptr':str(G_A0_ptr),'G_A1_ptr':str(G_A1_ptr),\
                            'G_C0_ptr':str(G_C0_ptr),'G_C1_ptr':str(G_C1_ptr),'G_C0_ptw':str(G_C0_ptw),'G_C1_ptw':str(G_C1_ptw)})
                # 最后一个
                elif m_serial==m_loop_times-1 and k_serial==k_loop_times-1 and n_serial==n_loop_times-1:
                    A_load, B_load, C_load_in, C_load_out = 0, 0, 0, 1
                    A_load0, B_load0, C_load_in0, C_load_out0 = 0, 0, 0, 1
                    while(on_chip_alter == 0):
                        G_A_in_use, G_A0_ready, G_A1_ready, G_A0_ptr, G_A0_ptw, G_A1_ptr, G_A1_ptw, a_loops, b_loops,\
                        G_B0_ready, G_B1_ready, G_B0_ptr, G_B0_ptw, G_B1_ptr, G_B1_ptw, G_B_in_use, B_SA_load,\
                        A_interval_work, A_interval_spare, G_C0_ready, G_C0_ptw, G_C0_ptr, G_C1_ready, G_C1_ptw,\
                        G_C1_ptr, G_Cw_in_use, G_Cr_in_use, cw_loops, cr_loops,\
                        on_chip_alter, bw_in_use, communication_Y_cycle, G_B_ptr_add, communication_on_chip,\
                        communication_interval, communication_internal = tlast.buf_process(G_A_in_use,
                                                        G_A0_ready,
                                                        G_A1_ready,
                                                        G_A0_ptr,
                                                        G_A0_ptw,
                                                        G_A1_ptr,
                                                        G_A1_ptw,
                                                        a_loops,
                                                        b_loops,
                                                        G_B0_ready,
                                                        G_B1_ready,
                                                        G_B0_ptr,
                                                        G_B0_ptw,
                                                        G_B1_ptr,
                                                        G_B1_ptw,
                                                        G_B_in_use,
                                                        B_SA_load,
                                                        A_interval_work,
                                                        A_interval_spare,
                                                        G_C0_ready,
                                                        G_C0_ptw,
                                                        G_C0_ptr,
                                                        G_C1_ready,
                                                        G_C1_ptw,
                                                        G_C1_ptr,
                                                        G_Cw_in_use,
                                                        G_Cr_in_use,
                                                        cw_loops,
                                                        cr_loops,
                                                        Array_i,
                                                        Array_j,
                                                        DR_list,
                                                        SA_list,
                                                        m_num,
                                                        n_num,
                                                        k_num,
                                                        m_num_before,
                                                        n_num_before,
                                                        k_num_before,
                                                        n_num_next,
                                                        k_num_next,
                                                        bandwidth,
                                                        compute_pipeline,
                                                        on_chip_alter,
                                                        A_buf_tile_trans,
                                                        B_buf_tile_trans,
                                                        C_buf_tile_trans,
                                                        Y_buf_tile_trans,
                                                        bw_in_use,
                                                        communication_Y_cycle,
                                                        C_load_out,
                                                        G_B_ptr_add,
                                                        communication_on_chip,
                                                        communication_interval,
                                                        communication_internal
                                                        )
                        cycle += 1
                        if tracking == 1:
                            writer.writerow({'cycle':str(cycle),'G_B0_ptw':str(G_B0_ptw),'G_B1_ptw':str(G_B1_ptw),'G_B0_ptr':str(G_B0_ptr),'G_B1_ptr':str(G_B1_ptr),\
                            'G_A0_ptw':str(G_A0_ptw),'G_A1_ptw':str(G_A1_ptw),'G_A0_ptr':str(G_A0_ptr),'G_A1_ptr':str(G_A1_ptr),\
                            'G_C0_ptr':str(G_C0_ptr),'G_C1_ptr':str(G_C1_ptr),'G_C0_ptw':str(G_C0_ptw),'G_C1_ptw':str(G_C1_ptw)})
                else:
                    A_load, B_load, C_load_in, C_load_out = 1, 0, 1, 1
                    A_load0, B_load0, C_load_in0, C_load_out0 = 1, 0, 1, 1
                    bw_in_use = 'B'
                    if G_B_in_use == 0:
                        G_B1_ready = 0
                    else:
                        G_B0_ready = 0
                    while(on_chip_alter == 0):
                        G_A_in_use, G_A0_ready, G_A1_ready, G_A0_ptr, G_A0_ptw, G_A1_ptr, G_A1_ptw, a_loops, b_loops,\
                        G_B0_ready, G_B1_ready, G_B0_ptr, G_B0_ptw, G_B1_ptr, G_B1_ptw, G_B_in_use, B_SA_load,\
                        A_interval_work, A_interval_spare, G_C0_ready, G_C0_ptw, G_C0_ptr, G_C1_ready, G_C1_ptw,\
                        G_C1_ptr, G_Cw_in_use, G_Cr_in_use, cw_loops, cr_loops,\
                        on_chip_alter, bw_in_use, communication_A_cycle, communication_B_cycle, communication_C_cycle,\
                        communication_Y_cycle, B_load, G_B_ptr_add, communication_on_chip,\
                        communication_interval, communication_internal = t.buf_process(G_A_in_use,
                                                        G_A0_ready,
                                                        G_A1_ready,
                                                        G_A0_ptr,
                                                        G_A0_ptw,
                                                        G_A1_ptr,
                                                        G_A1_ptw,
                                                        a_loops,
                                                        b_loops,
                                                        G_B0_ready,
                                                        G_B1_ready,
                                                        G_B0_ptr,
                                                        G_B0_ptw,
                                                        G_B1_ptr,
                                                        G_B1_ptw,
                                                        G_B_in_use,
                                                        B_SA_load,
                                                        A_interval_work,
                                                        A_interval_spare,
                                                        G_C0_ready,
                                                        G_C0_ptw,
                                                        G_C0_ptr,
                                                        G_C1_ready,
                                                        G_C1_ptw,
                                                        G_C1_ptr,
                                                        G_Cw_in_use,
                                                        G_Cr_in_use,
                                                        cw_loops,
                                                        cr_loops,
                                                        Array_i,
                                                        Array_j,
                                                        DR_list,
                                                        SA_list,
                                                        m_num,
                                                        n_num,
                                                        k_num,
                                                        m_num_before,
                                                        n_num_before,
                                                        k_num_before,
                                                        n_num_next,
                                                        k_num_next,
                                                        bandwidth,
                                                        compute_pipeline,
                                                        on_chip_alter,
                                                        A_buf_tile_trans,
                                                        B_buf_tile_trans,
                                                        C_buf_tile_trans,
                                                        Y_buf_tile_trans,
                                                        bw_in_use,
                                                        communication_A_cycle,
                                                        communication_B_cycle,
                                                        communication_C_cycle,
                                                        communication_Y_cycle,
                                                        A_load,
                                                        B_load,
                                                        C_load_in,
                                                        C_load_out,
                                                        G_B_ptr_add,
                                                        communication_on_chip,
                                                        communication_interval,
                                                        communication_internal
                                                        )
                        cycle += 1
                        if tracking == 1:
                            writer.writerow({'cycle':str(cycle),'G_B0_ptw':str(G_B0_ptw),'G_B1_ptw':str(G_B1_ptw),'G_B0_ptr':str(G_B0_ptr),'G_B1_ptr':str(G_B1_ptr),\
                            'G_A0_ptw':str(G_A0_ptw),'G_A1_ptw':str(G_A1_ptw),'G_A0_ptr':str(G_A0_ptr),'G_A1_ptr':str(G_A1_ptr),\
                            'G_C0_ptr':str(G_C0_ptr),'G_C1_ptr':str(G_C1_ptr),'G_C0_ptw':str(G_C0_ptw),'G_C1_ptw':str(G_C1_ptw)})

                communication_off_chip += A_load0 * f.ceil(m_num_next * k_num_next, bandwidth)\
                + B_load0 * f.ceil(k_num_next * n_num_next, bandwidth) + C_load_in0 * partial_sum[m_serial_next][n_serial_next] * f.ceil(m_num_next * n_num_next, bandwidth)\
                + C_load_out0 * f.ceil(m_num_before * n_num_before, bandwidth)
                # 记录片外的部分和状态，“读入”到片上一次后，下次读取处于存在部分和的状态
                if C_load_in == 1:
                    partial_sum[m_serial_next][n_serial_next] = 1
                print("\r", end="")
                print("Simulation progress: {}%: ".format(int(counter/counter_total * 100)), end="")
                sys.stdout.flush()
if m_loop_times == 2 and k_loop_times == 1 and n_loop_times >= 2:
    for n_serial in range(0,n_loop_times):
        for m_serial in range(0,m_loop_times):
            for k_serial in range(0,k_loop_times):
                counter += 1
                # 计算本次片上的mkn规模大小
                m_num, k_num, n_num = f.mkn_num(m_serial, k_serial, n_serial, m_loop_times, k_loop_times, n_loop_times, M, K, N, m, k, n)
                m_serial_next, k_serial_next, n_serial_next = f.next_num_nmk(m_serial, k_serial, n_serial, m_loop_times, k_loop_times, n_loop_times)
                m_serial_before, k_serial_before, n_serial_before = f.before_num_nmk(m_serial, k_serial, n_serial, m_loop_times, k_loop_times, n_loop_times)
                m_num_next, k_num_next, n_num_next = f.mkn_num(m_serial_next, k_serial_next, n_serial_next, m_loop_times, k_loop_times, n_loop_times, M, K, N, m, k, n)
                m_num_before, k_num_before, n_num_before = f.mkn_num(m_serial_before, k_serial_before, n_serial_before, m_loop_times, k_loop_times, n_loop_times, M, K, N, m, k, n)

                m_coordinate, k_coordinate, n_coordinate = m_serial_next * m, k_serial_next * k, n_serial_next * n

                A_buf_tile_trans = f.ceil(m_num_next * k_num_next, bandwidth)
                B_buf_tile_trans = f.ceil(k_num_next * n_num_next, bandwidth)
                # 判断C是否要从片外读入，第一次是不需要读入的
                if partial_sum[m_serial_next][n_serial_next] == 0:
                    C_buf_tile_trans = 0
                else:
                    C_buf_tile_trans = f.ceil(m_num_next * n_num_next, bandwidth)
                Y_buf_tile_trans = f.ceil(m_num_before * n_num_before, bandwidth)
                on_chip_alter = 0
                # 第一个
                if m_serial==0 and k_serial==0 and n_serial==0:
                    A_load, B_load, C_load_in, C_load_out = 1, 0, 1, 0
                    A_load0, B_load0, C_load_in0, C_load_out0 = 1, 0, 1, 0
                    while(on_chip_alter == 0):
                        G_A_in_use, G_A0_ready, G_A1_ready, G_A0_ptr, G_A0_ptw, G_A1_ptr, G_A1_ptw, a_loops, b_loops,\
                        G_B0_ready, G_B1_ready, G_B0_ptr, G_B0_ptw, G_B1_ptr, G_B1_ptw, G_B_in_use, B_SA_load,\
                        A_interval_work, A_interval_spare, G_C0_ready, G_C0_ptw, G_C0_ptr, G_C1_ready, G_C1_ptw,\
                        G_C1_ptr, G_Cw_in_use, G_Cr_in_use, cw_loops, cr_loops,\
                        on_chip_alter, bw_in_use, communication_A_cycle, communication_B_cycle, communication_C_cycle,\
                        communication_Y_cycle, B_load, G_B_ptr_add, communication_on_chip,\
                        communication_interval, communication_internal = t.buf_process(G_A_in_use,
                                                        G_A0_ready,
                                                        G_A1_ready,
                                                        G_A0_ptr,
                                                        G_A0_ptw,
                                                        G_A1_ptr,
                                                        G_A1_ptw,
                                                        a_loops,
                                                        b_loops,
                                                        G_B0_ready,
                                                        G_B1_ready,
                                                        G_B0_ptr,
                                                        G_B0_ptw,
                                                        G_B1_ptr,
                                                        G_B1_ptw,
                                                        G_B_in_use,
                                                        B_SA_load,
                                                        A_interval_work,
                                                        A_interval_spare,
                                                        G_C0_ready,
                                                        G_C0_ptw,
                                                        G_C0_ptr,
                                                        G_C1_ready,
                                                        G_C1_ptw,
                                                        G_C1_ptr,
                                                        G_Cw_in_use,
                                                        G_Cr_in_use,
                                                        cw_loops,
                                                        cr_loops,
                                                        Array_i,
                                                        Array_j,
                                                        DR_list,
                                                        SA_list,
                                                        m_num,
                                                        n_num,
                                                        k_num,
                                                        m_num_before,
                                                        n_num_before,
                                                        k_num_before,
                                                        n_num_next,
                                                        k_num_next,
                                                        bandwidth,
                                                        compute_pipeline,
                                                        on_chip_alter,
                                                        A_buf_tile_trans,
                                                        B_buf_tile_trans,
                                                        C_buf_tile_trans,
                                                        Y_buf_tile_trans,
                                                        bw_in_use,
                                                        communication_A_cycle,
                                                        communication_B_cycle,
                                                        communication_C_cycle,
                                                        communication_Y_cycle,
                                                        A_load,
                                                        B_load,
                                                        C_load_in,
                                                        C_load_out,
                                                        G_B_ptr_add,
                                                        communication_on_chip,
                                                        communication_interval,
                                                        communication_internal
                                                        )
                        cycle += 1
                        if tracking == 1:
                            writer.writerow({'cycle':str(cycle),'G_B0_ptw':str(G_B0_ptw),'G_B1_ptw':str(G_B1_ptw),'G_B0_ptr':str(G_B0_ptr),'G_B1_ptr':str(G_B1_ptr),\
                            'G_A0_ptw':str(G_A0_ptw),'G_A1_ptw':str(G_A1_ptw),'G_A0_ptr':str(G_A0_ptr),'G_A1_ptr':str(G_A1_ptr),\
                            'G_C0_ptr':str(G_C0_ptr),'G_C1_ptr':str(G_C1_ptr),'G_C0_ptw':str(G_C0_ptw),'G_C1_ptw':str(G_C1_ptw)})
                # 最后一个
                elif m_serial==m_loop_times-1 and k_serial==k_loop_times-1 and n_serial==n_loop_times-1:
                    A_load, B_load, C_load_in, C_load_out = 0, 0, 0, 1
                    A_load0, B_load0, C_load_in0, C_load_out0 = 0, 0, 0, 1
                    while(on_chip_alter == 0):
                        G_A_in_use, G_A0_ready, G_A1_ready, G_A0_ptr, G_A0_ptw, G_A1_ptr, G_A1_ptw, a_loops, b_loops,\
                        G_B0_ready, G_B1_ready, G_B0_ptr, G_B0_ptw, G_B1_ptr, G_B1_ptw, G_B_in_use, B_SA_load,\
                        A_interval_work, A_interval_spare, G_C0_ready, G_C0_ptw, G_C0_ptr, G_C1_ready, G_C1_ptw,\
                        G_C1_ptr, G_Cw_in_use, G_Cr_in_use, cw_loops, cr_loops,\
                        on_chip_alter, bw_in_use, communication_Y_cycle, G_B_ptr_add, communication_on_chip,\
                        communication_interval, communication_internal = tlast.buf_process(G_A_in_use,
                                                        G_A0_ready,
                                                        G_A1_ready,
                                                        G_A0_ptr,
                                                        G_A0_ptw,
                                                        G_A1_ptr,
                                                        G_A1_ptw,
                                                        a_loops,
                                                        b_loops,
                                                        G_B0_ready,
                                                        G_B1_ready,
                                                        G_B0_ptr,
                                                        G_B0_ptw,
                                                        G_B1_ptr,
                                                        G_B1_ptw,
                                                        G_B_in_use,
                                                        B_SA_load,
                                                        A_interval_work,
                                                        A_interval_spare,
                                                        G_C0_ready,
                                                        G_C0_ptw,
                                                        G_C0_ptr,
                                                        G_C1_ready,
                                                        G_C1_ptw,
                                                        G_C1_ptr,
                                                        G_Cw_in_use,
                                                        G_Cr_in_use,
                                                        cw_loops,
                                                        cr_loops,
                                                        Array_i,
                                                        Array_j,
                                                        DR_list,
                                                        SA_list,
                                                        m_num,
                                                        n_num,
                                                        k_num,
                                                        m_num_before,
                                                        n_num_before,
                                                        k_num_before,
                                                        n_num_next,
                                                        k_num_next,
                                                        bandwidth,
                                                        compute_pipeline,
                                                        on_chip_alter,
                                                        A_buf_tile_trans,
                                                        B_buf_tile_trans,
                                                        C_buf_tile_trans,
                                                        Y_buf_tile_trans,
                                                        bw_in_use,
                                                        communication_Y_cycle,
                                                        C_load_out,
                                                        G_B_ptr_add,
                                                        communication_on_chip,
                                                        communication_interval,
                                                        communication_internal
                                                        )
                        cycle += 1
                        if tracking == 1:
                            writer.writerow({'cycle':str(cycle),'G_B0_ptw':str(G_B0_ptw),'G_B1_ptw':str(G_B1_ptw),'G_B0_ptr':str(G_B0_ptr),'G_B1_ptr':str(G_B1_ptr),\
                            'G_A0_ptw':str(G_A0_ptw),'G_A1_ptw':str(G_A1_ptw),'G_A0_ptr':str(G_A0_ptr),'G_A1_ptr':str(G_A1_ptr),\
                            'G_C0_ptr':str(G_C0_ptr),'G_C1_ptr':str(G_C1_ptr),'G_C0_ptw':str(G_C0_ptw),'G_C1_ptw':str(G_C1_ptw)})
                elif m_serial==0 and k_serial==0 and n_serial!=0:
                    A_load, B_load, C_load_in, C_load_out = 0, 0, 1, 1
                    A_load0, B_load0, C_load_in0, C_load_out0 = 0, 0, 1, 1
                    while(on_chip_alter == 0):
                        G_A_in_use, G_A0_ready, G_A1_ready, G_A0_ptr, G_A0_ptw, G_A1_ptr, G_A1_ptw, a_loops, b_loops,\
                        G_B0_ready, G_B1_ready, G_B0_ptr, G_B0_ptw, G_B1_ptr, G_B1_ptw, G_B_in_use, B_SA_load,\
                        A_interval_work, A_interval_spare, G_C0_ready, G_C0_ptw, G_C0_ptr, G_C1_ready, G_C1_ptw,\
                        G_C1_ptr, G_Cw_in_use, G_Cr_in_use, cw_loops, cr_loops,\
                        on_chip_alter, bw_in_use, communication_A_cycle, communication_B_cycle, communication_C_cycle,\
                        communication_Y_cycle, B_load, G_B_ptr_add, communication_on_chip,\
                        communication_interval, communication_internal = t.buf_process(G_A_in_use,
                                                        G_A0_ready,
                                                        G_A1_ready,
                                                        G_A0_ptr,
                                                        G_A0_ptw,
                                                        G_A1_ptr,
                                                        G_A1_ptw,
                                                        a_loops,
                                                        b_loops,
                                                        G_B0_ready,
                                                        G_B1_ready,
                                                        G_B0_ptr,
                                                        G_B0_ptw,
                                                        G_B1_ptr,
                                                        G_B1_ptw,
                                                        G_B_in_use,
                                                        B_SA_load,
                                                        A_interval_work,
                                                        A_interval_spare,
                                                        G_C0_ready,
                                                        G_C0_ptw,
                                                        G_C0_ptr,
                                                        G_C1_ready,
                                                        G_C1_ptw,
                                                        G_C1_ptr,
                                                        G_Cw_in_use,
                                                        G_Cr_in_use,
                                                        cw_loops,
                                                        cr_loops,
                                                        Array_i,
                                                        Array_j,
                                                        DR_list,
                                                        SA_list,
                                                        m_num,
                                                        n_num,
                                                        k_num,
                                                        m_num_before,
                                                        n_num_before,
                                                        k_num_before,
                                                        n_num_next,
                                                        k_num_next,
                                                        bandwidth,
                                                        compute_pipeline,
                                                        on_chip_alter,
                                                        A_buf_tile_trans,
                                                        B_buf_tile_trans,
                                                        C_buf_tile_trans,
                                                        Y_buf_tile_trans,
                                                        bw_in_use,
                                                        communication_A_cycle,
                                                        communication_B_cycle,
                                                        communication_C_cycle,
                                                        communication_Y_cycle,
                                                        A_load,
                                                        B_load,
                                                        C_load_in,
                                                        C_load_out,
                                                        G_B_ptr_add,
                                                        communication_on_chip,
                                                        communication_interval,
                                                        communication_internal
                                                        )
                        cycle += 1
                        if tracking == 1:
                            writer.writerow({'cycle':str(cycle),'G_B0_ptw':str(G_B0_ptw),'G_B1_ptw':str(G_B1_ptw),'G_B0_ptr':str(G_B0_ptr),'G_B1_ptr':str(G_B1_ptr),\
                            'G_A0_ptw':str(G_A0_ptw),'G_A1_ptw':str(G_A1_ptw),'G_A0_ptr':str(G_A0_ptr),'G_A1_ptr':str(G_A1_ptr),\
                            'G_C0_ptr':str(G_C0_ptr),'G_C1_ptr':str(G_C1_ptr),'G_C0_ptw':str(G_C0_ptw),'G_C1_ptw':str(G_C1_ptw)})
                else:
                    A_load, B_load, C_load_in, C_load_out = 0, 1, 1, 1
                    A_load0, B_load0, C_load_in0, C_load_out0 = 0, 1, 1, 1
                    bw_in_use = 'B'
                    if G_B_in_use == 0:
                        G_B1_ready = 0
                    else:
                        G_B0_ready = 0
                    while(on_chip_alter == 0):
                        G_A_in_use, G_A0_ready, G_A1_ready, G_A0_ptr, G_A0_ptw, G_A1_ptr, G_A1_ptw, a_loops, b_loops,\
                        G_B0_ready, G_B1_ready, G_B0_ptr, G_B0_ptw, G_B1_ptr, G_B1_ptw, G_B_in_use, B_SA_load,\
                        A_interval_work, A_interval_spare, G_C0_ready, G_C0_ptw, G_C0_ptr, G_C1_ready, G_C1_ptw,\
                        G_C1_ptr, G_Cw_in_use, G_Cr_in_use, cw_loops, cr_loops,\
                        on_chip_alter, bw_in_use, communication_A_cycle, communication_B_cycle, communication_C_cycle,\
                        communication_Y_cycle, B_load, G_B_ptr_add, communication_on_chip,\
                        communication_interval, communication_internal = t.buf_process(G_A_in_use,
                                                        G_A0_ready,
                                                        G_A1_ready,
                                                        G_A0_ptr,
                                                        G_A0_ptw,
                                                        G_A1_ptr,
                                                        G_A1_ptw,
                                                        a_loops,
                                                        b_loops,
                                                        G_B0_ready,
                                                        G_B1_ready,
                                                        G_B0_ptr,
                                                        G_B0_ptw,
                                                        G_B1_ptr,
                                                        G_B1_ptw,
                                                        G_B_in_use,
                                                        B_SA_load,
                                                        A_interval_work,
                                                        A_interval_spare,
                                                        G_C0_ready,
                                                        G_C0_ptw,
                                                        G_C0_ptr,
                                                        G_C1_ready,
                                                        G_C1_ptw,
                                                        G_C1_ptr,
                                                        G_Cw_in_use,
                                                        G_Cr_in_use,
                                                        cw_loops,
                                                        cr_loops,
                                                        Array_i,
                                                        Array_j,
                                                        DR_list,
                                                        SA_list,
                                                        m_num,
                                                        n_num,
                                                        k_num,
                                                        m_num_before,
                                                        n_num_before,
                                                        k_num_before,
                                                        n_num_next,
                                                        k_num_next,
                                                        bandwidth,
                                                        compute_pipeline,
                                                        on_chip_alter,
                                                        A_buf_tile_trans,
                                                        B_buf_tile_trans,
                                                        C_buf_tile_trans,
                                                        Y_buf_tile_trans,
                                                        bw_in_use,
                                                        communication_A_cycle,
                                                        communication_B_cycle,
                                                        communication_C_cycle,
                                                        communication_Y_cycle,
                                                        A_load,
                                                        B_load,
                                                        C_load_in,
                                                        C_load_out,
                                                        G_B_ptr_add,
                                                        communication_on_chip,
                                                        communication_interval,
                                                        communication_internal
                                                        )
                        cycle += 1
                        if tracking == 1:
                            writer.writerow({'cycle':str(cycle),'G_B0_ptw':str(G_B0_ptw),'G_B1_ptw':str(G_B1_ptw),'G_B0_ptr':str(G_B0_ptr),'G_B1_ptr':str(G_B1_ptr),\
                            'G_A0_ptw':str(G_A0_ptw),'G_A1_ptw':str(G_A1_ptw),'G_A0_ptr':str(G_A0_ptr),'G_A1_ptr':str(G_A1_ptr),\
                            'G_C0_ptr':str(G_C0_ptr),'G_C1_ptr':str(G_C1_ptr),'G_C0_ptw':str(G_C0_ptw),'G_C1_ptw':str(G_C1_ptw)})

                communication_off_chip += A_load0 * f.ceil(m_num_next * k_num_next, bandwidth)\
                + B_load0 * f.ceil(k_num_next * n_num_next, bandwidth) + C_load_in0 * partial_sum[m_serial_next][n_serial_next] * f.ceil(m_num_next * n_num_next, bandwidth)\
                + C_load_out0 * f.ceil(m_num_before * n_num_before, bandwidth)
                # 记录片外的部分和状态，“读入”到片上一次后，下次读取处于存在部分和的状态
                if C_load_in == 1:
                    partial_sum[m_serial_next][n_serial_next] = 1
                print("\r", end="")
                print("Simulation progress: {}%: ".format(int(counter/counter_total * 100)), end="")
                sys.stdout.flush()
if m_loop_times == 2 and k_loop_times == 2 and n_loop_times >= 1:
    for n_serial in range(0,n_loop_times):
        for m_serial in range(0,m_loop_times):
            for k_serial in range(0,k_loop_times):
                counter += 1
                # 计算本次片上的mkn规模大小
                m_num, k_num, n_num = f.mkn_num(m_serial, k_serial, n_serial, m_loop_times, k_loop_times, n_loop_times, M, K, N, m, k, n)
                m_serial_next, k_serial_next, n_serial_next = f.next_num_nmk(m_serial, k_serial, n_serial, m_loop_times, k_loop_times, n_loop_times)
                m_serial_before, k_serial_before, n_serial_before = f.before_num_nmk(m_serial, k_serial, n_serial, m_loop_times, k_loop_times, n_loop_times)
                m_num_next, k_num_next, n_num_next = f.mkn_num(m_serial_next, k_serial_next, n_serial_next, m_loop_times, k_loop_times, n_loop_times, M, K, N, m, k, n)
                m_num_before, k_num_before, n_num_before = f.mkn_num(m_serial_before, k_serial_before, n_serial_before, m_loop_times, k_loop_times, n_loop_times, M, K, N, m, k, n)

                m_coordinate, k_coordinate, n_coordinate = m_serial_next * m, k_serial_next * k, n_serial_next * n

                A_buf_tile_trans = f.ceil(m_num_next * k_num_next, bandwidth)
                B_buf_tile_trans = f.ceil(k_num_next * n_num_next, bandwidth)
                # 判断C是否要从片外读入，第一次是不需要读入的
                if partial_sum[m_serial_next][n_serial_next] == 0:
                    C_buf_tile_trans = 0
                else:
                    C_buf_tile_trans = f.ceil(m_num_next * n_num_next, bandwidth)
                Y_buf_tile_trans = f.ceil(m_num_before * n_num_before, bandwidth)
                on_chip_alter = 0
                # 第一个
                if m_serial==0 and k_serial==0:
                    A_load, B_load, C_load_in, C_load_out = 1, 1, 0, 0
                    A_load0, B_load0, C_load_in0, C_load_out0 = 1, 1, 0, 0
                    bw_in_use = 'B'
                    if G_B_in_use == 0:
                        G_B1_ready = 0
                    else:
                        G_B0_ready = 0
                    while(on_chip_alter == 0):
                        G_A_in_use, G_A0_ready, G_A1_ready, G_A0_ptr, G_A0_ptw, G_A1_ptr, G_A1_ptw, a_loops, b_loops,\
                        G_B0_ready, G_B1_ready, G_B0_ptr, G_B0_ptw, G_B1_ptr, G_B1_ptw, G_B_in_use, B_SA_load,\
                        A_interval_work, A_interval_spare, G_C0_ready, G_C0_ptw, G_C0_ptr, G_C1_ready, G_C1_ptw,\
                        G_C1_ptr, G_Cw_in_use, G_Cr_in_use, cw_loops, cr_loops,\
                        on_chip_alter, bw_in_use, communication_A_cycle, communication_B_cycle, communication_C_cycle,\
                        communication_Y_cycle, B_load, G_B_ptr_add, communication_on_chip,\
                        communication_interval, communication_internal = t.buf_process(G_A_in_use,
                                                        G_A0_ready,
                                                        G_A1_ready,
                                                        G_A0_ptr,
                                                        G_A0_ptw,
                                                        G_A1_ptr,
                                                        G_A1_ptw,
                                                        a_loops,
                                                        b_loops,
                                                        G_B0_ready,
                                                        G_B1_ready,
                                                        G_B0_ptr,
                                                        G_B0_ptw,
                                                        G_B1_ptr,
                                                        G_B1_ptw,
                                                        G_B_in_use,
                                                        B_SA_load,
                                                        A_interval_work,
                                                        A_interval_spare,
                                                        G_C0_ready,
                                                        G_C0_ptw,
                                                        G_C0_ptr,
                                                        G_C1_ready,
                                                        G_C1_ptw,
                                                        G_C1_ptr,
                                                        G_Cw_in_use,
                                                        G_Cr_in_use,
                                                        cw_loops,
                                                        cr_loops,
                                                        Array_i,
                                                        Array_j,
                                                        DR_list,
                                                        SA_list,
                                                        m_num,
                                                        n_num,
                                                        k_num,
                                                        m_num_before,
                                                        n_num_before,
                                                        k_num_before,
                                                        n_num_next,
                                                        k_num_next,
                                                        bandwidth,
                                                        compute_pipeline,
                                                        on_chip_alter,
                                                        A_buf_tile_trans,
                                                        B_buf_tile_trans,
                                                        C_buf_tile_trans,
                                                        Y_buf_tile_trans,
                                                        bw_in_use,
                                                        communication_A_cycle,
                                                        communication_B_cycle,
                                                        communication_C_cycle,
                                                        communication_Y_cycle,
                                                        A_load,
                                                        B_load,
                                                        C_load_in,
                                                        C_load_out,
                                                        G_B_ptr_add,
                                                        communication_on_chip,
                                                        communication_interval,
                                                        communication_internal
                                                        )
                        cycle += 1
                        if tracking == 1:
                            writer.writerow({'cycle':str(cycle),'G_B0_ptw':str(G_B0_ptw),'G_B1_ptw':str(G_B1_ptw),'G_B0_ptr':str(G_B0_ptr),'G_B1_ptr':str(G_B1_ptr),\
                            'G_A0_ptw':str(G_A0_ptw),'G_A1_ptw':str(G_A1_ptw),'G_A0_ptr':str(G_A0_ptr),'G_A1_ptr':str(G_A1_ptr),\
                            'G_C0_ptr':str(G_C0_ptr),'G_C1_ptr':str(G_C1_ptr),'G_C0_ptw':str(G_C0_ptw),'G_C1_ptw':str(G_C1_ptw)})
                # 最后一个
                elif m_serial==m_loop_times-1 and k_serial==k_loop_times-1 and n_serial==n_loop_times-1:
                    A_load, B_load, C_load_in, C_load_out = 0, 0, 0, 1
                    A_load0, B_load0, C_load_in0, C_load_out0 = 0, 0, 0, 1
                    bw_in_use = 'B'
                    if G_B_in_use == 0:
                        G_B1_ready = 0
                    else:
                        G_B0_ready = 0
                    while(on_chip_alter == 0):
                        G_A_in_use, G_A0_ready, G_A1_ready, G_A0_ptr, G_A0_ptw, G_A1_ptr, G_A1_ptw, a_loops, b_loops,\
                        G_B0_ready, G_B1_ready, G_B0_ptr, G_B0_ptw, G_B1_ptr, G_B1_ptw, G_B_in_use, B_SA_load,\
                        A_interval_work, A_interval_spare, G_C0_ready, G_C0_ptw, G_C0_ptr, G_C1_ready, G_C1_ptw,\
                        G_C1_ptr, G_Cw_in_use, G_Cr_in_use, cw_loops, cr_loops,\
                        on_chip_alter, bw_in_use, communication_Y_cycle, G_B_ptr_add, communication_on_chip,\
                        communication_interval, communication_internal = tlast.buf_process(G_A_in_use,
                                                        G_A0_ready,
                                                        G_A1_ready,
                                                        G_A0_ptr,
                                                        G_A0_ptw,
                                                        G_A1_ptr,
                                                        G_A1_ptw,
                                                        a_loops,
                                                        b_loops,
                                                        G_B0_ready,
                                                        G_B1_ready,
                                                        G_B0_ptr,
                                                        G_B0_ptw,
                                                        G_B1_ptr,
                                                        G_B1_ptw,
                                                        G_B_in_use,
                                                        B_SA_load,
                                                        A_interval_work,
                                                        A_interval_spare,
                                                        G_C0_ready,
                                                        G_C0_ptw,
                                                        G_C0_ptr,
                                                        G_C1_ready,
                                                        G_C1_ptw,
                                                        G_C1_ptr,
                                                        G_Cw_in_use,
                                                        G_Cr_in_use,
                                                        cw_loops,
                                                        cr_loops,
                                                        Array_i,
                                                        Array_j,
                                                        DR_list,
                                                        SA_list,
                                                        m_num,
                                                        n_num,
                                                        k_num,
                                                        m_num_before,
                                                        n_num_before,
                                                        k_num_before,
                                                        n_num_next,
                                                        k_num_next,
                                                        bandwidth,
                                                        compute_pipeline,
                                                        on_chip_alter,
                                                        A_buf_tile_trans,
                                                        B_buf_tile_trans,
                                                        C_buf_tile_trans,
                                                        Y_buf_tile_trans,
                                                        bw_in_use,
                                                        communication_Y_cycle,
                                                        C_load_out,
                                                        G_B_ptr_add,
                                                        communication_on_chip,
                                                        communication_interval,
                                                        communication_internal
                                                        )
                        cycle += 1
                        if tracking == 1:
                            writer.writerow({'cycle':str(cycle),'G_B0_ptw':str(G_B0_ptw),'G_B1_ptw':str(G_B1_ptw),'G_B0_ptr':str(G_B0_ptr),'G_B1_ptr':str(G_B1_ptr),\
                            'G_A0_ptw':str(G_A0_ptw),'G_A1_ptw':str(G_A1_ptw),'G_A0_ptr':str(G_A0_ptr),'G_A1_ptr':str(G_A1_ptr),\
                            'G_C0_ptr':str(G_C0_ptr),'G_C1_ptr':str(G_C1_ptr),'G_C0_ptw':str(G_C0_ptw),'G_C1_ptw':str(G_C1_ptw)})
                elif m_serial==0 and k_serial==k_loop_times-1 and n_serial==0:
                    A_load, B_load, C_load_in, C_load_out = 1, 0, 1, 0
                    A_load0, B_load0, C_load_in0, C_load_out0 = 1, 0, 1, 0
                    bw_in_use = 'B'
                    if G_B_in_use == 0:
                        G_B1_ready = 0
                    else:
                        G_B0_ready = 0
                    while(on_chip_alter == 0):
                        G_A_in_use, G_A0_ready, G_A1_ready, G_A0_ptr, G_A0_ptw, G_A1_ptr, G_A1_ptw, a_loops, b_loops,\
                        G_B0_ready, G_B1_ready, G_B0_ptr, G_B0_ptw, G_B1_ptr, G_B1_ptw, G_B_in_use, B_SA_load,\
                        A_interval_work, A_interval_spare, G_C0_ready, G_C0_ptw, G_C0_ptr, G_C1_ready, G_C1_ptw,\
                        G_C1_ptr, G_Cw_in_use, G_Cr_in_use, cw_loops, cr_loops,\
                        on_chip_alter, bw_in_use, communication_A_cycle, communication_B_cycle, communication_C_cycle,\
                        communication_Y_cycle, B_load, G_B_ptr_add, communication_on_chip,\
                        communication_interval, communication_internal = t.buf_process(G_A_in_use,
                                                        G_A0_ready,
                                                        G_A1_ready,
                                                        G_A0_ptr,
                                                        G_A0_ptw,
                                                        G_A1_ptr,
                                                        G_A1_ptw,
                                                        a_loops,
                                                        b_loops,
                                                        G_B0_ready,
                                                        G_B1_ready,
                                                        G_B0_ptr,
                                                        G_B0_ptw,
                                                        G_B1_ptr,
                                                        G_B1_ptw,
                                                        G_B_in_use,
                                                        B_SA_load,
                                                        A_interval_work,
                                                        A_interval_spare,
                                                        G_C0_ready,
                                                        G_C0_ptw,
                                                        G_C0_ptr,
                                                        G_C1_ready,
                                                        G_C1_ptw,
                                                        G_C1_ptr,
                                                        G_Cw_in_use,
                                                        G_Cr_in_use,
                                                        cw_loops,
                                                        cr_loops,
                                                        Array_i,
                                                        Array_j,
                                                        DR_list,
                                                        SA_list,
                                                        m_num,
                                                        n_num,
                                                        k_num,
                                                        m_num_before,
                                                        n_num_before,
                                                        k_num_before,
                                                        n_num_next,
                                                        k_num_next,
                                                        bandwidth,
                                                        compute_pipeline,
                                                        on_chip_alter,
                                                        A_buf_tile_trans,
                                                        B_buf_tile_trans,
                                                        C_buf_tile_trans,
                                                        Y_buf_tile_trans,
                                                        bw_in_use,
                                                        communication_A_cycle,
                                                        communication_B_cycle,
                                                        communication_C_cycle,
                                                        communication_Y_cycle,
                                                        A_load,
                                                        B_load,
                                                        C_load_in,
                                                        C_load_out,
                                                        G_B_ptr_add,
                                                        communication_on_chip,
                                                        communication_interval,
                                                        communication_internal
                                                        )
                        cycle += 1
                        if tracking == 1:
                            writer.writerow({'cycle':str(cycle),'G_B0_ptw':str(G_B0_ptw),'G_B1_ptw':str(G_B1_ptw),'G_B0_ptr':str(G_B0_ptr),'G_B1_ptr':str(G_B1_ptr),\
                            'G_A0_ptw':str(G_A0_ptw),'G_A1_ptw':str(G_A1_ptw),'G_A0_ptr':str(G_A0_ptr),'G_A1_ptr':str(G_A1_ptr),\
                            'G_C0_ptr':str(G_C0_ptr),'G_C1_ptr':str(G_C1_ptr),'G_C0_ptw':str(G_C0_ptw),'G_C1_ptw':str(G_C1_ptw)})
                elif m_serial==m_loop_times-1 and k_serial==k_loop_times-1 and n_serial!=n_loop_times-1:
                    A_load, B_load, C_load_in, C_load_out = 1, 1, 1, 1
                    A_load0, B_load0, C_load_in0, C_load_out0 = 1, 1, 1, 1
                    bw_in_use = 'B'
                    if G_B_in_use == 0:
                        G_B1_ready = 0
                    else:
                        G_B0_ready = 0
                    while(on_chip_alter == 0):
                        G_A_in_use, G_A0_ready, G_A1_ready, G_A0_ptr, G_A0_ptw, G_A1_ptr, G_A1_ptw, a_loops, b_loops,\
                        G_B0_ready, G_B1_ready, G_B0_ptr, G_B0_ptw, G_B1_ptr, G_B1_ptw, G_B_in_use, B_SA_load,\
                        A_interval_work, A_interval_spare, G_C0_ready, G_C0_ptw, G_C0_ptr, G_C1_ready, G_C1_ptw,\
                        G_C1_ptr, G_Cw_in_use, G_Cr_in_use, cw_loops, cr_loops,\
                        on_chip_alter, bw_in_use, communication_A_cycle, communication_B_cycle, communication_C_cycle,\
                        communication_Y_cycle, B_load, G_B_ptr_add, communication_on_chip,\
                        communication_interval, communication_internal = t.buf_process(G_A_in_use,
                                                        G_A0_ready,
                                                        G_A1_ready,
                                                        G_A0_ptr,
                                                        G_A0_ptw,
                                                        G_A1_ptr,
                                                        G_A1_ptw,
                                                        a_loops,
                                                        b_loops,
                                                        G_B0_ready,
                                                        G_B1_ready,
                                                        G_B0_ptr,
                                                        G_B0_ptw,
                                                        G_B1_ptr,
                                                        G_B1_ptw,
                                                        G_B_in_use,
                                                        B_SA_load,
                                                        A_interval_work,
                                                        A_interval_spare,
                                                        G_C0_ready,
                                                        G_C0_ptw,
                                                        G_C0_ptr,
                                                        G_C1_ready,
                                                        G_C1_ptw,
                                                        G_C1_ptr,
                                                        G_Cw_in_use,
                                                        G_Cr_in_use,
                                                        cw_loops,
                                                        cr_loops,
                                                        Array_i,
                                                        Array_j,
                                                        DR_list,
                                                        SA_list,
                                                        m_num,
                                                        n_num,
                                                        k_num,
                                                        m_num_before,
                                                        n_num_before,
                                                        k_num_before,
                                                        n_num_next,
                                                        k_num_next,
                                                        bandwidth,
                                                        compute_pipeline,
                                                        on_chip_alter,
                                                        A_buf_tile_trans,
                                                        B_buf_tile_trans,
                                                        C_buf_tile_trans,
                                                        Y_buf_tile_trans,
                                                        bw_in_use,
                                                        communication_A_cycle,
                                                        communication_B_cycle,
                                                        communication_C_cycle,
                                                        communication_Y_cycle,
                                                        A_load,
                                                        B_load,
                                                        C_load_in,
                                                        C_load_out,
                                                        G_B_ptr_add,
                                                        communication_on_chip,
                                                        communication_interval,
                                                        communication_internal
                                                        )
                        cycle += 1
                        if tracking == 1:
                            writer.writerow({'cycle':str(cycle),'G_B0_ptw':str(G_B0_ptw),'G_B1_ptw':str(G_B1_ptw),'G_B0_ptr':str(G_B0_ptr),'G_B1_ptr':str(G_B1_ptr),\
                            'G_A0_ptw':str(G_A0_ptw),'G_A1_ptw':str(G_A1_ptw),'G_A0_ptr':str(G_A0_ptr),'G_A1_ptr':str(G_A1_ptr),\
                            'G_C0_ptr':str(G_C0_ptr),'G_C1_ptr':str(G_C1_ptr),'G_C0_ptw':str(G_C0_ptw),'G_C1_ptw':str(G_C1_ptw)})
                elif m_serial==0 and k_serial==k_loop_times-1 and n_serial!=0:
                    A_load, B_load, C_load_in, C_load_out = 1, 0, 1, 1
                    A_load0, B_load0, C_load_in0, C_load_out0 = 1, 0, 1, 1
                    bw_in_use = 'B'
                    if G_B_in_use == 0:
                        G_B1_ready = 0
                    else:
                        G_B0_ready = 0
                    while(on_chip_alter == 0):
                        G_A_in_use, G_A0_ready, G_A1_ready, G_A0_ptr, G_A0_ptw, G_A1_ptr, G_A1_ptw, a_loops, b_loops,\
                        G_B0_ready, G_B1_ready, G_B0_ptr, G_B0_ptw, G_B1_ptr, G_B1_ptw, G_B_in_use, B_SA_load,\
                        A_interval_work, A_interval_spare, G_C0_ready, G_C0_ptw, G_C0_ptr, G_C1_ready, G_C1_ptw,\
                        G_C1_ptr, G_Cw_in_use, G_Cr_in_use, cw_loops, cr_loops,\
                        on_chip_alter, bw_in_use, communication_A_cycle, communication_B_cycle, communication_C_cycle,\
                        communication_Y_cycle, B_load, G_B_ptr_add, communication_on_chip,\
                        communication_interval, communication_internal = t.buf_process(G_A_in_use,
                                                        G_A0_ready,
                                                        G_A1_ready,
                                                        G_A0_ptr,
                                                        G_A0_ptw,
                                                        G_A1_ptr,
                                                        G_A1_ptw,
                                                        a_loops,
                                                        b_loops,
                                                        G_B0_ready,
                                                        G_B1_ready,
                                                        G_B0_ptr,
                                                        G_B0_ptw,
                                                        G_B1_ptr,
                                                        G_B1_ptw,
                                                        G_B_in_use,
                                                        B_SA_load,
                                                        A_interval_work,
                                                        A_interval_spare,
                                                        G_C0_ready,
                                                        G_C0_ptw,
                                                        G_C0_ptr,
                                                        G_C1_ready,
                                                        G_C1_ptw,
                                                        G_C1_ptr,
                                                        G_Cw_in_use,
                                                        G_Cr_in_use,
                                                        cw_loops,
                                                        cr_loops,
                                                        Array_i,
                                                        Array_j,
                                                        DR_list,
                                                        SA_list,
                                                        m_num,
                                                        n_num,
                                                        k_num,
                                                        m_num_before,
                                                        n_num_before,
                                                        k_num_before,
                                                        n_num_next,
                                                        k_num_next,
                                                        bandwidth,
                                                        compute_pipeline,
                                                        on_chip_alter,
                                                        A_buf_tile_trans,
                                                        B_buf_tile_trans,
                                                        C_buf_tile_trans,
                                                        Y_buf_tile_trans,
                                                        bw_in_use,
                                                        communication_A_cycle,
                                                        communication_B_cycle,
                                                        communication_C_cycle,
                                                        communication_Y_cycle,
                                                        A_load,
                                                        B_load,
                                                        C_load_in,
                                                        C_load_out,
                                                        G_B_ptr_add,
                                                        communication_on_chip,
                                                        communication_interval,
                                                        communication_internal
                                                        )
                        cycle += 1
                        if tracking == 1:
                            writer.writerow({'cycle':str(cycle),'G_B0_ptw':str(G_B0_ptw),'G_B1_ptw':str(G_B1_ptw),'G_B0_ptr':str(G_B0_ptr),'G_B1_ptr':str(G_B1_ptr),\
                            'G_A0_ptw':str(G_A0_ptw),'G_A1_ptw':str(G_A1_ptw),'G_A0_ptr':str(G_A0_ptr),'G_A1_ptr':str(G_A1_ptr),\
                            'G_C0_ptr':str(G_C0_ptr),'G_C1_ptr':str(G_C1_ptr),'G_C0_ptw':str(G_C0_ptw),'G_C1_ptw':str(G_C1_ptw)})
                else:
                    A_load, B_load, C_load_in, C_load_out = 1, 0, 0, 0
                    A_load0, B_load0, C_load_in0, C_load_out0 = 1, 0, 0, 0
                    bw_in_use = 'B'
                    if G_B_in_use == 0:
                        G_B1_ready = 0
                    else:
                        G_B0_ready = 0
                    while(on_chip_alter == 0):
                        G_A_in_use, G_A0_ready, G_A1_ready, G_A0_ptr, G_A0_ptw, G_A1_ptr, G_A1_ptw, a_loops, b_loops,\
                        G_B0_ready, G_B1_ready, G_B0_ptr, G_B0_ptw, G_B1_ptr, G_B1_ptw, G_B_in_use, B_SA_load,\
                        A_interval_work, A_interval_spare, G_C0_ready, G_C0_ptw, G_C0_ptr, G_C1_ready, G_C1_ptw,\
                        G_C1_ptr, G_Cw_in_use, G_Cr_in_use, cw_loops, cr_loops,\
                        on_chip_alter, bw_in_use, communication_A_cycle, communication_B_cycle, communication_C_cycle,\
                        communication_Y_cycle, B_load, G_B_ptr_add, communication_on_chip,\
                        communication_interval, communication_internal = t.buf_process(G_A_in_use,
                                                        G_A0_ready,
                                                        G_A1_ready,
                                                        G_A0_ptr,
                                                        G_A0_ptw,
                                                        G_A1_ptr,
                                                        G_A1_ptw,
                                                        a_loops,
                                                        b_loops,
                                                        G_B0_ready,
                                                        G_B1_ready,
                                                        G_B0_ptr,
                                                        G_B0_ptw,
                                                        G_B1_ptr,
                                                        G_B1_ptw,
                                                        G_B_in_use,
                                                        B_SA_load,
                                                        A_interval_work,
                                                        A_interval_spare,
                                                        G_C0_ready,
                                                        G_C0_ptw,
                                                        G_C0_ptr,
                                                        G_C1_ready,
                                                        G_C1_ptw,
                                                        G_C1_ptr,
                                                        G_Cw_in_use,
                                                        G_Cr_in_use,
                                                        cw_loops,
                                                        cr_loops,
                                                        Array_i,
                                                        Array_j,
                                                        DR_list,
                                                        SA_list,
                                                        m_num,
                                                        n_num,
                                                        k_num,
                                                        m_num_before,
                                                        n_num_before,
                                                        k_num_before,
                                                        n_num_next,
                                                        k_num_next,
                                                        bandwidth,
                                                        compute_pipeline,
                                                        on_chip_alter,
                                                        A_buf_tile_trans,
                                                        B_buf_tile_trans,
                                                        C_buf_tile_trans,
                                                        Y_buf_tile_trans,
                                                        bw_in_use,
                                                        communication_A_cycle,
                                                        communication_B_cycle,
                                                        communication_C_cycle,
                                                        communication_Y_cycle,
                                                        A_load,
                                                        B_load,
                                                        C_load_in,
                                                        C_load_out,
                                                        G_B_ptr_add,
                                                        communication_on_chip,
                                                        communication_interval,
                                                        communication_internal
                                                        )
                        cycle += 1
                        if tracking == 1:
                            writer.writerow({'cycle':str(cycle),'G_B0_ptw':str(G_B0_ptw),'G_B1_ptw':str(G_B1_ptw),'G_B0_ptr':str(G_B0_ptr),'G_B1_ptr':str(G_B1_ptr),\
                            'G_A0_ptw':str(G_A0_ptw),'G_A1_ptw':str(G_A1_ptw),'G_A0_ptr':str(G_A0_ptr),'G_A1_ptr':str(G_A1_ptr),\
                            'G_C0_ptr':str(G_C0_ptr),'G_C1_ptr':str(G_C1_ptr),'G_C0_ptw':str(G_C0_ptw),'G_C1_ptw':str(G_C1_ptw)})

                communication_off_chip += A_load0 * f.ceil(m_num_next * k_num_next, bandwidth)\
                + B_load0 * f.ceil(k_num_next * n_num_next, bandwidth) + C_load_in0 * partial_sum[m_serial_next][n_serial_next] * f.ceil(m_num_next * n_num_next, bandwidth)\
                + C_load_out0 * f.ceil(m_num_before * n_num_before, bandwidth)
                # 记录片外的部分和状态，“读入”到片上一次后，下次读取处于存在部分和的状态
                if C_load_in == 1:
                    partial_sum[m_serial_next][n_serial_next] = 1
                print("\r", end="")
                print("Simulation progress: {}%: ".format(int(counter/counter_total * 100)), end="")
                sys.stdout.flush()
if m_loop_times == 2 and k_loop_times >= 3 and n_loop_times >= 1:
    for n_serial in range(0,n_loop_times):
        for m_serial in range(0,m_loop_times):
            for k_serial in range(0,k_loop_times):
                counter += 1
                # 计算本次片上的mkn规模大小
                m_num, k_num, n_num = f.mkn_num(m_serial, k_serial, n_serial, m_loop_times, k_loop_times, n_loop_times, M, K, N, m, k, n)
                m_serial_next, k_serial_next, n_serial_next = f.next_num_nmk(m_serial, k_serial, n_serial, m_loop_times, k_loop_times, n_loop_times)
                m_serial_before, k_serial_before, n_serial_before = f.before_num_nmk(m_serial, k_serial, n_serial, m_loop_times, k_loop_times, n_loop_times)
                m_num_next, k_num_next, n_num_next = f.mkn_num(m_serial_next, k_serial_next, n_serial_next, m_loop_times, k_loop_times, n_loop_times, M, K, N, m, k, n)
                m_num_before, k_num_before, n_num_before = f.mkn_num(m_serial_before, k_serial_before, n_serial_before, m_loop_times, k_loop_times, n_loop_times, M, K, N, m, k, n)

                m_coordinate, k_coordinate, n_coordinate = m_serial_next * m, k_serial_next * k, n_serial_next * n

                A_buf_tile_trans = f.ceil(m_num_next * k_num_next, bandwidth)
                B_buf_tile_trans = f.ceil(k_num_next * n_num_next, bandwidth)
                # 判断C是否要从片外读入，第一次是不需要读入的
                if partial_sum[m_serial_next][n_serial_next] == 0:
                    C_buf_tile_trans = 0
                else:
                    C_buf_tile_trans = f.ceil(m_num_next * n_num_next, bandwidth)
                Y_buf_tile_trans = f.ceil(m_num_before * n_num_before, bandwidth)
                on_chip_alter = 0
                # 第一个
                if m_serial==0 and k_serial==k_loop_times-1 and n_serial==0:
                    A_load, B_load, C_load_in, C_load_out = 1, 1, 1, 0
                    A_load0, B_load0, C_load_in0, C_load_out0 = 1, 1, 1, 0
                    bw_in_use = 'B'
                    if G_B_in_use == 0:
                        G_B1_ready = 0
                    else:
                        G_B0_ready = 0
                    while(on_chip_alter == 0):
                        G_A_in_use, G_A0_ready, G_A1_ready, G_A0_ptr, G_A0_ptw, G_A1_ptr, G_A1_ptw, a_loops, b_loops,\
                        G_B0_ready, G_B1_ready, G_B0_ptr, G_B0_ptw, G_B1_ptr, G_B1_ptw, G_B_in_use, B_SA_load,\
                        A_interval_work, A_interval_spare, G_C0_ready, G_C0_ptw, G_C0_ptr, G_C1_ready, G_C1_ptw,\
                        G_C1_ptr, G_Cw_in_use, G_Cr_in_use, cw_loops, cr_loops,\
                        on_chip_alter, bw_in_use, communication_A_cycle, communication_B_cycle, communication_C_cycle,\
                        communication_Y_cycle, B_load, G_B_ptr_add, communication_on_chip,\
                        communication_interval, communication_internal = t.buf_process(G_A_in_use,
                                                        G_A0_ready,
                                                        G_A1_ready,
                                                        G_A0_ptr,
                                                        G_A0_ptw,
                                                        G_A1_ptr,
                                                        G_A1_ptw,
                                                        a_loops,
                                                        b_loops,
                                                        G_B0_ready,
                                                        G_B1_ready,
                                                        G_B0_ptr,
                                                        G_B0_ptw,
                                                        G_B1_ptr,
                                                        G_B1_ptw,
                                                        G_B_in_use,
                                                        B_SA_load,
                                                        A_interval_work,
                                                        A_interval_spare,
                                                        G_C0_ready,
                                                        G_C0_ptw,
                                                        G_C0_ptr,
                                                        G_C1_ready,
                                                        G_C1_ptw,
                                                        G_C1_ptr,
                                                        G_Cw_in_use,
                                                        G_Cr_in_use,
                                                        cw_loops,
                                                        cr_loops,
                                                        Array_i,
                                                        Array_j,
                                                        DR_list,
                                                        SA_list,
                                                        m_num,
                                                        n_num,
                                                        k_num,
                                                        m_num_before,
                                                        n_num_before,
                                                        k_num_before,
                                                        n_num_next,
                                                        k_num_next,
                                                        bandwidth,
                                                        compute_pipeline,
                                                        on_chip_alter,
                                                        A_buf_tile_trans,
                                                        B_buf_tile_trans,
                                                        C_buf_tile_trans,
                                                        Y_buf_tile_trans,
                                                        bw_in_use,
                                                        communication_A_cycle,
                                                        communication_B_cycle,
                                                        communication_C_cycle,
                                                        communication_Y_cycle,
                                                        A_load,
                                                        B_load,
                                                        C_load_in,
                                                        C_load_out,
                                                        G_B_ptr_add,
                                                        communication_on_chip,
                                                        communication_interval,
                                                        communication_internal
                                                        )
                        cycle += 1
                        if tracking == 1:
                            writer.writerow({'cycle':str(cycle),'G_B0_ptw':str(G_B0_ptw),'G_B1_ptw':str(G_B1_ptw),'G_B0_ptr':str(G_B0_ptr),'G_B1_ptr':str(G_B1_ptr),\
                            'G_A0_ptw':str(G_A0_ptw),'G_A1_ptw':str(G_A1_ptw),'G_A0_ptr':str(G_A0_ptr),'G_A1_ptr':str(G_A1_ptr),\
                            'G_C0_ptr':str(G_C0_ptr),'G_C1_ptr':str(G_C1_ptr),'G_C0_ptw':str(G_C0_ptw),'G_C1_ptw':str(G_C1_ptw)})
                # 最后一个
                elif m_serial==m_loop_times-1 and k_serial==k_loop_times-1 and n_serial==n_loop_times-1:
                    A_load, B_load, C_load_in, C_load_out = 0, 0, 0, 1
                    A_load0, B_load0, C_load_in0, C_load_out0 = 0, 0, 0, 1
                    while(on_chip_alter == 0):
                        G_A_in_use, G_A0_ready, G_A1_ready, G_A0_ptr, G_A0_ptw, G_A1_ptr, G_A1_ptw, a_loops, b_loops,\
                        G_B0_ready, G_B1_ready, G_B0_ptr, G_B0_ptw, G_B1_ptr, G_B1_ptw, G_B_in_use, B_SA_load,\
                        A_interval_work, A_interval_spare, G_C0_ready, G_C0_ptw, G_C0_ptr, G_C1_ready, G_C1_ptw,\
                        G_C1_ptr, G_Cw_in_use, G_Cr_in_use, cw_loops, cr_loops,\
                        on_chip_alter, bw_in_use, communication_Y_cycle, G_B_ptr_add, communication_on_chip,\
                        communication_interval, communication_internal = tlast.buf_process(G_A_in_use,
                                                        G_A0_ready,
                                                        G_A1_ready,
                                                        G_A0_ptr,
                                                        G_A0_ptw,
                                                        G_A1_ptr,
                                                        G_A1_ptw,
                                                        a_loops,
                                                        b_loops,
                                                        G_B0_ready,
                                                        G_B1_ready,
                                                        G_B0_ptr,
                                                        G_B0_ptw,
                                                        G_B1_ptr,
                                                        G_B1_ptw,
                                                        G_B_in_use,
                                                        B_SA_load,
                                                        A_interval_work,
                                                        A_interval_spare,
                                                        G_C0_ready,
                                                        G_C0_ptw,
                                                        G_C0_ptr,
                                                        G_C1_ready,
                                                        G_C1_ptw,
                                                        G_C1_ptr,
                                                        G_Cw_in_use,
                                                        G_Cr_in_use,
                                                        cw_loops,
                                                        cr_loops,
                                                        Array_i,
                                                        Array_j,
                                                        DR_list,
                                                        SA_list,
                                                        m_num,
                                                        n_num,
                                                        k_num,
                                                        m_num_before,
                                                        n_num_before,
                                                        k_num_before,
                                                        n_num_next,
                                                        k_num_next,
                                                        bandwidth,
                                                        compute_pipeline,
                                                        on_chip_alter,
                                                        A_buf_tile_trans,
                                                        B_buf_tile_trans,
                                                        C_buf_tile_trans,
                                                        Y_buf_tile_trans,
                                                        bw_in_use,
                                                        communication_Y_cycle,
                                                        C_load_out,
                                                        G_B_ptr_add,
                                                        communication_on_chip,
                                                        communication_interval,
                                                        communication_internal
                                                        )
                        cycle += 1
                        if tracking == 1:
                            writer.writerow({'cycle':str(cycle),'G_B0_ptw':str(G_B0_ptw),'G_B1_ptw':str(G_B1_ptw),'G_B0_ptr':str(G_B0_ptr),'G_B1_ptr':str(G_B1_ptr),\
                            'G_A0_ptw':str(G_A0_ptw),'G_A1_ptw':str(G_A1_ptw),'G_A0_ptr':str(G_A0_ptr),'G_A1_ptr':str(G_A1_ptr),\
                            'G_C0_ptr':str(G_C0_ptr),'G_C1_ptr':str(G_C1_ptr),'G_C0_ptw':str(G_C0_ptw),'G_C1_ptw':str(G_C1_ptw)})
                elif k_serial!=k_loop_times-1:
                    A_load, B_load, C_load_in, C_load_out = 1, 1, 0, 0
                    A_load0, B_load0, C_load_in0, C_load_out0 = 1, 1, 0, 0
                    bw_in_use = 'B'
                    if G_B_in_use == 0:
                        G_B1_ready = 0
                    else:
                        G_B0_ready = 0
                    while(on_chip_alter == 0):
                        G_A_in_use, G_A0_ready, G_A1_ready, G_A0_ptr, G_A0_ptw, G_A1_ptr, G_A1_ptw, a_loops, b_loops,\
                        G_B0_ready, G_B1_ready, G_B0_ptr, G_B0_ptw, G_B1_ptr, G_B1_ptw, G_B_in_use, B_SA_load,\
                        A_interval_work, A_interval_spare, G_C0_ready, G_C0_ptw, G_C0_ptr, G_C1_ready, G_C1_ptw,\
                        G_C1_ptr, G_Cw_in_use, G_Cr_in_use, cw_loops, cr_loops,\
                        on_chip_alter, bw_in_use, communication_A_cycle, communication_B_cycle, communication_C_cycle,\
                        communication_Y_cycle, B_load, G_B_ptr_add, communication_on_chip,\
                        communication_interval, communication_internal = t.buf_process(G_A_in_use,
                                                        G_A0_ready,
                                                        G_A1_ready,
                                                        G_A0_ptr,
                                                        G_A0_ptw,
                                                        G_A1_ptr,
                                                        G_A1_ptw,
                                                        a_loops,
                                                        b_loops,
                                                        G_B0_ready,
                                                        G_B1_ready,
                                                        G_B0_ptr,
                                                        G_B0_ptw,
                                                        G_B1_ptr,
                                                        G_B1_ptw,
                                                        G_B_in_use,
                                                        B_SA_load,
                                                        A_interval_work,
                                                        A_interval_spare,
                                                        G_C0_ready,
                                                        G_C0_ptw,
                                                        G_C0_ptr,
                                                        G_C1_ready,
                                                        G_C1_ptw,
                                                        G_C1_ptr,
                                                        G_Cw_in_use,
                                                        G_Cr_in_use,
                                                        cw_loops,
                                                        cr_loops,
                                                        Array_i,
                                                        Array_j,
                                                        DR_list,
                                                        SA_list,
                                                        m_num,
                                                        n_num,
                                                        k_num,
                                                        m_num_before,
                                                        n_num_before,
                                                        k_num_before,
                                                        n_num_next,
                                                        k_num_next,
                                                        bandwidth,
                                                        compute_pipeline,
                                                        on_chip_alter,
                                                        A_buf_tile_trans,
                                                        B_buf_tile_trans,
                                                        C_buf_tile_trans,
                                                        Y_buf_tile_trans,
                                                        bw_in_use,
                                                        communication_A_cycle,
                                                        communication_B_cycle,
                                                        communication_C_cycle,
                                                        communication_Y_cycle,
                                                        A_load,
                                                        B_load,
                                                        C_load_in,
                                                        C_load_out,
                                                        G_B_ptr_add,
                                                        communication_on_chip,
                                                        communication_interval,
                                                        communication_internal
                                                        )
                        cycle += 1
                        if tracking == 1:
                            writer.writerow({'cycle':str(cycle),'G_B0_ptw':str(G_B0_ptw),'G_B1_ptw':str(G_B1_ptw),'G_B0_ptr':str(G_B0_ptr),'G_B1_ptr':str(G_B1_ptr),\
                            'G_A0_ptw':str(G_A0_ptw),'G_A1_ptw':str(G_A1_ptw),'G_A0_ptr':str(G_A0_ptr),'G_A1_ptr':str(G_A1_ptr),\
                            'G_C0_ptr':str(G_C0_ptr),'G_C1_ptr':str(G_C1_ptr),'G_C0_ptw':str(G_C0_ptw),'G_C1_ptw':str(G_C1_ptw)})
                else:
                    A_load, B_load, C_load_in, C_load_out = 1, 1, 1, 1
                    A_load0, B_load0, C_load_in0, C_load_out0 = 1, 1, 1, 1
                    bw_in_use = 'B'
                    if G_B_in_use == 0:
                        G_B1_ready = 0
                    else:
                        G_B0_ready = 0
                    while(on_chip_alter == 0):
                        G_A_in_use, G_A0_ready, G_A1_ready, G_A0_ptr, G_A0_ptw, G_A1_ptr, G_A1_ptw, a_loops, b_loops,\
                        G_B0_ready, G_B1_ready, G_B0_ptr, G_B0_ptw, G_B1_ptr, G_B1_ptw, G_B_in_use, B_SA_load,\
                        A_interval_work, A_interval_spare, G_C0_ready, G_C0_ptw, G_C0_ptr, G_C1_ready, G_C1_ptw,\
                        G_C1_ptr, G_Cw_in_use, G_Cr_in_use, cw_loops, cr_loops,\
                        on_chip_alter, bw_in_use, communication_A_cycle, communication_B_cycle, communication_C_cycle,\
                        communication_Y_cycle, B_load, G_B_ptr_add, communication_on_chip,\
                        communication_interval, communication_internal = t.buf_process(G_A_in_use,
                                                        G_A0_ready,
                                                        G_A1_ready,
                                                        G_A0_ptr,
                                                        G_A0_ptw,
                                                        G_A1_ptr,
                                                        G_A1_ptw,
                                                        a_loops,
                                                        b_loops,
                                                        G_B0_ready,
                                                        G_B1_ready,
                                                        G_B0_ptr,
                                                        G_B0_ptw,
                                                        G_B1_ptr,
                                                        G_B1_ptw,
                                                        G_B_in_use,
                                                        B_SA_load,
                                                        A_interval_work,
                                                        A_interval_spare,
                                                        G_C0_ready,
                                                        G_C0_ptw,
                                                        G_C0_ptr,
                                                        G_C1_ready,
                                                        G_C1_ptw,
                                                        G_C1_ptr,
                                                        G_Cw_in_use,
                                                        G_Cr_in_use,
                                                        cw_loops,
                                                        cr_loops,
                                                        Array_i,
                                                        Array_j,
                                                        DR_list,
                                                        SA_list,
                                                        m_num,
                                                        n_num,
                                                        k_num,
                                                        m_num_before,
                                                        n_num_before,
                                                        k_num_before,
                                                        n_num_next,
                                                        k_num_next,
                                                        bandwidth,
                                                        compute_pipeline,
                                                        on_chip_alter,
                                                        A_buf_tile_trans,
                                                        B_buf_tile_trans,
                                                        C_buf_tile_trans,
                                                        Y_buf_tile_trans,
                                                        bw_in_use,
                                                        communication_A_cycle,
                                                        communication_B_cycle,
                                                        communication_C_cycle,
                                                        communication_Y_cycle,
                                                        A_load,
                                                        B_load,
                                                        C_load_in,
                                                        C_load_out,
                                                        G_B_ptr_add,
                                                        communication_on_chip,
                                                        communication_interval,
                                                        communication_internal
                                                        )
                        cycle += 1
                        if tracking == 1:
                            writer.writerow({'cycle':str(cycle),'G_B0_ptw':str(G_B0_ptw),'G_B1_ptw':str(G_B1_ptw),'G_B0_ptr':str(G_B0_ptr),'G_B1_ptr':str(G_B1_ptr),\
                            'G_A0_ptw':str(G_A0_ptw),'G_A1_ptw':str(G_A1_ptw),'G_A0_ptr':str(G_A0_ptr),'G_A1_ptr':str(G_A1_ptr),\
                            'G_C0_ptr':str(G_C0_ptr),'G_C1_ptr':str(G_C1_ptr),'G_C0_ptw':str(G_C0_ptw),'G_C1_ptw':str(G_C1_ptw)})

                communication_off_chip += A_load0 * f.ceil(m_num_next * k_num_next, bandwidth)\
                + B_load0 * f.ceil(k_num_next * n_num_next, bandwidth) + C_load_in0 * partial_sum[m_serial_next][n_serial_next] * f.ceil(m_num_next * n_num_next, bandwidth)\
                + C_load_out0 * f.ceil(m_num_before * n_num_before, bandwidth)
                # 记录片外的部分和状态，“读入”到片上一次后，下次读取处于存在部分和的状态
                if C_load_in == 1:
                    partial_sum[m_serial_next][n_serial_next] = 1
                print("\r", end="")
                print("Simulation progress: {}%: ".format(int(counter/counter_total * 100)), end="")
                sys.stdout.flush()
if m_loop_times >= 3 and k_loop_times == 1 and n_loop_times >= 2:
    for n_serial in range(0,n_loop_times):
        for m_serial in range(0,m_loop_times):
            for k_serial in range(0,k_loop_times):
                counter += 1
                # 计算本次片上的mkn规模大小
                m_num, k_num, n_num = f.mkn_num(m_serial, k_serial, n_serial, m_loop_times, k_loop_times, n_loop_times, M, K, N, m, k, n)
                m_serial_next, k_serial_next, n_serial_next = f.next_num_nmk(m_serial, k_serial, n_serial, m_loop_times, k_loop_times, n_loop_times)
                m_serial_before, k_serial_before, n_serial_before = f.before_num_nmk(m_serial, k_serial, n_serial, m_loop_times, k_loop_times, n_loop_times)
                m_num_next, k_num_next, n_num_next = f.mkn_num(m_serial_next, k_serial_next, n_serial_next, m_loop_times, k_loop_times, n_loop_times, M, K, N, m, k, n)
                m_num_before, k_num_before, n_num_before = f.mkn_num(m_serial_before, k_serial_before, n_serial_before, m_loop_times, k_loop_times, n_loop_times, M, K, N, m, k, n)

                m_coordinate, k_coordinate, n_coordinate = m_serial_next * m, k_serial_next * k, n_serial_next * n

                A_buf_tile_trans = f.ceil(m_num_next * k_num_next, bandwidth)
                B_buf_tile_trans = f.ceil(k_num_next * n_num_next, bandwidth)
                # 判断C是否要从片外读入，第一次是不需要读入的
                if partial_sum[m_serial_next][n_serial_next] == 0:
                    C_buf_tile_trans = 0
                else:
                    C_buf_tile_trans = f.ceil(m_num_next * n_num_next, bandwidth)
                Y_buf_tile_trans = f.ceil(m_num_before * n_num_before, bandwidth)
                on_chip_alter = 0
                # 第一个
                if m_serial==0 and k_serial==0 and n_serial==0:
                    A_load, B_load, C_load_in, C_load_out = 1, 0, 1, 0
                    A_load0, B_load0, C_load_in0, C_load_out0 = 1, 0, 1, 0
                    while(on_chip_alter == 0):
                        G_A_in_use, G_A0_ready, G_A1_ready, G_A0_ptr, G_A0_ptw, G_A1_ptr, G_A1_ptw, a_loops, b_loops,\
                        G_B0_ready, G_B1_ready, G_B0_ptr, G_B0_ptw, G_B1_ptr, G_B1_ptw, G_B_in_use, B_SA_load,\
                        A_interval_work, A_interval_spare, G_C0_ready, G_C0_ptw, G_C0_ptr, G_C1_ready, G_C1_ptw,\
                        G_C1_ptr, G_Cw_in_use, G_Cr_in_use, cw_loops, cr_loops,\
                        on_chip_alter, bw_in_use, communication_A_cycle, communication_B_cycle, communication_C_cycle,\
                        communication_Y_cycle, B_load, G_B_ptr_add, communication_on_chip,\
                        communication_interval, communication_internal = t.buf_process(G_A_in_use,
                                                        G_A0_ready,
                                                        G_A1_ready,
                                                        G_A0_ptr,
                                                        G_A0_ptw,
                                                        G_A1_ptr,
                                                        G_A1_ptw,
                                                        a_loops,
                                                        b_loops,
                                                        G_B0_ready,
                                                        G_B1_ready,
                                                        G_B0_ptr,
                                                        G_B0_ptw,
                                                        G_B1_ptr,
                                                        G_B1_ptw,
                                                        G_B_in_use,
                                                        B_SA_load,
                                                        A_interval_work,
                                                        A_interval_spare,
                                                        G_C0_ready,
                                                        G_C0_ptw,
                                                        G_C0_ptr,
                                                        G_C1_ready,
                                                        G_C1_ptw,
                                                        G_C1_ptr,
                                                        G_Cw_in_use,
                                                        G_Cr_in_use,
                                                        cw_loops,
                                                        cr_loops,
                                                        Array_i,
                                                        Array_j,
                                                        DR_list,
                                                        SA_list,
                                                        m_num,
                                                        n_num,
                                                        k_num,
                                                        m_num_before,
                                                        n_num_before,
                                                        k_num_before,
                                                        n_num_next,
                                                        k_num_next,
                                                        bandwidth,
                                                        compute_pipeline,
                                                        on_chip_alter,
                                                        A_buf_tile_trans,
                                                        B_buf_tile_trans,
                                                        C_buf_tile_trans,
                                                        Y_buf_tile_trans,
                                                        bw_in_use,
                                                        communication_A_cycle,
                                                        communication_B_cycle,
                                                        communication_C_cycle,
                                                        communication_Y_cycle,
                                                        A_load,
                                                        B_load,
                                                        C_load_in,
                                                        C_load_out,
                                                        G_B_ptr_add,
                                                        communication_on_chip,
                                                        communication_interval,
                                                        communication_internal
                                                        )
                        cycle += 1
                        if tracking == 1:
                            writer.writerow({'cycle':str(cycle),'G_B0_ptw':str(G_B0_ptw),'G_B1_ptw':str(G_B1_ptw),'G_B0_ptr':str(G_B0_ptr),'G_B1_ptr':str(G_B1_ptr),\
                            'G_A0_ptw':str(G_A0_ptw),'G_A1_ptw':str(G_A1_ptw),'G_A0_ptr':str(G_A0_ptr),'G_A1_ptr':str(G_A1_ptr),\
                            'G_C0_ptr':str(G_C0_ptr),'G_C1_ptr':str(G_C1_ptr),'G_C0_ptw':str(G_C0_ptw),'G_C1_ptw':str(G_C1_ptw)})
                # 最后一个
                elif m_serial==m_loop_times-1 and k_serial==k_loop_times-1 and n_serial==n_loop_times-1:
                    A_load, B_load, C_load_in, C_load_out = 0, 0, 0, 1
                    A_load0, B_load0, C_load_in0, C_load_out0 = 0, 0, 0, 1
                    while(on_chip_alter == 0):
                        G_A_in_use, G_A0_ready, G_A1_ready, G_A0_ptr, G_A0_ptw, G_A1_ptr, G_A1_ptw, a_loops, b_loops,\
                        G_B0_ready, G_B1_ready, G_B0_ptr, G_B0_ptw, G_B1_ptr, G_B1_ptw, G_B_in_use, B_SA_load,\
                        A_interval_work, A_interval_spare, G_C0_ready, G_C0_ptw, G_C0_ptr, G_C1_ready, G_C1_ptw,\
                        G_C1_ptr, G_Cw_in_use, G_Cr_in_use, cw_loops, cr_loops,\
                        on_chip_alter, bw_in_use, communication_Y_cycle, G_B_ptr_add, communication_on_chip,\
                        communication_interval, communication_internal = tlast.buf_process(G_A_in_use,
                                                        G_A0_ready,
                                                        G_A1_ready,
                                                        G_A0_ptr,
                                                        G_A0_ptw,
                                                        G_A1_ptr,
                                                        G_A1_ptw,
                                                        a_loops,
                                                        b_loops,
                                                        G_B0_ready,
                                                        G_B1_ready,
                                                        G_B0_ptr,
                                                        G_B0_ptw,
                                                        G_B1_ptr,
                                                        G_B1_ptw,
                                                        G_B_in_use,
                                                        B_SA_load,
                                                        A_interval_work,
                                                        A_interval_spare,
                                                        G_C0_ready,
                                                        G_C0_ptw,
                                                        G_C0_ptr,
                                                        G_C1_ready,
                                                        G_C1_ptw,
                                                        G_C1_ptr,
                                                        G_Cw_in_use,
                                                        G_Cr_in_use,
                                                        cw_loops,
                                                        cr_loops,
                                                        Array_i,
                                                        Array_j,
                                                        DR_list,
                                                        SA_list,
                                                        m_num,
                                                        n_num,
                                                        k_num,
                                                        m_num_before,
                                                        n_num_before,
                                                        k_num_before,
                                                        n_num_next,
                                                        k_num_next,
                                                        bandwidth,
                                                        compute_pipeline,
                                                        on_chip_alter,
                                                        A_buf_tile_trans,
                                                        B_buf_tile_trans,
                                                        C_buf_tile_trans,
                                                        Y_buf_tile_trans,
                                                        bw_in_use,
                                                        communication_Y_cycle,
                                                        C_load_out,
                                                        G_B_ptr_add,
                                                        communication_on_chip,
                                                        communication_interval,
                                                        communication_internal
                                                        )
                        cycle += 1
                        if tracking == 1:
                            writer.writerow({'cycle':str(cycle),'G_B0_ptw':str(G_B0_ptw),'G_B1_ptw':str(G_B1_ptw),'G_B0_ptr':str(G_B0_ptr),'G_B1_ptr':str(G_B1_ptr),\
                            'G_A0_ptw':str(G_A0_ptw),'G_A1_ptw':str(G_A1_ptw),'G_A0_ptr':str(G_A0_ptr),'G_A1_ptr':str(G_A1_ptr),\
                            'G_C0_ptr':str(G_C0_ptr),'G_C1_ptr':str(G_C1_ptr),'G_C0_ptw':str(G_C0_ptw),'G_C1_ptw':str(G_C1_ptw)})
                elif m_serial==m_loop_times-1 and k_serial==0 and n_serial!=n_loop_times-1:
                    A_load, B_load, C_load_in, C_load_out = 1, 1, 1, 1
                    A_load0, B_load0, C_load_in0, C_load_out0 = 1, 1, 1, 1
                    bw_in_use = 'B'
                    if G_B_in_use == 0:
                        G_B1_ready = 0
                    else:
                        G_B0_ready = 0
                    while(on_chip_alter == 0):
                        G_A_in_use, G_A0_ready, G_A1_ready, G_A0_ptr, G_A0_ptw, G_A1_ptr, G_A1_ptw, a_loops, b_loops,\
                        G_B0_ready, G_B1_ready, G_B0_ptr, G_B0_ptw, G_B1_ptr, G_B1_ptw, G_B_in_use, B_SA_load,\
                        A_interval_work, A_interval_spare, G_C0_ready, G_C0_ptw, G_C0_ptr, G_C1_ready, G_C1_ptw,\
                        G_C1_ptr, G_Cw_in_use, G_Cr_in_use, cw_loops, cr_loops,\
                        on_chip_alter, bw_in_use, communication_A_cycle, communication_B_cycle, communication_C_cycle,\
                        communication_Y_cycle, B_load, G_B_ptr_add, communication_on_chip,\
                        communication_interval, communication_internal = t.buf_process(G_A_in_use,
                                                        G_A0_ready,
                                                        G_A1_ready,
                                                        G_A0_ptr,
                                                        G_A0_ptw,
                                                        G_A1_ptr,
                                                        G_A1_ptw,
                                                        a_loops,
                                                        b_loops,
                                                        G_B0_ready,
                                                        G_B1_ready,
                                                        G_B0_ptr,
                                                        G_B0_ptw,
                                                        G_B1_ptr,
                                                        G_B1_ptw,
                                                        G_B_in_use,
                                                        B_SA_load,
                                                        A_interval_work,
                                                        A_interval_spare,
                                                        G_C0_ready,
                                                        G_C0_ptw,
                                                        G_C0_ptr,
                                                        G_C1_ready,
                                                        G_C1_ptw,
                                                        G_C1_ptr,
                                                        G_Cw_in_use,
                                                        G_Cr_in_use,
                                                        cw_loops,
                                                        cr_loops,
                                                        Array_i,
                                                        Array_j,
                                                        DR_list,
                                                        SA_list,
                                                        m_num,
                                                        n_num,
                                                        k_num,
                                                        m_num_before,
                                                        n_num_before,
                                                        k_num_before,
                                                        n_num_next,
                                                        k_num_next,
                                                        bandwidth,
                                                        compute_pipeline,
                                                        on_chip_alter,
                                                        A_buf_tile_trans,
                                                        B_buf_tile_trans,
                                                        C_buf_tile_trans,
                                                        Y_buf_tile_trans,
                                                        bw_in_use,
                                                        communication_A_cycle,
                                                        communication_B_cycle,
                                                        communication_C_cycle,
                                                        communication_Y_cycle,
                                                        A_load,
                                                        B_load,
                                                        C_load_in,
                                                        C_load_out,
                                                        G_B_ptr_add,
                                                        communication_on_chip,
                                                        communication_interval,
                                                        communication_internal
                                                        )
                        cycle += 1
                        if tracking == 1:
                            writer.writerow({'cycle':str(cycle),'G_B0_ptw':str(G_B0_ptw),'G_B1_ptw':str(G_B1_ptw),'G_B0_ptr':str(G_B0_ptr),'G_B1_ptr':str(G_B1_ptr),\
                            'G_A0_ptw':str(G_A0_ptw),'G_A1_ptw':str(G_A1_ptw),'G_A0_ptr':str(G_A0_ptr),'G_A1_ptr':str(G_A1_ptr),\
                            'G_C0_ptr':str(G_C0_ptr),'G_C1_ptr':str(G_C1_ptr),'G_C0_ptw':str(G_C0_ptw),'G_C1_ptw':str(G_C1_ptw)})
                else:
                    A_load, B_load, C_load_in, C_load_out = 1, 0, 1, 1
                    A_load0, B_load0, C_load_in0, C_load_out0 = 1, 0, 1, 1
                    bw_in_use = 'B'
                    if G_B_in_use == 0:
                        G_B1_ready = 0
                    else:
                        G_B0_ready = 0
                    while(on_chip_alter == 0):
                        G_A_in_use, G_A0_ready, G_A1_ready, G_A0_ptr, G_A0_ptw, G_A1_ptr, G_A1_ptw, a_loops, b_loops,\
                        G_B0_ready, G_B1_ready, G_B0_ptr, G_B0_ptw, G_B1_ptr, G_B1_ptw, G_B_in_use, B_SA_load,\
                        A_interval_work, A_interval_spare, G_C0_ready, G_C0_ptw, G_C0_ptr, G_C1_ready, G_C1_ptw,\
                        G_C1_ptr, G_Cw_in_use, G_Cr_in_use, cw_loops, cr_loops,\
                        on_chip_alter, bw_in_use, communication_A_cycle, communication_B_cycle, communication_C_cycle,\
                        communication_Y_cycle, B_load, G_B_ptr_add, communication_on_chip,\
                        communication_interval, communication_internal = t.buf_process(G_A_in_use,
                                                        G_A0_ready,
                                                        G_A1_ready,
                                                        G_A0_ptr,
                                                        G_A0_ptw,
                                                        G_A1_ptr,
                                                        G_A1_ptw,
                                                        a_loops,
                                                        b_loops,
                                                        G_B0_ready,
                                                        G_B1_ready,
                                                        G_B0_ptr,
                                                        G_B0_ptw,
                                                        G_B1_ptr,
                                                        G_B1_ptw,
                                                        G_B_in_use,
                                                        B_SA_load,
                                                        A_interval_work,
                                                        A_interval_spare,
                                                        G_C0_ready,
                                                        G_C0_ptw,
                                                        G_C0_ptr,
                                                        G_C1_ready,
                                                        G_C1_ptw,
                                                        G_C1_ptr,
                                                        G_Cw_in_use,
                                                        G_Cr_in_use,
                                                        cw_loops,
                                                        cr_loops,
                                                        Array_i,
                                                        Array_j,
                                                        DR_list,
                                                        SA_list,
                                                        m_num,
                                                        n_num,
                                                        k_num,
                                                        m_num_before,
                                                        n_num_before,
                                                        k_num_before,
                                                        n_num_next,
                                                        k_num_next,
                                                        bandwidth,
                                                        compute_pipeline,
                                                        on_chip_alter,
                                                        A_buf_tile_trans,
                                                        B_buf_tile_trans,
                                                        C_buf_tile_trans,
                                                        Y_buf_tile_trans,
                                                        bw_in_use,
                                                        communication_A_cycle,
                                                        communication_B_cycle,
                                                        communication_C_cycle,
                                                        communication_Y_cycle,
                                                        A_load,
                                                        B_load,
                                                        C_load_in,
                                                        C_load_out,
                                                        G_B_ptr_add,
                                                        communication_on_chip,
                                                        communication_interval,
                                                        communication_internal
                                                        )
                        cycle += 1
                        if tracking == 1:
                            writer.writerow({'cycle':str(cycle),'G_B0_ptw':str(G_B0_ptw),'G_B1_ptw':str(G_B1_ptw),'G_B0_ptr':str(G_B0_ptr),'G_B1_ptr':str(G_B1_ptr),\
                            'G_A0_ptw':str(G_A0_ptw),'G_A1_ptw':str(G_A1_ptw),'G_A0_ptr':str(G_A0_ptr),'G_A1_ptr':str(G_A1_ptr),\
                            'G_C0_ptr':str(G_C0_ptr),'G_C1_ptr':str(G_C1_ptr),'G_C0_ptw':str(G_C0_ptw),'G_C1_ptw':str(G_C1_ptw)})

                communication_off_chip += A_load0 * f.ceil(m_num_next * k_num_next, bandwidth)\
                + B_load0 * f.ceil(k_num_next * n_num_next, bandwidth) + C_load_in0 * partial_sum[m_serial_next][n_serial_next] * f.ceil(m_num_next * n_num_next, bandwidth)\
                + C_load_out0 * f.ceil(m_num_before * n_num_before, bandwidth)
                # 记录片外的部分和状态，“读入”到片上一次后，下次读取处于存在部分和的状态
                if C_load_in == 1:
                    partial_sum[m_serial_next][n_serial_next] = 1
                print("\r", end="")
                print("Simulation progress: {}%: ".format(int(counter/counter_total * 100)), end="")
                sys.stdout.flush()
if m_loop_times >= 3 and k_loop_times == 2 and n_loop_times >= 1:
    for n_serial in range(0,n_loop_times):
        for m_serial in range(0,m_loop_times):
            for k_serial in range(0,k_loop_times):
                counter += 1
                # 计算本次片上的mkn规模大小
                m_num, k_num, n_num = f.mkn_num(m_serial, k_serial, n_serial, m_loop_times, k_loop_times, n_loop_times, M, K, N, m, k, n)
                m_serial_next, k_serial_next, n_serial_next = f.next_num_nmk(m_serial, k_serial, n_serial, m_loop_times, k_loop_times, n_loop_times)
                m_serial_before, k_serial_before, n_serial_before = f.before_num_nmk(m_serial, k_serial, n_serial, m_loop_times, k_loop_times, n_loop_times)
                m_num_next, k_num_next, n_num_next = f.mkn_num(m_serial_next, k_serial_next, n_serial_next, m_loop_times, k_loop_times, n_loop_times, M, K, N, m, k, n)
                m_num_before, k_num_before, n_num_before = f.mkn_num(m_serial_before, k_serial_before, n_serial_before, m_loop_times, k_loop_times, n_loop_times, M, K, N, m, k, n)

                m_coordinate, k_coordinate, n_coordinate = m_serial_next * m, k_serial_next * k, n_serial_next * n

                A_buf_tile_trans = f.ceil(m_num_next * k_num_next, bandwidth)
                B_buf_tile_trans = f.ceil(k_num_next * n_num_next, bandwidth)
                # 判断C是否要从片外读入，第一次是不需要读入的
                if partial_sum[m_serial_next][n_serial_next] == 0:
                    C_buf_tile_trans = 0
                else:
                    C_buf_tile_trans = f.ceil(m_num_next * n_num_next, bandwidth)
                Y_buf_tile_trans = f.ceil(m_num_before * n_num_before, bandwidth)
                on_chip_alter = 0
                # 第一个
                if m_serial==0 and k_serial==k_loop_times-1 and n_serial==0:
                    A_load, B_load, C_load_in, C_load_out = 1, 0, 1, 0
                    A_load0, B_load0, C_load_in0, C_load_out0 = 1, 0, 1, 0
                    bw_in_use = 'B'
                    if G_B_in_use == 0:
                        G_B1_ready = 0
                    else:
                        G_B0_ready = 0
                    while(on_chip_alter == 0):
                        G_A_in_use, G_A0_ready, G_A1_ready, G_A0_ptr, G_A0_ptw, G_A1_ptr, G_A1_ptw, a_loops, b_loops,\
                        G_B0_ready, G_B1_ready, G_B0_ptr, G_B0_ptw, G_B1_ptr, G_B1_ptw, G_B_in_use, B_SA_load,\
                        A_interval_work, A_interval_spare, G_C0_ready, G_C0_ptw, G_C0_ptr, G_C1_ready, G_C1_ptw,\
                        G_C1_ptr, G_Cw_in_use, G_Cr_in_use, cw_loops, cr_loops,\
                        on_chip_alter, bw_in_use, communication_A_cycle, communication_B_cycle, communication_C_cycle,\
                        communication_Y_cycle, B_load, G_B_ptr_add, communication_on_chip,\
                        communication_interval, communication_internal = t.buf_process(G_A_in_use,
                                                        G_A0_ready,
                                                        G_A1_ready,
                                                        G_A0_ptr,
                                                        G_A0_ptw,
                                                        G_A1_ptr,
                                                        G_A1_ptw,
                                                        a_loops,
                                                        b_loops,
                                                        G_B0_ready,
                                                        G_B1_ready,
                                                        G_B0_ptr,
                                                        G_B0_ptw,
                                                        G_B1_ptr,
                                                        G_B1_ptw,
                                                        G_B_in_use,
                                                        B_SA_load,
                                                        A_interval_work,
                                                        A_interval_spare,
                                                        G_C0_ready,
                                                        G_C0_ptw,
                                                        G_C0_ptr,
                                                        G_C1_ready,
                                                        G_C1_ptw,
                                                        G_C1_ptr,
                                                        G_Cw_in_use,
                                                        G_Cr_in_use,
                                                        cw_loops,
                                                        cr_loops,
                                                        Array_i,
                                                        Array_j,
                                                        DR_list,
                                                        SA_list,
                                                        m_num,
                                                        n_num,
                                                        k_num,
                                                        m_num_before,
                                                        n_num_before,
                                                        k_num_before,
                                                        n_num_next,
                                                        k_num_next,
                                                        bandwidth,
                                                        compute_pipeline,
                                                        on_chip_alter,
                                                        A_buf_tile_trans,
                                                        B_buf_tile_trans,
                                                        C_buf_tile_trans,
                                                        Y_buf_tile_trans,
                                                        bw_in_use,
                                                        communication_A_cycle,
                                                        communication_B_cycle,
                                                        communication_C_cycle,
                                                        communication_Y_cycle,
                                                        A_load,
                                                        B_load,
                                                        C_load_in,
                                                        C_load_out,
                                                        G_B_ptr_add,
                                                        communication_on_chip,
                                                        communication_interval,
                                                        communication_internal
                                                        )
                        cycle += 1
                        if tracking == 1:
                            writer.writerow({'cycle':str(cycle),'G_B0_ptw':str(G_B0_ptw),'G_B1_ptw':str(G_B1_ptw),'G_B0_ptr':str(G_B0_ptr),'G_B1_ptr':str(G_B1_ptr),\
                            'G_A0_ptw':str(G_A0_ptw),'G_A1_ptw':str(G_A1_ptw),'G_A0_ptr':str(G_A0_ptr),'G_A1_ptr':str(G_A1_ptr),\
                            'G_C0_ptr':str(G_C0_ptr),'G_C1_ptr':str(G_C1_ptr),'G_C0_ptw':str(G_C0_ptw),'G_C1_ptw':str(G_C1_ptw)})
                # 最后一个
                elif m_serial==m_loop_times-1 and k_serial==k_loop_times-1 and n_serial==n_loop_times-1:
                    A_load, B_load, C_load_in, C_load_out = 0, 0, 0, 1
                    A_load0, B_load0, C_load_in0, C_load_out0 = 0, 0, 0, 1
                    while(on_chip_alter == 0):
                        G_A_in_use, G_A0_ready, G_A1_ready, G_A0_ptr, G_A0_ptw, G_A1_ptr, G_A1_ptw, a_loops, b_loops,\
                        G_B0_ready, G_B1_ready, G_B0_ptr, G_B0_ptw, G_B1_ptr, G_B1_ptw, G_B_in_use, B_SA_load,\
                        A_interval_work, A_interval_spare, G_C0_ready, G_C0_ptw, G_C0_ptr, G_C1_ready, G_C1_ptw,\
                        G_C1_ptr, G_Cw_in_use, G_Cr_in_use, cw_loops, cr_loops,\
                        on_chip_alter, bw_in_use, communication_Y_cycle, G_B_ptr_add, communication_on_chip,\
                        communication_interval, communication_internal = tlast.buf_process(G_A_in_use,
                                                        G_A0_ready,
                                                        G_A1_ready,
                                                        G_A0_ptr,
                                                        G_A0_ptw,
                                                        G_A1_ptr,
                                                        G_A1_ptw,
                                                        a_loops,
                                                        b_loops,
                                                        G_B0_ready,
                                                        G_B1_ready,
                                                        G_B0_ptr,
                                                        G_B0_ptw,
                                                        G_B1_ptr,
                                                        G_B1_ptw,
                                                        G_B_in_use,
                                                        B_SA_load,
                                                        A_interval_work,
                                                        A_interval_spare,
                                                        G_C0_ready,
                                                        G_C0_ptw,
                                                        G_C0_ptr,
                                                        G_C1_ready,
                                                        G_C1_ptw,
                                                        G_C1_ptr,
                                                        G_Cw_in_use,
                                                        G_Cr_in_use,
                                                        cw_loops,
                                                        cr_loops,
                                                        Array_i,
                                                        Array_j,
                                                        DR_list,
                                                        SA_list,
                                                        m_num,
                                                        n_num,
                                                        k_num,
                                                        m_num_before,
                                                        n_num_before,
                                                        k_num_before,
                                                        n_num_next,
                                                        k_num_next,
                                                        bandwidth,
                                                        compute_pipeline,
                                                        on_chip_alter,
                                                        A_buf_tile_trans,
                                                        B_buf_tile_trans,
                                                        C_buf_tile_trans,
                                                        Y_buf_tile_trans,
                                                        bw_in_use,
                                                        communication_Y_cycle,
                                                        C_load_out,
                                                        G_B_ptr_add,
                                                        communication_on_chip,
                                                        communication_interval,
                                                        communication_internal
                                                        )
                        cycle += 1
                        if tracking == 1:
                            writer.writerow({'cycle':str(cycle),'G_B0_ptw':str(G_B0_ptw),'G_B1_ptw':str(G_B1_ptw),'G_B0_ptr':str(G_B0_ptr),'G_B1_ptr':str(G_B1_ptr),\
                            'G_A0_ptw':str(G_A0_ptw),'G_A1_ptw':str(G_A1_ptw),'G_A0_ptr':str(G_A0_ptr),'G_A1_ptr':str(G_A1_ptr),\
                            'G_C0_ptr':str(G_C0_ptr),'G_C1_ptr':str(G_C1_ptr),'G_C0_ptw':str(G_C0_ptw),'G_C1_ptw':str(G_C1_ptw)})
                elif m_serial==0 and k_serial==0:
                    A_load, B_load, C_load_in, C_load_out = 1, 1, 0, 0
                    A_load0, B_load0, C_load_in0, C_load_out0 = 1, 1, 0, 0
                    bw_in_use = 'B'
                    if G_B_in_use == 0:
                        G_B1_ready = 0
                    else:
                        G_B0_ready = 0
                    while(on_chip_alter == 0):
                        G_A_in_use, G_A0_ready, G_A1_ready, G_A0_ptr, G_A0_ptw, G_A1_ptr, G_A1_ptw, a_loops, b_loops,\
                        G_B0_ready, G_B1_ready, G_B0_ptr, G_B0_ptw, G_B1_ptr, G_B1_ptw, G_B_in_use, B_SA_load,\
                        A_interval_work, A_interval_spare, G_C0_ready, G_C0_ptw, G_C0_ptr, G_C1_ready, G_C1_ptw,\
                        G_C1_ptr, G_Cw_in_use, G_Cr_in_use, cw_loops, cr_loops,\
                        on_chip_alter, bw_in_use, communication_A_cycle, communication_B_cycle, communication_C_cycle,\
                        communication_Y_cycle, B_load, G_B_ptr_add, communication_on_chip,\
                        communication_interval, communication_internal = t.buf_process(G_A_in_use,
                                                        G_A0_ready,
                                                        G_A1_ready,
                                                        G_A0_ptr,
                                                        G_A0_ptw,
                                                        G_A1_ptr,
                                                        G_A1_ptw,
                                                        a_loops,
                                                        b_loops,
                                                        G_B0_ready,
                                                        G_B1_ready,
                                                        G_B0_ptr,
                                                        G_B0_ptw,
                                                        G_B1_ptr,
                                                        G_B1_ptw,
                                                        G_B_in_use,
                                                        B_SA_load,
                                                        A_interval_work,
                                                        A_interval_spare,
                                                        G_C0_ready,
                                                        G_C0_ptw,
                                                        G_C0_ptr,
                                                        G_C1_ready,
                                                        G_C1_ptw,
                                                        G_C1_ptr,
                                                        G_Cw_in_use,
                                                        G_Cr_in_use,
                                                        cw_loops,
                                                        cr_loops,
                                                        Array_i,
                                                        Array_j,
                                                        DR_list,
                                                        SA_list,
                                                        m_num,
                                                        n_num,
                                                        k_num,
                                                        m_num_before,
                                                        n_num_before,
                                                        k_num_before,
                                                        n_num_next,
                                                        k_num_next,
                                                        bandwidth,
                                                        compute_pipeline,
                                                        on_chip_alter,
                                                        A_buf_tile_trans,
                                                        B_buf_tile_trans,
                                                        C_buf_tile_trans,
                                                        Y_buf_tile_trans,
                                                        bw_in_use,
                                                        communication_A_cycle,
                                                        communication_B_cycle,
                                                        communication_C_cycle,
                                                        communication_Y_cycle,
                                                        A_load,
                                                        B_load,
                                                        C_load_in,
                                                        C_load_out,
                                                        G_B_ptr_add,
                                                        communication_on_chip,
                                                        communication_interval,
                                                        communication_internal
                                                        )
                        cycle += 1
                        if tracking == 1:
                            writer.writerow({'cycle':str(cycle),'G_B0_ptw':str(G_B0_ptw),'G_B1_ptw':str(G_B1_ptw),'G_B0_ptr':str(G_B0_ptr),'G_B1_ptr':str(G_B1_ptr),\
                            'G_A0_ptw':str(G_A0_ptw),'G_A1_ptw':str(G_A1_ptw),'G_A0_ptr':str(G_A0_ptr),'G_A1_ptr':str(G_A1_ptr),\
                            'G_C0_ptr':str(G_C0_ptr),'G_C1_ptr':str(G_C1_ptr),'G_C0_ptw':str(G_C0_ptw),'G_C1_ptw':str(G_C1_ptw)})
                elif m_serial!=0 and k_serial==0:
                    A_load, B_load, C_load_in, C_load_out = 1, 0, 0, 0
                    A_load0, B_load0, C_load_in0, C_load_out0 = 1, 0, 0, 0
                    bw_in_use = 'B'
                    if G_B_in_use == 0:
                        G_B1_ready = 0
                    else:
                        G_B0_ready = 0
                    while(on_chip_alter == 0):
                        G_A_in_use, G_A0_ready, G_A1_ready, G_A0_ptr, G_A0_ptw, G_A1_ptr, G_A1_ptw, a_loops, b_loops,\
                        G_B0_ready, G_B1_ready, G_B0_ptr, G_B0_ptw, G_B1_ptr, G_B1_ptw, G_B_in_use, B_SA_load,\
                        A_interval_work, A_interval_spare, G_C0_ready, G_C0_ptw, G_C0_ptr, G_C1_ready, G_C1_ptw,\
                        G_C1_ptr, G_Cw_in_use, G_Cr_in_use, cw_loops, cr_loops,\
                        on_chip_alter, bw_in_use, communication_A_cycle, communication_B_cycle, communication_C_cycle,\
                        communication_Y_cycle, B_load, G_B_ptr_add, communication_on_chip,\
                        communication_interval, communication_internal = t.buf_process(G_A_in_use,
                                                        G_A0_ready,
                                                        G_A1_ready,
                                                        G_A0_ptr,
                                                        G_A0_ptw,
                                                        G_A1_ptr,
                                                        G_A1_ptw,
                                                        a_loops,
                                                        b_loops,
                                                        G_B0_ready,
                                                        G_B1_ready,
                                                        G_B0_ptr,
                                                        G_B0_ptw,
                                                        G_B1_ptr,
                                                        G_B1_ptw,
                                                        G_B_in_use,
                                                        B_SA_load,
                                                        A_interval_work,
                                                        A_interval_spare,
                                                        G_C0_ready,
                                                        G_C0_ptw,
                                                        G_C0_ptr,
                                                        G_C1_ready,
                                                        G_C1_ptw,
                                                        G_C1_ptr,
                                                        G_Cw_in_use,
                                                        G_Cr_in_use,
                                                        cw_loops,
                                                        cr_loops,
                                                        Array_i,
                                                        Array_j,
                                                        DR_list,
                                                        SA_list,
                                                        m_num,
                                                        n_num,
                                                        k_num,
                                                        m_num_before,
                                                        n_num_before,
                                                        k_num_before,
                                                        n_num_next,
                                                        k_num_next,
                                                        bandwidth,
                                                        compute_pipeline,
                                                        on_chip_alter,
                                                        A_buf_tile_trans,
                                                        B_buf_tile_trans,
                                                        C_buf_tile_trans,
                                                        Y_buf_tile_trans,
                                                        bw_in_use,
                                                        communication_A_cycle,
                                                        communication_B_cycle,
                                                        communication_C_cycle,
                                                        communication_Y_cycle,
                                                        A_load,
                                                        B_load,
                                                        C_load_in,
                                                        C_load_out,
                                                        G_B_ptr_add,
                                                        communication_on_chip,
                                                        communication_interval,
                                                        communication_internal
                                                        )
                        cycle += 1
                        if tracking == 1:
                            writer.writerow({'cycle':str(cycle),'G_B0_ptw':str(G_B0_ptw),'G_B1_ptw':str(G_B1_ptw),'G_B0_ptr':str(G_B0_ptr),'G_B1_ptr':str(G_B1_ptr),\
                            'G_A0_ptw':str(G_A0_ptw),'G_A1_ptw':str(G_A1_ptw),'G_A0_ptr':str(G_A0_ptr),'G_A1_ptr':str(G_A1_ptr),\
                            'G_C0_ptr':str(G_C0_ptr),'G_C1_ptr':str(G_C1_ptr),'G_C0_ptw':str(G_C0_ptw),'G_C1_ptw':str(G_C1_ptw)})
                elif m_serial!=m_loop_times-1 and k_serial==k_loop_times-1 and not(m_serial==0 and n_serial==0):
                    A_load, B_load, C_load_in, C_load_out = 1, 0, 1, 1
                    A_load0, B_load0, C_load_in0, C_load_out0 = 1, 0, 1, 1
                    bw_in_use = 'B'
                    if G_B_in_use == 0:
                        G_B1_ready = 0
                    else:
                        G_B0_ready = 0
                    while(on_chip_alter == 0):
                        G_A_in_use, G_A0_ready, G_A1_ready, G_A0_ptr, G_A0_ptw, G_A1_ptr, G_A1_ptw, a_loops, b_loops,\
                        G_B0_ready, G_B1_ready, G_B0_ptr, G_B0_ptw, G_B1_ptr, G_B1_ptw, G_B_in_use, B_SA_load,\
                        A_interval_work, A_interval_spare, G_C0_ready, G_C0_ptw, G_C0_ptr, G_C1_ready, G_C1_ptw,\
                        G_C1_ptr, G_Cw_in_use, G_Cr_in_use, cw_loops, cr_loops,\
                        on_chip_alter, bw_in_use, communication_A_cycle, communication_B_cycle, communication_C_cycle,\
                        communication_Y_cycle, B_load, G_B_ptr_add, communication_on_chip,\
                        communication_interval, communication_internal = t.buf_process(G_A_in_use,
                                                        G_A0_ready,
                                                        G_A1_ready,
                                                        G_A0_ptr,
                                                        G_A0_ptw,
                                                        G_A1_ptr,
                                                        G_A1_ptw,
                                                        a_loops,
                                                        b_loops,
                                                        G_B0_ready,
                                                        G_B1_ready,
                                                        G_B0_ptr,
                                                        G_B0_ptw,
                                                        G_B1_ptr,
                                                        G_B1_ptw,
                                                        G_B_in_use,
                                                        B_SA_load,
                                                        A_interval_work,
                                                        A_interval_spare,
                                                        G_C0_ready,
                                                        G_C0_ptw,
                                                        G_C0_ptr,
                                                        G_C1_ready,
                                                        G_C1_ptw,
                                                        G_C1_ptr,
                                                        G_Cw_in_use,
                                                        G_Cr_in_use,
                                                        cw_loops,
                                                        cr_loops,
                                                        Array_i,
                                                        Array_j,
                                                        DR_list,
                                                        SA_list,
                                                        m_num,
                                                        n_num,
                                                        k_num,
                                                        m_num_before,
                                                        n_num_before,
                                                        k_num_before,
                                                        n_num_next,
                                                        k_num_next,
                                                        bandwidth,
                                                        compute_pipeline,
                                                        on_chip_alter,
                                                        A_buf_tile_trans,
                                                        B_buf_tile_trans,
                                                        C_buf_tile_trans,
                                                        Y_buf_tile_trans,
                                                        bw_in_use,
                                                        communication_A_cycle,
                                                        communication_B_cycle,
                                                        communication_C_cycle,
                                                        communication_Y_cycle,
                                                        A_load,
                                                        B_load,
                                                        C_load_in,
                                                        C_load_out,
                                                        G_B_ptr_add,
                                                        communication_on_chip,
                                                        communication_interval,
                                                        communication_internal
                                                        )
                        cycle += 1
                        if tracking == 1:
                            writer.writerow({'cycle':str(cycle),'G_B0_ptw':str(G_B0_ptw),'G_B1_ptw':str(G_B1_ptw),'G_B0_ptr':str(G_B0_ptr),'G_B1_ptr':str(G_B1_ptr),\
                            'G_A0_ptw':str(G_A0_ptw),'G_A1_ptw':str(G_A1_ptw),'G_A0_ptr':str(G_A0_ptr),'G_A1_ptr':str(G_A1_ptr),\
                            'G_C0_ptr':str(G_C0_ptr),'G_C1_ptr':str(G_C1_ptr),'G_C0_ptw':str(G_C0_ptw),'G_C1_ptw':str(G_C1_ptw)})
                else:
                    A_load, B_load, C_load_in, C_load_out = 1, 1, 1, 1
                    A_load0, B_load0, C_load_in0, C_load_out0 = 1, 1, 1, 1
                    bw_in_use = 'B'
                    if G_B_in_use == 0:
                        G_B1_ready = 0
                    else:
                        G_B0_ready = 0
                    while(on_chip_alter == 0):
                        G_A_in_use, G_A0_ready, G_A1_ready, G_A0_ptr, G_A0_ptw, G_A1_ptr, G_A1_ptw, a_loops, b_loops,\
                        G_B0_ready, G_B1_ready, G_B0_ptr, G_B0_ptw, G_B1_ptr, G_B1_ptw, G_B_in_use, B_SA_load,\
                        A_interval_work, A_interval_spare, G_C0_ready, G_C0_ptw, G_C0_ptr, G_C1_ready, G_C1_ptw,\
                        G_C1_ptr, G_Cw_in_use, G_Cr_in_use, cw_loops, cr_loops,\
                        on_chip_alter, bw_in_use, communication_A_cycle, communication_B_cycle, communication_C_cycle,\
                        communication_Y_cycle, B_load, G_B_ptr_add, communication_on_chip,\
                        communication_interval, communication_internal = t.buf_process(G_A_in_use,
                                                        G_A0_ready,
                                                        G_A1_ready,
                                                        G_A0_ptr,
                                                        G_A0_ptw,
                                                        G_A1_ptr,
                                                        G_A1_ptw,
                                                        a_loops,
                                                        b_loops,
                                                        G_B0_ready,
                                                        G_B1_ready,
                                                        G_B0_ptr,
                                                        G_B0_ptw,
                                                        G_B1_ptr,
                                                        G_B1_ptw,
                                                        G_B_in_use,
                                                        B_SA_load,
                                                        A_interval_work,
                                                        A_interval_spare,
                                                        G_C0_ready,
                                                        G_C0_ptw,
                                                        G_C0_ptr,
                                                        G_C1_ready,
                                                        G_C1_ptw,
                                                        G_C1_ptr,
                                                        G_Cw_in_use,
                                                        G_Cr_in_use,
                                                        cw_loops,
                                                        cr_loops,
                                                        Array_i,
                                                        Array_j,
                                                        DR_list,
                                                        SA_list,
                                                        m_num,
                                                        n_num,
                                                        k_num,
                                                        m_num_before,
                                                        n_num_before,
                                                        k_num_before,
                                                        n_num_next,
                                                        k_num_next,
                                                        bandwidth,
                                                        compute_pipeline,
                                                        on_chip_alter,
                                                        A_buf_tile_trans,
                                                        B_buf_tile_trans,
                                                        C_buf_tile_trans,
                                                        Y_buf_tile_trans,
                                                        bw_in_use,
                                                        communication_A_cycle,
                                                        communication_B_cycle,
                                                        communication_C_cycle,
                                                        communication_Y_cycle,
                                                        A_load,
                                                        B_load,
                                                        C_load_in,
                                                        C_load_out,
                                                        G_B_ptr_add,
                                                        communication_on_chip,
                                                        communication_interval,
                                                        communication_internal
                                                        )
                        cycle += 1
                        if tracking == 1:
                            writer.writerow({'cycle':str(cycle),'G_B0_ptw':str(G_B0_ptw),'G_B1_ptw':str(G_B1_ptw),'G_B0_ptr':str(G_B0_ptr),'G_B1_ptr':str(G_B1_ptr),\
                            'G_A0_ptw':str(G_A0_ptw),'G_A1_ptw':str(G_A1_ptw),'G_A0_ptr':str(G_A0_ptr),'G_A1_ptr':str(G_A1_ptr),\
                            'G_C0_ptr':str(G_C0_ptr),'G_C1_ptr':str(G_C1_ptr),'G_C0_ptw':str(G_C0_ptw),'G_C1_ptw':str(G_C1_ptw)})

                communication_off_chip += A_load0 * f.ceil(m_num_next * k_num_next, bandwidth)\
                + B_load0 * f.ceil(k_num_next * n_num_next, bandwidth) + C_load_in0 * partial_sum[m_serial_next][n_serial_next] * f.ceil(m_num_next * n_num_next, bandwidth)\
                + C_load_out0 * f.ceil(m_num_before * n_num_before, bandwidth)
                # 记录片外的部分和状态，“读入”到片上一次后，下次读取处于存在部分和的状态
                if C_load_in == 1:
                    partial_sum[m_serial_next][n_serial_next] = 1
                print("\r", end="")
                print("Simulation progress: {}%: ".format(int(counter/counter_total * 100)), end="")
                sys.stdout.flush()
if m_loop_times >= 3 and k_loop_times >= 3 and n_loop_times >= 1:
    for n_serial in range(0,n_loop_times):
        for m_serial in range(0,m_loop_times):
            for k_serial in range(0,k_loop_times):
                counter += 1
                # 计算本次片上的mkn规模大小
                m_num, k_num, n_num = f.mkn_num(m_serial, k_serial, n_serial, m_loop_times, k_loop_times, n_loop_times, M, K, N, m, k, n)
                m_serial_next, k_serial_next, n_serial_next = f.next_num_nmk(m_serial, k_serial, n_serial, m_loop_times, k_loop_times, n_loop_times)
                m_serial_before, k_serial_before, n_serial_before = f.before_num_nmk(m_serial, k_serial, n_serial, m_loop_times, k_loop_times, n_loop_times)
                m_num_next, k_num_next, n_num_next = f.mkn_num(m_serial_next, k_serial_next, n_serial_next, m_loop_times, k_loop_times, n_loop_times, M, K, N, m, k, n)
                m_num_before, k_num_before, n_num_before = f.mkn_num(m_serial_before, k_serial_before, n_serial_before, m_loop_times, k_loop_times, n_loop_times, M, K, N, m, k, n)

                m_coordinate, k_coordinate, n_coordinate = m_serial_next * m, k_serial_next * k, n_serial_next * n

                A_buf_tile_trans = f.ceil(m_num_next * k_num_next, bandwidth)
                B_buf_tile_trans = f.ceil(k_num_next * n_num_next, bandwidth)
                # 判断C是否要从片外读入，第一次是不需要读入的
                if partial_sum[m_serial_next][n_serial_next] == 0:
                    C_buf_tile_trans = 0
                else:
                    C_buf_tile_trans = f.ceil(m_num_next * n_num_next, bandwidth)
                Y_buf_tile_trans = f.ceil(m_num_before * n_num_before, bandwidth)
                on_chip_alter = 0
                # 第一个
                if m_serial==0 and k_serial==k_loop_times-1 and n_serial==0:
                    A_load, B_load, C_load_in, C_load_out = 1, 1, 1, 0
                    A_load0, B_load0, C_load_in0, C_load_out0 = 1, 1, 1, 0
                    bw_in_use = 'B'
                    if G_B_in_use == 0:
                        G_B1_ready = 0
                    else:
                        G_B0_ready = 0
                    while(on_chip_alter == 0):
                        G_A_in_use, G_A0_ready, G_A1_ready, G_A0_ptr, G_A0_ptw, G_A1_ptr, G_A1_ptw, a_loops, b_loops,\
                        G_B0_ready, G_B1_ready, G_B0_ptr, G_B0_ptw, G_B1_ptr, G_B1_ptw, G_B_in_use, B_SA_load,\
                        A_interval_work, A_interval_spare, G_C0_ready, G_C0_ptw, G_C0_ptr, G_C1_ready, G_C1_ptw,\
                        G_C1_ptr, G_Cw_in_use, G_Cr_in_use, cw_loops, cr_loops,\
                        on_chip_alter, bw_in_use, communication_A_cycle, communication_B_cycle, communication_C_cycle,\
                        communication_Y_cycle, B_load, G_B_ptr_add, communication_on_chip,\
                        communication_interval, communication_internal = t.buf_process(G_A_in_use,
                                                        G_A0_ready,
                                                        G_A1_ready,
                                                        G_A0_ptr,
                                                        G_A0_ptw,
                                                        G_A1_ptr,
                                                        G_A1_ptw,
                                                        a_loops,
                                                        b_loops,
                                                        G_B0_ready,
                                                        G_B1_ready,
                                                        G_B0_ptr,
                                                        G_B0_ptw,
                                                        G_B1_ptr,
                                                        G_B1_ptw,
                                                        G_B_in_use,
                                                        B_SA_load,
                                                        A_interval_work,
                                                        A_interval_spare,
                                                        G_C0_ready,
                                                        G_C0_ptw,
                                                        G_C0_ptr,
                                                        G_C1_ready,
                                                        G_C1_ptw,
                                                        G_C1_ptr,
                                                        G_Cw_in_use,
                                                        G_Cr_in_use,
                                                        cw_loops,
                                                        cr_loops,
                                                        Array_i,
                                                        Array_j,
                                                        DR_list,
                                                        SA_list,
                                                        m_num,
                                                        n_num,
                                                        k_num,
                                                        m_num_before,
                                                        n_num_before,
                                                        k_num_before,
                                                        n_num_next,
                                                        k_num_next,
                                                        bandwidth,
                                                        compute_pipeline,
                                                        on_chip_alter,
                                                        A_buf_tile_trans,
                                                        B_buf_tile_trans,
                                                        C_buf_tile_trans,
                                                        Y_buf_tile_trans,
                                                        bw_in_use,
                                                        communication_A_cycle,
                                                        communication_B_cycle,
                                                        communication_C_cycle,
                                                        communication_Y_cycle,
                                                        A_load,
                                                        B_load,
                                                        C_load_in,
                                                        C_load_out,
                                                        G_B_ptr_add,
                                                        communication_on_chip,
                                                        communication_interval,
                                                        communication_internal
                                                        )
                        cycle += 1
                        if tracking == 1:
                            writer.writerow({'cycle':str(cycle),'G_B0_ptw':str(G_B0_ptw),'G_B1_ptw':str(G_B1_ptw),'G_B0_ptr':str(G_B0_ptr),'G_B1_ptr':str(G_B1_ptr),\
                            'G_A0_ptw':str(G_A0_ptw),'G_A1_ptw':str(G_A1_ptw),'G_A0_ptr':str(G_A0_ptr),'G_A1_ptr':str(G_A1_ptr),\
                            'G_C0_ptr':str(G_C0_ptr),'G_C1_ptr':str(G_C1_ptr),'G_C0_ptw':str(G_C0_ptw),'G_C1_ptw':str(G_C1_ptw)})
                # 最后一个
                elif m_serial==m_loop_times-1 and k_serial==k_loop_times-1 and n_serial==n_loop_times-1:
                    A_load, B_load, C_load_in, C_load_out = 0, 0, 0, 1
                    A_load0, B_load0, C_load_in0, C_load_out0 = 0, 0, 0, 1
                    while(on_chip_alter == 0):
                        G_A_in_use, G_A0_ready, G_A1_ready, G_A0_ptr, G_A0_ptw, G_A1_ptr, G_A1_ptw, a_loops, b_loops,\
                        G_B0_ready, G_B1_ready, G_B0_ptr, G_B0_ptw, G_B1_ptr, G_B1_ptw, G_B_in_use, B_SA_load,\
                        A_interval_work, A_interval_spare, G_C0_ready, G_C0_ptw, G_C0_ptr, G_C1_ready, G_C1_ptw,\
                        G_C1_ptr, G_Cw_in_use, G_Cr_in_use, cw_loops, cr_loops,\
                        on_chip_alter, bw_in_use, communication_Y_cycle, G_B_ptr_add, communication_on_chip,\
                        communication_interval, communication_internal = tlast.buf_process(G_A_in_use,
                                                        G_A0_ready,
                                                        G_A1_ready,
                                                        G_A0_ptr,
                                                        G_A0_ptw,
                                                        G_A1_ptr,
                                                        G_A1_ptw,
                                                        a_loops,
                                                        b_loops,
                                                        G_B0_ready,
                                                        G_B1_ready,
                                                        G_B0_ptr,
                                                        G_B0_ptw,
                                                        G_B1_ptr,
                                                        G_B1_ptw,
                                                        G_B_in_use,
                                                        B_SA_load,
                                                        A_interval_work,
                                                        A_interval_spare,
                                                        G_C0_ready,
                                                        G_C0_ptw,
                                                        G_C0_ptr,
                                                        G_C1_ready,
                                                        G_C1_ptw,
                                                        G_C1_ptr,
                                                        G_Cw_in_use,
                                                        G_Cr_in_use,
                                                        cw_loops,
                                                        cr_loops,
                                                        Array_i,
                                                        Array_j,
                                                        DR_list,
                                                        SA_list,
                                                        m_num,
                                                        n_num,
                                                        k_num,
                                                        m_num_before,
                                                        n_num_before,
                                                        k_num_before,
                                                        n_num_next,
                                                        k_num_next,
                                                        bandwidth,
                                                        compute_pipeline,
                                                        on_chip_alter,
                                                        A_buf_tile_trans,
                                                        B_buf_tile_trans,
                                                        C_buf_tile_trans,
                                                        Y_buf_tile_trans,
                                                        bw_in_use,
                                                        communication_Y_cycle,
                                                        C_load_out,
                                                        G_B_ptr_add,
                                                        communication_on_chip,
                                                        communication_interval,
                                                        communication_internal
                                                        )
                        cycle += 1
                        if tracking == 1:
                            writer.writerow({'cycle':str(cycle),'G_B0_ptw':str(G_B0_ptw),'G_B1_ptw':str(G_B1_ptw),'G_B0_ptr':str(G_B0_ptr),'G_B1_ptr':str(G_B1_ptr),\
                            'G_A0_ptw':str(G_A0_ptw),'G_A1_ptw':str(G_A1_ptw),'G_A0_ptr':str(G_A0_ptr),'G_A1_ptr':str(G_A1_ptr),\
                            'G_C0_ptr':str(G_C0_ptr),'G_C1_ptr':str(G_C1_ptr),'G_C0_ptw':str(G_C0_ptw),'G_C1_ptw':str(G_C1_ptw)})
                elif k_serial!=k_loop_times-1:
                    A_load, B_load, C_load_in, C_load_out = 1, 1, 0, 0
                    A_load0, B_load0, C_load_in0, C_load_out0 = 1, 1, 0, 0
                    bw_in_use = 'B'
                    if G_B_in_use == 0:
                        G_B1_ready = 0
                    else:
                        G_B0_ready = 0
                    while(on_chip_alter == 0):
                        G_A_in_use, G_A0_ready, G_A1_ready, G_A0_ptr, G_A0_ptw, G_A1_ptr, G_A1_ptw, a_loops, b_loops,\
                        G_B0_ready, G_B1_ready, G_B0_ptr, G_B0_ptw, G_B1_ptr, G_B1_ptw, G_B_in_use, B_SA_load,\
                        A_interval_work, A_interval_spare, G_C0_ready, G_C0_ptw, G_C0_ptr, G_C1_ready, G_C1_ptw,\
                        G_C1_ptr, G_Cw_in_use, G_Cr_in_use, cw_loops, cr_loops,\
                        on_chip_alter, bw_in_use, communication_A_cycle, communication_B_cycle, communication_C_cycle,\
                        communication_Y_cycle, B_load, G_B_ptr_add, communication_on_chip,\
                        communication_interval, communication_internal = t.buf_process(G_A_in_use,
                                                        G_A0_ready,
                                                        G_A1_ready,
                                                        G_A0_ptr,
                                                        G_A0_ptw,
                                                        G_A1_ptr,
                                                        G_A1_ptw,
                                                        a_loops,
                                                        b_loops,
                                                        G_B0_ready,
                                                        G_B1_ready,
                                                        G_B0_ptr,
                                                        G_B0_ptw,
                                                        G_B1_ptr,
                                                        G_B1_ptw,
                                                        G_B_in_use,
                                                        B_SA_load,
                                                        A_interval_work,
                                                        A_interval_spare,
                                                        G_C0_ready,
                                                        G_C0_ptw,
                                                        G_C0_ptr,
                                                        G_C1_ready,
                                                        G_C1_ptw,
                                                        G_C1_ptr,
                                                        G_Cw_in_use,
                                                        G_Cr_in_use,
                                                        cw_loops,
                                                        cr_loops,
                                                        Array_i,
                                                        Array_j,
                                                        DR_list,
                                                        SA_list,
                                                        m_num,
                                                        n_num,
                                                        k_num,
                                                        m_num_before,
                                                        n_num_before,
                                                        k_num_before,
                                                        n_num_next,
                                                        k_num_next,
                                                        bandwidth,
                                                        compute_pipeline,
                                                        on_chip_alter,
                                                        A_buf_tile_trans,
                                                        B_buf_tile_trans,
                                                        C_buf_tile_trans,
                                                        Y_buf_tile_trans,
                                                        bw_in_use,
                                                        communication_A_cycle,
                                                        communication_B_cycle,
                                                        communication_C_cycle,
                                                        communication_Y_cycle,
                                                        A_load,
                                                        B_load,
                                                        C_load_in,
                                                        C_load_out,
                                                        G_B_ptr_add,
                                                        communication_on_chip,
                                                        communication_interval,
                                                        communication_internal
                                                        )
                        cycle += 1
                        if tracking == 1:
                            writer.writerow({'cycle':str(cycle),'G_B0_ptw':str(G_B0_ptw),'G_B1_ptw':str(G_B1_ptw),'G_B0_ptr':str(G_B0_ptr),'G_B1_ptr':str(G_B1_ptr),\
                            'G_A0_ptw':str(G_A0_ptw),'G_A1_ptw':str(G_A1_ptw),'G_A0_ptr':str(G_A0_ptr),'G_A1_ptr':str(G_A1_ptr),\
                            'G_C0_ptr':str(G_C0_ptr),'G_C1_ptr':str(G_C1_ptr),'G_C0_ptw':str(G_C0_ptw),'G_C1_ptw':str(G_C1_ptw)})
                elif m_serial!=0 and k_serial==0:
                    A_load, B_load, C_load_in, C_load_out = 1, 1, 1, 1
                    A_load0, B_load0, C_load_in0, C_load_out0 = 1, 1, 1, 1
                    bw_in_use = 'B'
                    if G_B_in_use == 0:
                        G_B1_ready = 0
                    else:
                        G_B0_ready = 0
                    while(on_chip_alter == 0):
                        G_A_in_use, G_A0_ready, G_A1_ready, G_A0_ptr, G_A0_ptw, G_A1_ptr, G_A1_ptw, a_loops, b_loops,\
                        G_B0_ready, G_B1_ready, G_B0_ptr, G_B0_ptw, G_B1_ptr, G_B1_ptw, G_B_in_use, B_SA_load,\
                        A_interval_work, A_interval_spare, G_C0_ready, G_C0_ptw, G_C0_ptr, G_C1_ready, G_C1_ptw,\
                        G_C1_ptr, G_Cw_in_use, G_Cr_in_use, cw_loops, cr_loops,\
                        on_chip_alter, bw_in_use, communication_A_cycle, communication_B_cycle, communication_C_cycle,\
                        communication_Y_cycle, B_load, G_B_ptr_add, communication_on_chip,\
                        communication_interval, communication_internal = t.buf_process(G_A_in_use,
                                                        G_A0_ready,
                                                        G_A1_ready,
                                                        G_A0_ptr,
                                                        G_A0_ptw,
                                                        G_A1_ptr,
                                                        G_A1_ptw,
                                                        a_loops,
                                                        b_loops,
                                                        G_B0_ready,
                                                        G_B1_ready,
                                                        G_B0_ptr,
                                                        G_B0_ptw,
                                                        G_B1_ptr,
                                                        G_B1_ptw,
                                                        G_B_in_use,
                                                        B_SA_load,
                                                        A_interval_work,
                                                        A_interval_spare,
                                                        G_C0_ready,
                                                        G_C0_ptw,
                                                        G_C0_ptr,
                                                        G_C1_ready,
                                                        G_C1_ptw,
                                                        G_C1_ptr,
                                                        G_Cw_in_use,
                                                        G_Cr_in_use,
                                                        cw_loops,
                                                        cr_loops,
                                                        Array_i,
                                                        Array_j,
                                                        DR_list,
                                                        SA_list,
                                                        m_num,
                                                        n_num,
                                                        k_num,
                                                        m_num_before,
                                                        n_num_before,
                                                        k_num_before,
                                                        n_num_next,
                                                        k_num_next,
                                                        bandwidth,
                                                        compute_pipeline,
                                                        on_chip_alter,
                                                        A_buf_tile_trans,
                                                        B_buf_tile_trans,
                                                        C_buf_tile_trans,
                                                        Y_buf_tile_trans,
                                                        bw_in_use,
                                                        communication_A_cycle,
                                                        communication_B_cycle,
                                                        communication_C_cycle,
                                                        communication_Y_cycle,
                                                        A_load,
                                                        B_load,
                                                        C_load_in,
                                                        C_load_out,
                                                        G_B_ptr_add,
                                                        communication_on_chip,
                                                        communication_interval,
                                                        communication_internal
                                                        )
                        cycle += 1
                        if tracking == 1:
                            writer.writerow({'cycle':str(cycle),'G_B0_ptw':str(G_B0_ptw),'G_B1_ptw':str(G_B1_ptw),'G_B0_ptr':str(G_B0_ptr),'G_B1_ptr':str(G_B1_ptr),\
                            'G_A0_ptw':str(G_A0_ptw),'G_A1_ptw':str(G_A1_ptw),'G_A0_ptr':str(G_A0_ptr),'G_A1_ptr':str(G_A1_ptr),\
                            'G_C0_ptr':str(G_C0_ptr),'G_C1_ptr':str(G_C1_ptr),'G_C0_ptw':str(G_C0_ptw),'G_C1_ptw':str(G_C1_ptw)})
                else:
                    A_load, B_load, C_load_in, C_load_out = 1, 1, 1, 1
                    A_load0, B_load0, C_load_in0, C_load_out0 = 1, 1, 1, 1
                    bw_in_use = 'B'
                    if G_B_in_use == 0:
                        G_B1_ready = 0
                    else:
                        G_B0_ready = 0
                    while(on_chip_alter == 0):
                        G_A_in_use, G_A0_ready, G_A1_ready, G_A0_ptr, G_A0_ptw, G_A1_ptr, G_A1_ptw, a_loops, b_loops,\
                        G_B0_ready, G_B1_ready, G_B0_ptr, G_B0_ptw, G_B1_ptr, G_B1_ptw, G_B_in_use, B_SA_load,\
                        A_interval_work, A_interval_spare, G_C0_ready, G_C0_ptw, G_C0_ptr, G_C1_ready, G_C1_ptw,\
                        G_C1_ptr, G_Cw_in_use, G_Cr_in_use, cw_loops, cr_loops,\
                        on_chip_alter, bw_in_use, communication_A_cycle, communication_B_cycle, communication_C_cycle,\
                        communication_Y_cycle, B_load, G_B_ptr_add, communication_on_chip,\
                        communication_interval, communication_internal = t.buf_process(G_A_in_use,
                                                        G_A0_ready,
                                                        G_A1_ready,
                                                        G_A0_ptr,
                                                        G_A0_ptw,
                                                        G_A1_ptr,
                                                        G_A1_ptw,
                                                        a_loops,
                                                        b_loops,
                                                        G_B0_ready,
                                                        G_B1_ready,
                                                        G_B0_ptr,
                                                        G_B0_ptw,
                                                        G_B1_ptr,
                                                        G_B1_ptw,
                                                        G_B_in_use,
                                                        B_SA_load,
                                                        A_interval_work,
                                                        A_interval_spare,
                                                        G_C0_ready,
                                                        G_C0_ptw,
                                                        G_C0_ptr,
                                                        G_C1_ready,
                                                        G_C1_ptw,
                                                        G_C1_ptr,
                                                        G_Cw_in_use,
                                                        G_Cr_in_use,
                                                        cw_loops,
                                                        cr_loops,
                                                        Array_i,
                                                        Array_j,
                                                        DR_list,
                                                        SA_list,
                                                        m_num,
                                                        n_num,
                                                        k_num,
                                                        m_num_before,
                                                        n_num_before,
                                                        k_num_before,
                                                        n_num_next,
                                                        k_num_next,
                                                        bandwidth,
                                                        compute_pipeline,
                                                        on_chip_alter,
                                                        A_buf_tile_trans,
                                                        B_buf_tile_trans,
                                                        C_buf_tile_trans,
                                                        Y_buf_tile_trans,
                                                        bw_in_use,
                                                        communication_A_cycle,
                                                        communication_B_cycle,
                                                        communication_C_cycle,
                                                        communication_Y_cycle,
                                                        A_load,
                                                        B_load,
                                                        C_load_in,
                                                        C_load_out,
                                                        G_B_ptr_add,
                                                        communication_on_chip,
                                                        communication_interval,
                                                        communication_internal
                                                        )
                        cycle += 1
                        if tracking == 1:
                            writer.writerow({'cycle':str(cycle),'G_B0_ptw':str(G_B0_ptw),'G_B1_ptw':str(G_B1_ptw),'G_B0_ptr':str(G_B0_ptr),'G_B1_ptr':str(G_B1_ptr),\
                            'G_A0_ptw':str(G_A0_ptw),'G_A1_ptw':str(G_A1_ptw),'G_A0_ptr':str(G_A0_ptr),'G_A1_ptr':str(G_A1_ptr),\
                            'G_C0_ptr':str(G_C0_ptr),'G_C1_ptr':str(G_C1_ptr),'G_C0_ptw':str(G_C0_ptw),'G_C1_ptw':str(G_C1_ptw)})

                communication_off_chip += A_load0 * f.ceil(m_num_next * k_num_next, bandwidth)\
                + B_load0 * f.ceil(k_num_next * n_num_next, bandwidth) + C_load_in0 * partial_sum[m_serial_next][n_serial_next] * f.ceil(m_num_next * n_num_next, bandwidth)\
                + C_load_out0 * f.ceil(m_num_before * n_num_before, bandwidth)
                # 记录片外的部分和状态，“读入”到片上一次后，下次读取处于存在部分和的状态
                if C_load_in == 1:
                    partial_sum[m_serial_next][n_serial_next] = 1
                print("\r", end="")
                print("Simulation progress: {}%: ".format(int(counter/counter_total * 100)), end="")
                sys.stdout.flush()
#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
# 将C_buf中的数据写回HBM
# write_back_flag意义(0表示C_buf还在写入,1表示C_buf写回HBM,2表示写回完成)
m_num_next, k_num_next, n_num_next = f.mkn_num(m_serial_next, k_serial_next, n_serial_next, m_loop_times, k_loop_times, n_loop_times, M, K, N, m, k, n)
Y_buf_tile_trans = f.ceil(m_num_next * n_num_next, bandwidth)
write_back_flag = 0
if G_Cr_in_use == 1:
    if G_Cw_in_use == 1:
        write_back_flag = 1
    else:
        write_back_flag = 0
else:
    if G_Cw_in_use == 0:
        write_back_flag = 1
    else:
        write_back_flag = 0
while(write_back_flag != 2):
    cycle += 1
    write_back_flag, G_Cr_in_use, G_C0_ptr, G_C0_ptw, G_C1_ptr, G_C1_ptw, communication_Y_cycle = \
    t.write_back(write_back_flag,
                    m_num,
                    n_num,
                    G_Cr_in_use,
                    G_Cw_in_use,
                    G_C0_ptr,
                    G_C0_ptw,
                    G_C1_ptr,
                    G_C1_ptw,
                    DR_list,
                    SA_list,
                    Array_i,
                    Array_j,
                    compute_pipeline,
                    bandwidth,
                    Y_buf_tile_trans,
                    communication_Y_cycle)
    if tracking == 1:
        writer.writerow({'cycle':str(cycle),'G_B0_ptw':str(G_B0_ptw),'G_B1_ptw':str(G_B1_ptw),'G_B0_ptr':str(G_B0_ptr),'G_B1_ptr':str(G_B1_ptr),\
        'G_A0_ptw':str(G_A0_ptw),'G_A1_ptw':str(G_A1_ptw),'G_A0_ptr':str(G_A0_ptr),'G_A1_ptr':str(G_A1_ptr),\
        'G_C0_ptr':str(G_C0_ptr),'G_C1_ptr':str(G_C1_ptr),'G_C0_ptw':str(G_C0_ptw),'G_C1_ptw':str(G_C1_ptw)})
communication_off_chip += f.ceil(m_num_next * n_num_next, bandwidth)
if tracking == 1:
    output_file.close()

# 统计结果
communication_cycle = communication_cycle + communication_A_cycle + communication_B_cycle + communication_C_cycle + communication_Y_cycle
PE_utility = M * K * N / (Array_i * Array_j * cycle)
throughput = M * K * N / cycle
energy_cost_communication = communication_off_chip * 200 + communication_on_chip * 6 + communication_interval * 2 + communication_internal
energy_cost_computation = communication_internal // 3
computation_cycle = Array_i * compute_pipeline + N * f.roundup(M, Array_j) * f.roundup(K, Array_i) + Array_j
output_file = open('./data/output/output_%d.csv'%(level_num), 'w')
fieldnames = ['cycle','communication_cycle','communication_A_cycle','communication_B_cycle','communication_C_cycle',\
'communication_Y_cycle','PE_utility','throughput','communication_off_chip','communication_on_chip','communication_interval',\
'communication_internal','energy_cost_communication','energy_cost_computation']
writer = csv.DictWriter(output_file, fieldnames=fieldnames)
writer.writeheader()
writer.writerow({'cycle':str(cycle),'communication_cycle':str(communication_cycle),'communication_A_cycle':str(communication_A_cycle),\
'communication_B_cycle':str(communication_B_cycle),'communication_C_cycle':str(communication_C_cycle),\
'communication_Y_cycle':str(communication_Y_cycle),'PE_utility':str(PE_utility),'throughput':str(throughput),\
'communication_off_chip':str(int(communication_off_chip)),'communication_on_chip':str(int(communication_on_chip)),\
'communication_interval':str(int(communication_interval)),'communication_internal':str(int(communication_internal)),\
'energy_cost_communication':str(int(energy_cost_communication)),'energy_cost_computation':str(int(energy_cost_computation))})
output_file.close()
