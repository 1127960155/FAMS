# encoding: utf-8
from __future__ import division
from __future__ import print_function
# GSM内各个buffer读写的记录
# 包括weight在A_buf的读出，备用A_buf与B_buf的写入
# 在一次性完成的计算中，不需要buf1，仅需buf0即可全部完成
import numpy as np
import functions as f
def buf_process(
                G_A_in_use,
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
                m,
                n,
                k,
                bandwidth,
                compute_pipeline,
                on_chip_alter,
                bw_in_use,
                communication_Y_cycle,
                G_B_ptr_add,
                communication_on_chip,
                communication_interval,
                communication_internal
                ):
    new_compute = 0
    k_loops = f.roundup(k, Array_i)
    n_loops = f.roundup(n, Array_j)
    A_buf_tile, B_buf_tile = m * f.expansion(k, Array_i), f.expansion(k, Array_i) * f.expansion(n, Array_j)
    C_buf_tile = m * f.expansion(n, Array_j)

    # 当C_buf0为工作buf时的读SA流出的数据读取数据到工作C_buf的过程*************************
    if G_Cw_in_use == 0:
        if DR_list[Array_j-2] == 1:
            G_C0_ptw += Array_j
        # 在G_C0_ptw写满之后，判断此分块是否已完全循环结束，是则更换，否则接着循环
        if G_C0_ptw == C_buf_tile:
            if cw_loops != k_loops:
                cw_loops += 1
                G_C0_ptw = 0
            else:
                cw_loops = 1
                G_Cw_in_use = 1
                G_C0_ptw = 0
                G_C0_ready = 0


    # 模拟部分和DR模块的FIFO功能,对齐数据********************************************************
    for i in range(1,Array_j-1):
        DR_list[Array_j-1-i] = DR_list[Array_j-2-i]
    DR_list[0] = SA_list[-1]

    # 当A_buf_0为工作buf时的过程****************************************************************
    if G_A_in_use == 0:

        # 将A_buf0的数据读出
        # 在每一次A_buf0的第一组数据读出之前,检查B中小分块在SA中是否到位(B_SA_load=1则到位,0则仍需等待),\
        # 因为有可能HBM与B_buf的通信导致延迟
        if G_A0_ptr == 0 and a_loops == 1:
            if B_SA_load >= 1:
                B_SA_load -= 1
                G_A0_ptr += Array_i
                # a读取最后一列
                if a_loops - n_loops * (k_loops - 1) > 0:
                    # Array_i个数据从A_buf读出
                    communication_on_chip += k + Array_i - f.expansion(k, Array_i)
                    # Array_i个数据从A_buf到DR FIFO数据重组经历的额外读写次数
                    communication_interval += k + Array_i - f.expansion(k, Array_i) - 1
                    # Array_i个数据在PE之间要传递Array_j-1次
                    communication_interval += (k + Array_i - f.expansion(k, Array_i)) * (Array_j - 1)
                else:
                    # Array_i个数据从A_buf读出
                    communication_on_chip += Array_i
                    # Array_i个数据从A_buf到DR FIFO数据重组经历的额外读写次数
                    communication_interval += Array_i - 1
                    # Array_i个数据在PE之间要传递Array_j-1次
                    communication_interval += Array_i * (Array_j - 1)
                new_compute = 1
                if a_loops - n_loops * (k_loops - 1) > 0 and a_loops != n_loops * k_loops:
                    communication_internal += (k + Array_i - f.expansion(k, Array_i)) * Array_j * 3
                elif a_loops % n_loops == 0 and a_loops != n_loops * k_loops:
                    communication_internal += Array_i * (n + Array_j - f.expansion(n, Array_j)) * 3
                elif a_loops == n_loops * k_loops:
                    communication_internal += (k + Array_i - f.expansion(k, Array_i)) * (n + Array_j - f.expansion(n, Array_j)) * 3
                else:
                    communication_internal += Array_i * Array_j * 3
                A_interval_work = A_interval_spare
                A_interval_spare = (Array_i-1)*compute_pipeline + m + Array_j

        # A_buf已经读完此列，但没有用尽整个A_buf
        elif G_A0_ptr%(m*Array_i)==0 and a_loops != n_loops*k_loops:
            if a_loops % n_loops == 0:
                if B_SA_load >= 1:
                    B_SA_load -= 1
                    G_A0_ptr += Array_i
                    # a读取最后一列
                    if a_loops - n_loops * (k_loops - 1) > 0:
                        # Array_i个数据从A_buf读出
                        communication_on_chip += k + Array_i - f.expansion(k, Array_i)
                        # Array_i个数据从A_buf到DR FIFO数据重组经历的额外读写次数
                        communication_interval += k + Array_i - f.expansion(k, Array_i) - 1
                        # Array_i个数据在PE之间要传递Array_j-1次
                        communication_interval += (k + Array_i - f.expansion(k, Array_i)) * (Array_j - 1)
                    else:
                        # Array_i个数据从A_buf读出
                        communication_on_chip += Array_i
                        # Array_i个数据从A_buf到DR FIFO数据重组经历的额外读写次数
                        communication_interval += Array_i - 1
                        # Array_i个数据在PE之间要传递Array_j-1次
                        communication_interval += Array_i * (Array_j - 1)
                    new_compute = 1
                    if a_loops - n_loops * (k_loops - 1) > 0 and a_loops != n_loops * k_loops:
                        communication_internal += (k + Array_i - f.expansion(k, Array_i)) * Array_j * 3
                    elif a_loops % n_loops == 0 and a_loops != n_loops * k_loops:
                        communication_internal += Array_i * (n + Array_j - f.expansion(n, Array_j)) * 3
                    elif a_loops == n_loops * k_loops:
                        communication_internal += (k + Array_i - f.expansion(k, Array_i)) * (n + Array_j - f.expansion(n, Array_j)) * 3
                    else:
                        communication_internal += Array_i * Array_j * 3
                    A_interval_work = A_interval_spare
                    A_interval_spare = (Array_i-1)*compute_pipeline + m + Array_j
                    a_loops += 1
            # 如果没用完,把A重新读一遍
            else:
                if B_SA_load >= 1:
                    B_SA_load -= 1
                    G_A0_ptr -= (m - 1) * Array_i
                    # a读取最后一列
                    if a_loops - n_loops * (k_loops - 1) > 0:
                        # Array_i个数据从A_buf读出
                        communication_on_chip += k + Array_i - f.expansion(k, Array_i)
                        # Array_i个数据从A_buf到DR FIFO数据重组经历的额外读写次数
                        communication_interval += k + Array_i - f.expansion(k, Array_i) - 1
                        # Array_i个数据在PE之间要传递Array_j-1次
                        communication_interval += (k + Array_i - f.expansion(k, Array_i)) * (Array_j - 1)
                    else:
                        # Array_i个数据从A_buf读出
                        communication_on_chip += Array_i
                        # Array_i个数据从A_buf到DR FIFO数据重组经历的额外读写次数
                        communication_interval += Array_i - 1
                        # Array_i个数据在PE之间要传递Array_j-1次
                        communication_interval += Array_i * (Array_j - 1)
                    new_compute = 1
                    if a_loops - n_loops * (k_loops - 1) > 0 and a_loops != n_loops * k_loops:
                        communication_internal += (k + Array_i - f.expansion(k, Array_i)) * Array_j * 3
                    elif a_loops % n_loops == 0 and a_loops != n_loops * k_loops:
                        communication_internal += Array_i * (n + Array_j - f.expansion(n, Array_j)) * 3
                    elif a_loops == n_loops * k_loops:
                        communication_internal += (k + Array_i - f.expansion(k, Array_i)) * (n + Array_j - f.expansion(n, Array_j)) * 3
                    else:
                        communication_internal += Array_i * Array_j * 3
                    A_interval_work = A_interval_spare
                    A_interval_spare = (Array_i-1)*compute_pipeline + m + Array_j
                    a_loops += 1
        # 其他正常读数的情况以及A_buf已用尽的情况
        else:
            # A_buf未用尽，正常读数
            if G_A0_ptr != A_buf_tile:
                G_A0_ptr += Array_i
                # a读取最后一列
                if a_loops - n_loops * (k_loops - 1) > 0:
                    # Array_i个数据从A_buf读出
                    communication_on_chip += k + Array_i - f.expansion(k, Array_i)
                    # Array_i个数据从A_buf到DR FIFO数据重组经历的额外读写次数
                    communication_interval += k + Array_i - f.expansion(k, Array_i) - 1
                    # Array_i个数据在PE之间要传递Array_j-1次
                    communication_interval += (k + Array_i - f.expansion(k, Array_i)) * (Array_j - 1)
                else:
                    # Array_i个数据从A_buf读出
                    communication_on_chip += Array_i
                    # Array_i个数据从A_buf到DR FIFO数据重组经历的额外读写次数
                    communication_interval += Array_i - 1
                    # Array_i个数据在PE之间要传递Array_j-1次
                    communication_interval += Array_i * (Array_j - 1)
                new_compute = 1
                if a_loops - n_loops * (k_loops - 1) > 0 and a_loops != n_loops * k_loops:
                    communication_internal += (k + Array_i - f.expansion(k, Array_i)) * Array_j * 3
                elif a_loops % n_loops == 0 and a_loops != n_loops * k_loops:
                    communication_internal += Array_i * (n + Array_j - f.expansion(n, Array_j)) * 3
                elif a_loops == n_loops * k_loops:
                    communication_internal += (k + Array_i - f.expansion(k, Array_i)) * (n + Array_j - f.expansion(n, Array_j)) * 3
                else:
                    communication_internal += Array_i * Array_j * 3
            # A_buf已用尽
            if G_A0_ptr == A_buf_tile and a_loops == n_loops*k_loops:
                on_chip_alter = 1
                G_A0_ptr = 0
                G_C1_ptr = 0
                G_A_in_use = 1


    # 当B_buf_0为工作buf时的过程****************************************************************
    if G_B_in_use == 0:

        # 当B收到信号读取小矩阵时
        if A_interval_work == Array_i:
            G_B_ptr_add += Array_i
        if A_interval_spare == Array_i:
            G_B_ptr_add += Array_i
        if G_B_ptr_add > 0 and G_B0_ptr != B_buf_tile:
            G_B0_ptr += Array_j
            # b在读最后一列
            if b_loops % n_loops == 0 and b_loops != n_loops * k_loops:
                # Array_j个数据从B_buf读出
                communication_on_chip += n + Array_j - f.expansion(n, Array_j)
                # Array_j个数在PE间传递的次数不同，从1到Array_i-1，为了简化计算同时不失正确性，这里用平均数
                communication_interval += (Array_i - 1) * (n + Array_j - f.expansion(n, Array_j))
            # b在读最后一行
            elif b_loops - (k_loops - 1) * n_loops > 0 and b_loops != n_loops * k_loops:
                # Array_j个数据从B_buf读出
                communication_on_chip += Array_j * (k + Array_i - f.expansion(k, Array_i)) / Array_i
                # Array_j个数在PE间传递的次数不同，从1到Array_i-1，为了简化计算同时不失正确性，这里用平均数
                communication_interval += (Array_i - 1) * (Array_j * (k + Array_i - f.expansion(k, Array_i)) / Array_i)
            # b在读最后一块
            elif b_loops == n_loops * k_loops:
                # Array_j个数据从B_buf读出
                communication_on_chip += (n + Array_j - f.expansion(n, Array_j)) * (k + Array_i - f.expansion(k, Array_i)) / Array_i
                # Array_j个数在PE间传递的次数不同，从1到Array_i-1，为了简化计算同时不失正确性，这里用平均数
                communication_interval += (Array_i - 1) * ((n + Array_j - f.expansion(n, Array_j)) * (k + Array_i - f.expansion(k, Array_i)) / Array_i)
            else:
                # Array_j个数据从B_buf读出
                communication_on_chip += Array_j
                # Array_j个数在PE间传递的次数不同，从1到Array_i-1，为了简化计算同时不失正确性，这里用平均数
                communication_interval += (Array_i - 1) * Array_j
            G_B_ptr_add -= 1
            # B_buf装载完成后，SA内可用B分块+1
            if G_B0_ptr % (Array_i*Array_j) == 0:
                B_SA_load += 1
                b_loops += 1
        A_interval_work -= 1
        A_interval_spare -= 1
        # 当B_buf0中的数据用完时,更换B_buf1
        if G_B0_ptr == B_buf_tile:
            G_B_in_use = 1
            G_B0_ptr = 0
            G_B0_ready = 0

    # 模拟SA的计算流水过程
    # 每一个输出结果需要进行Array_i个MAC运算,每个MAC运算是compute_pipline拍流水,因此一个部分和在\
    # SA中输出时相比第一行数据输入时需要经过Array_i*compute_pipeline拍,这里采用数组进行一个部分和\
    # 每一拍都会向下一个元素流动,1表示存在MAC的数,0表示此拍空闲
    for i in range(1,Array_i*compute_pipeline):
        SA_list[Array_i*compute_pipeline-i] = SA_list[Array_i*compute_pipeline-i-1]
    SA_list[0] = new_compute


    # 当C_buf0为工作buf时，读取C_buf中的数据到SA中与A*B相加**************************************
    if G_Cr_in_use == 0:

        # 正常更新
        if new_compute == 1:
            G_C0_ptr += Array_j
            # c读取最后一列
            if a_loops % n_loops == 0:
                # Array_j个数据从C_buf读出与写回C_buf
                communication_on_chip += 2 * (n + Array_j - f.expansion(n, Array_j))
                # Array_j个数据从C_buf到DR FIFO数据重组经历的两次额外读写次数
                communication_interval += 2 * (n + Array_j - f.expansion(n, Array_j) - 1)
                # Array_j个数据在PE之间要传递Array_i-1次
                communication_interval += (n + Array_j - f.expansion(n, Array_j)) * (Array_i - 1)
            # c读取其他列
            else:
                # Array_j个数据从C_buf读出与写回C_buf
                communication_on_chip += 2 * Array_j
                # Array_j个数据从C_buf到DR FIFO数据重组经历的两次额外读写次数
                communication_interval += 2 * (Array_j - 1)
                # Array_j个数据在PE之间要传递Array_i-1次
                communication_interval += Array_j * (Array_i - 1)
        # 当C_buf已经全部读过一遍的时候
        if G_C0_ptr == C_buf_tile:
            if cr_loops != k_loops:
                cr_loops += 1
                G_C0_ptr = 0
            else:
                if G_A_in_use == 1:
                    G_Cr_in_use = 1
                    G_C0_ptr = 0

    return G_A_in_use, G_A0_ready, G_A1_ready, G_A0_ptr, G_A0_ptw, G_A1_ptr, G_A1_ptw, a_loops, b_loops,\
            G_B0_ready, G_B1_ready, G_B0_ptr, G_B0_ptw, G_B1_ptr, G_B1_ptw, G_B_in_use,\
            B_SA_load, A_interval_work, A_interval_spare, G_C0_ready, G_C0_ptw, G_C0_ptr,\
            G_C1_ready, G_C1_ptw, G_C1_ptr, G_Cw_in_use, G_Cr_in_use, cw_loops, cr_loops,\
            on_chip_alter, bw_in_use, G_B_ptr_add, communication_Y_cycle, communication_on_chip,\
            communication_interval, communication_internal

# 将C_buf的数据写完，并写回HBM
def write_back(
                write_back_flag,
                G_Cr_in_use,
                G_C0_ptr,
                G_C0_ptw,
                G_C1_ptr,
                G_C1_ptw,
                DR_list,
                SA_list,
                Array_i,
                Array_j,
                compute_pipeline,
                m,
                n,
                bandwidth,
                Y_buf_tile_trans,
                communication_Y_cycle
                ):
    C_buf_tile = m * f.expansion(n, Array_j)
    # 由于代码问题, G_Cr_in_use=1时使用的是C_buf0
    if G_Cr_in_use == 1:
        if write_back_flag == 0:
            if DR_list[Array_j-2] == 1:
                G_C0_ptw += Array_j
            for i in range(1,Array_j-1):
                DR_list[Array_j-1-i] = DR_list[Array_j-2-i]
            DR_list[0] = SA_list[-1]
            for i in range(1,Array_i*compute_pipeline):
                SA_list[Array_i*compute_pipeline-i] = SA_list[Array_i*compute_pipeline-i-1]
            SA_list[0] = 0
            if np.sum(np.array(SA_list))==0 and np.sum(np.array(DR_list))==0:
                G_C0_ptw = 0
                write_back_flag = 1
            if G_C0_ptw == C_buf_tile:
                G_C0_ptw = 0
        else:
            G_C0_ptr += bandwidth
            communication_Y_cycle += 1
            if G_C0_ptr == Y_buf_tile_trans:
                G_C0_ptr = 0
                write_back_flag = 2
    else:
        if write_back_flag == 0:
            if DR_list[Array_j-2] == 1:
                G_C1_ptw += Array_j
            for i in range(1,Array_j-1):
                DR_list[Array_j-1-i] = DR_list[Array_j-2-i]
            DR_list[0] = SA_list[-1]
            for i in range(1,Array_i*compute_pipeline):
                SA_list[Array_i*compute_pipeline-i] = SA_list[Array_i*compute_pipeline-i-1]
            SA_list[0] = 0
            if np.sum(np.array(SA_list))==0 and np.sum(np.array(DR_list))==0:
                G_C1_ptw = 0
                write_back_flag = 1
            if G_C1_ptw == C_buf_tile:
                G_C1_ptw = 0
        else:
            G_C1_ptr += bandwidth
            communication_Y_cycle += 1
            if G_C1_ptr == Y_buf_tile_trans:
                G_C1_ptr = 0
                write_back_flag = 2
    return write_back_flag, G_Cr_in_use, G_C0_ptr, G_C0_ptw, G_C1_ptr, G_C1_ptw, communication_Y_cycle
