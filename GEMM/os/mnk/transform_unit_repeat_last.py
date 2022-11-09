# encoding: utf-8
from __future__ import division
from __future__ import print_function
# GSM内各个buffer读写的记录
# 包括weight在A_buf的读出，备用A_buf与B_buf的写入
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
                G_B0_ready,
                G_B1_ready,
                G_B0_ptr,
                G_B0_ptw,
                G_B1_ptr,
                G_B1_ptw,
                G_B_in_use,
                C_SA_load,
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
                b_loops,
                Array_i,
                Array_j,
                m,
                n,
                k,
                m_before,
                n_before,
                m_next,
                n_next,
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
                G_C_ptr_add,
                G_C_ptw_add,
                C_compute_in_SA,
                communication_on_chip,
                communication_interval,
                communication_internal,
                counter_A_read,
                c_loops
                ):
    new_compute = 0
    m_loops = f.roundup(m, Array_i)
    n_loops = f.roundup(n, Array_j)
    m_loops_before = f.roundup(m_before, Array_i)
    n_loops_before = f.roundup(n_before, Array_j)
    A_buf_tile, B_buf_tile = f.expansion(m, Array_i) * k, k * f.expansion(n, Array_j)
    C_buf_tile_r, C_buf_tile_w = f.expansion(m, Array_i) * f.expansion(n, Array_j), f.expansion(m, Array_i) * f.expansion(n, Array_j)
    C_buf_tile_w_before = f.expansion(m_before, Array_i) * f.expansion(n_before, Array_j)
    C_buf_tile_r_next = f.expansion(m_next, Array_i) * f.expansion(n_next, Array_j)

    # 确定SA的计算结果写回C_buf的数据量
    if G_Cw_in_use != G_A_in_use:
        C_buf_tile_w = C_buf_tile_w_before
    # 确定进入SA的C上限
    if G_Cr_in_use != G_A_in_use:
        C_buf_tile_r = C_buf_tile_r_next

    # 当C_buf0为工作buf时的读SA流出的数据读取数据到工作C_buf的过程*************************
    if G_Cw_in_use == 0:
        # 将C_buf中的数据写入HBM
        if G_Cw_in_use == G_A_in_use:
            if C_load_out == 1:
                if G_C1_ptr != Y_buf_tile_trans and G_C1_ready == 0:
                    G_C1_ptr += bandwidth
                    communication_Y_cycle += 1
                    if G_C1_ptr == Y_buf_tile_trans:
                        G_C1_ready = 1
            elif C_load_out == 0 and G_C1_ready == 0:
                G_C1_ready = 1


        # 将SA中保存的C计算结果写回C_buf
        if C_compute_in_SA > 0:
            if G_C_ptw_add > 0 and G_C0_ptw != C_buf_tile_w:
                G_C0_ptw += Array_j
                G_C_ptw_add -= 1
                # C_buf写回完成后，SA内正在计算C分块-1
                if G_C0_ptw % (Array_i * Array_j) == 0:
                    C_compute_in_SA -= 1

        # 在G_C0_ptw写满之后，判断此分块是否已完全循环结束，是则更换，否则接着循环
        if G_C0_ptw == C_buf_tile_w and G_A_in_use == 1:
            G_Cw_in_use = 1
            G_C0_ptw = 0
            G_C0_ready = 0


    # 当C_buf1为工作buf时的读SA流出的数据读取数据到工作C_buf的过程*************************
    else:
        # 将C_buf中的数据写入HBM
        if G_Cw_in_use == G_A_in_use:
            if C_load_out == 1:
                if G_C0_ptr != Y_buf_tile_trans and G_C0_ready == 0:
                    G_C0_ptr += bandwidth
                    communication_Y_cycle += 1
                    if G_C0_ptr == Y_buf_tile_trans:
                        G_C0_ready = 1
            elif C_load_out == 0 and G_C0_ready == 0:
                G_C0_ready = 1


        # 将SA中保存的C计算结果写回C_buf
        if C_compute_in_SA > 0:
            if G_C_ptw_add > 0 and G_C1_ptw != C_buf_tile_w:
                G_C1_ptw += Array_j
                G_C_ptw_add -= 1
                # C_buf写回完成后，SA内正在计算C分块-1
                if G_C1_ptw % (Array_i * Array_j) == 0:
                    C_compute_in_SA -= 1
        # 在G_C1_ptw写满之后，判断此分块是否已完全循环结束，是则更换，否则接着循环
        if G_C1_ptw == C_buf_tile_w and G_A_in_use == 0:
            G_Cw_in_use = 0
            G_C1_ptw = 0
            G_C1_ready = 0



    # 当A_buf_0为工作buf时的过程****************************************************************
    if G_A_in_use == 0:

        # 将A_buf0的数据读出
        # 在每一次A_buf0的第一组数据读出之前,检查B中小分块在SA中是否到位(B_SA_load=1则到位,0则仍需等待),\
        # 因为有可能HBM与B_buf的通信导致延迟
        if G_A0_ptr == 0 and a_loops == 1:
            if C_SA_load >= 1 and counter_A_read <= 0:
                C_SA_load -= 1
                counter_A_read = compute_pipeline
                G_A0_ptr += Array_i
                # a读取最后一列
                if a_loops - n_loops * (m_loops - 1) > 0:
                    # Array_i个数据从A_buf读出
                    communication_on_chip += m + Array_i - f.expansion(m, Array_i)
                    # Array_i个数据从A_buf到DR FIFO数据重组经历的额外读写次数
                    communication_interval += m + Array_i - f.expansion(m, Array_i) - 1
                    # Array_i个数据在PE之间要传递Array_j-1次
                    communication_interval += (m + Array_i - f.expansion(m, Array_i)) * (Array_j - 1)
                else:
                    # Array_i个数据从A_buf读出
                    communication_on_chip += Array_i
                    # Array_i个数据从A_buf到DR FIFO数据重组经历的额外读写次数
                    communication_interval += Array_i - 1
                    # Array_i个数据在PE之间要传递Array_j-1次
                    communication_interval += Array_i * (Array_j - 1)
                new_compute = 1
                if a_loops - n_loops * (m_loops - 1) > 0 and a_loops != n_loops * m_loops:
                    communication_internal += (m + Array_i - f.expansion(m, Array_i)) * Array_j * 3
                elif a_loops % n_loops == 0 and a_loops != n_loops * m_loops:
                    communication_internal += Array_i * (n + Array_j - f.expansion(n, Array_j)) * 3
                elif a_loops == n_loops * m_loops:
                    communication_internal += (m + Array_i - f.expansion(m, Array_i)) * (n + Array_j - f.expansion(n, Array_j)) * 3
                else:
                    communication_internal += Array_i * Array_j * 3
                A_interval_work = A_interval_spare
                A_interval_spare = Array_i + Array_i + Array_j + k * compute_pipeline + 1

        # A_buf已经读完此列，但没有用尽整个A_buf
        elif G_A0_ptr%(k*Array_i)==0 and a_loops != m_loops*n_loops:
            # 如果C一行的小矩阵用完，此时a_loops循环了n_loops次
            if a_loops % n_loops == 0:
                if C_SA_load >= 1 and counter_A_read <= 0:
                    C_SA_load -= 1
                    counter_A_read = compute_pipeline
                    G_A0_ptr += Array_i
                    # a读取最后一列
                    if a_loops - n_loops * (m_loops - 1) > 0:
                        # Array_i个数据从A_buf读出
                        communication_on_chip += m + Array_i - f.expansion(m, Array_i)
                        # Array_i个数据从A_buf到DR FIFO数据重组经历的额外读写次数
                        communication_interval += m + Array_i - f.expansion(m, Array_i) - 1
                        # Array_i个数据在PE之间要传递Array_j-1次
                        communication_interval += (m + Array_i - f.expansion(m, Array_i)) * (Array_j - 1)
                    else:
                        # Array_i个数据从A_buf读出
                        communication_on_chip += Array_i
                        # Array_i个数据从A_buf到DR FIFO数据重组经历的额外读写次数
                        communication_interval += Array_i - 1
                        # Array_i个数据在PE之间要传递Array_j-1次
                        communication_interval += Array_i * (Array_j - 1)
                    new_compute = 1
                    if a_loops - n_loops * (m_loops - 1) > 0 and a_loops != n_loops * m_loops:
                        communication_internal += (m + Array_i - f.expansion(m, Array_i)) * Array_j * 3
                    elif a_loops % n_loops == 0 and a_loops != n_loops * m_loops:
                        communication_internal += Array_i * (n + Array_j - f.expansion(n, Array_j)) * 3
                    elif a_loops == n_loops * m_loops:
                        communication_internal += (m + Array_i - f.expansion(m, Array_i)) * (n + Array_j - f.expansion(n, Array_j)) * 3
                    else:
                        communication_internal += Array_i * Array_j * 3
                    A_interval_work = A_interval_spare
                    A_interval_spare = Array_i + Array_i + Array_j + k * compute_pipeline + 1
                    a_loops += 1

            # 如果没用完，把A重新读一遍
            else:
                if C_SA_load >= 1 and counter_A_read <= 0:
                    C_SA_load -= 1
                    counter_A_read = compute_pipeline
                    G_A0_ptr -= (k - 1) * Array_i
                    # a读取最后一列
                    if a_loops - n_loops * (m_loops - 1) > 0:
                        # Array_i个数据从A_buf读出
                        communication_on_chip += m + Array_i - f.expansion(m, Array_i)
                        # Array_i个数据从A_buf到DR FIFO数据重组经历的额外读写次数
                        communication_interval += m + Array_i - f.expansion(m, Array_i) - 1
                        # Array_i个数据在PE之间要传递Array_j-1次
                        communication_interval += (m + Array_i - f.expansion(m, Array_i)) * (Array_j - 1)
                    else:
                        # Array_i个数据从A_buf读出
                        communication_on_chip += Array_i
                        # Array_i个数据从A_buf到DR FIFO数据重组经历的额外读写次数
                        communication_interval += Array_i - 1
                        # Array_i个数据在PE之间要传递Array_j-1次
                        communication_interval += Array_i * (Array_j - 1)
                    new_compute = 1
                    if a_loops - n_loops * (m_loops - 1) > 0 and a_loops != n_loops * m_loops:
                        communication_internal += (m + Array_i - f.expansion(m, Array_i)) * Array_j * 3
                    elif a_loops % n_loops == 0 and a_loops != n_loops * m_loops:
                        communication_internal += Array_i * (n + Array_j - f.expansion(n, Array_j)) * 3
                    elif a_loops == n_loops * m_loops:
                        communication_internal += (m + Array_i - f.expansion(m, Array_i)) * (n + Array_j - f.expansion(n, Array_j)) * 3
                    else:
                        communication_internal += Array_i * Array_j * 3
                    A_interval_work = A_interval_spare
                    A_interval_spare = Array_i + Array_i + Array_j + k * compute_pipeline + 1
                    a_loops += 1
        # 其他正常读数的情况以及A_buf已用尽的情况
        else:
            # A_buf未用尽，正常读数
            if G_A0_ptr != A_buf_tile and counter_A_read <= 0:
                counter_A_read = compute_pipeline
                G_A0_ptr += Array_i
                # a读取最后一列
                if a_loops - n_loops * (m_loops - 1) > 0:
                    # Array_i个数据从A_buf读出
                    communication_on_chip += m + Array_i - f.expansion(m, Array_i)
                    # Array_i个数据从A_buf到DR FIFO数据重组经历的额外读写次数
                    communication_interval += m + Array_i - f.expansion(m, Array_i) - 1
                    # Array_i个数据在PE之间要传递Array_j-1次
                    communication_interval += (m + Array_i - f.expansion(m, Array_i)) * (Array_j - 1)
                else:
                    # Array_i个数据从A_buf读出
                    communication_on_chip += Array_i
                    # Array_i个数据从A_buf到DR FIFO数据重组经历的额外读写次数
                    communication_interval += Array_i - 1
                    # Array_i个数据在PE之间要传递Array_j-1次
                    communication_interval += Array_i * (Array_j - 1)
                new_compute = 1
                if a_loops - n_loops * (m_loops - 1) > 0 and a_loops != n_loops * m_loops:
                    communication_internal += (m + Array_i - f.expansion(m, Array_i)) * Array_j * 3
                elif a_loops % n_loops == 0 and a_loops != n_loops * m_loops:
                    communication_internal += Array_i * (n + Array_j - f.expansion(n, Array_j)) * 3
                elif a_loops == n_loops * m_loops:
                    communication_internal += (m + Array_i - f.expansion(m, Array_i)) * (n + Array_j - f.expansion(n, Array_j)) * 3
                else:
                    communication_internal += Array_i * Array_j * 3
            # A_buf已用尽
            if G_A0_ptr == A_buf_tile and a_loops == m_loops*n_loops and G_Cw_in_use == 0:
                on_chip_alter = 1
                G_A0_ptr = 0
                G_C1_ptr = 0
                G_A_in_use = 1

    # 当A_buf_1为工作buf时的过程****************************************************************
    else:
        # 将A_buf1的数据读出
        # 在每一次A_buf1的第一组数据读出之前,检查B中小分块在SA中是否到位(B_SA_load=1则到位,0则仍需等待),\
        # 因为有可能HBM与B_buf的通信导致延迟
        if G_A1_ptr == 0 and a_loops == 1:
            if C_SA_load >= 1 and counter_A_read <= 0:
                C_SA_load -= 1
                counter_A_read = compute_pipeline
                G_A1_ptr += Array_i
                # a读取最后一列
                if a_loops - n_loops * (m_loops - 1) > 0:
                    # Array_i个数据从A_buf读出
                    communication_on_chip += m + Array_i - f.expansion(m, Array_i)
                    # Array_i个数据从A_buf到DR FIFO数据重组经历的额外读写次数
                    communication_interval += m + Array_i - f.expansion(m, Array_i) - 1
                    # Array_i个数据在PE之间要传递Array_j-1次
                    communication_interval += (m + Array_i - f.expansion(m, Array_i)) * (Array_j - 1)
                else:
                    # Array_i个数据从A_buf读出
                    communication_on_chip += Array_i
                    # Array_i个数据从A_buf到DR FIFO数据重组经历的额外读写次数
                    communication_interval += Array_i - 1
                    # Array_i个数据在PE之间要传递Array_j-1次
                    communication_interval += Array_i * (Array_j - 1)
                new_compute = 1
                if a_loops - n_loops * (m_loops - 1) > 0 and a_loops != n_loops * m_loops:
                    communication_internal += (m + Array_i - f.expansion(m, Array_i)) * Array_j * 3
                elif a_loops % n_loops == 0 and a_loops != n_loops * m_loops:
                    communication_internal += Array_i * (n + Array_j - f.expansion(n, Array_j)) * 3
                elif a_loops == n_loops * m_loops:
                    communication_internal += (m + Array_i - f.expansion(m, Array_i)) * (n + Array_j - f.expansion(n, Array_j)) * 3
                else:
                    communication_internal += Array_i * Array_j * 3
                A_interval_work = A_interval_spare
                A_interval_spare = Array_i + Array_i + Array_j + k * compute_pipeline + 1

        # A_buf已经读完此列，但没有用尽整个A_buf
        elif G_A1_ptr%(k*Array_i)==0 and a_loops != m_loops*n_loops:
            # 如果C一行的小矩阵，此时a_loops循环了n_loops次
            if a_loops % n_loops == 0:
                if C_SA_load >= 1:
                    C_SA_load -= 1
                    counter_A_read = compute_pipeline
                    G_A1_ptr += Array_i
                    # a读取最后一列
                    if a_loops - n_loops * (m_loops - 1) > 0:
                        # Array_i个数据从A_buf读出
                        communication_on_chip += m + Array_i - f.expansion(m, Array_i)
                        # Array_i个数据从A_buf到DR FIFO数据重组经历的额外读写次数
                        communication_interval += m + Array_i - f.expansion(m, Array_i) - 1
                        # Array_i个数据在PE之间要传递Array_j-1次
                        communication_interval += (m + Array_i - f.expansion(m, Array_i)) * (Array_j - 1)
                    else:
                        # Array_i个数据从A_buf读出
                        communication_on_chip += Array_i
                        # Array_i个数据从A_buf到DR FIFO数据重组经历的额外读写次数
                        communication_interval += Array_i - 1
                        # Array_i个数据在PE之间要传递Array_j-1次
                        communication_interval += Array_i * (Array_j - 1)
                    new_compute = 1
                    if a_loops - n_loops * (m_loops - 1) > 0 and a_loops != n_loops * m_loops:
                        communication_internal += (m + Array_i - f.expansion(m, Array_i)) * Array_j * 3
                    elif a_loops % n_loops == 0 and a_loops != n_loops * m_loops:
                        communication_internal += Array_i * (n + Array_j - f.expansion(n, Array_j)) * 3
                    elif a_loops == n_loops * m_loops:
                        communication_internal += (m + Array_i - f.expansion(m, Array_i)) * (n + Array_j - f.expansion(n, Array_j)) * 3
                    else:
                        communication_internal += Array_i * Array_j * 3
                    A_interval_work = A_interval_spare
                    A_interval_spare = Array_i + Array_i + Array_j + k * compute_pipeline + 1
                    a_loops += 1

            # 如果没用完，把A重新读一遍
            else:
                if C_SA_load >= 1 and counter_A_read <= 0:
                    C_SA_load -= 1
                    counter_A_read = compute_pipeline
                    G_A1_ptr -= (k - 1) * Array_i
                    # a读取最后一列
                    if a_loops - n_loops * (m_loops - 1) > 0:
                        # Array_i个数据从A_buf读出
                        communication_on_chip += m + Array_i - f.expansion(m, Array_i)
                        # Array_i个数据从A_buf到DR FIFO数据重组经历的额外读写次数
                        communication_interval += m + Array_i - f.expansion(m, Array_i) - 1
                        # Array_i个数据在PE之间要传递Array_j-1次
                        communication_interval += (m + Array_i - f.expansion(m, Array_i)) * (Array_j - 1)
                    else:
                        # Array_i个数据从A_buf读出
                        communication_on_chip += Array_i
                        # Array_i个数据从A_buf到DR FIFO数据重组经历的额外读写次数
                        communication_interval += Array_i - 1
                        # Array_i个数据在PE之间要传递Array_j-1次
                        communication_interval += Array_i * (Array_j - 1)
                    new_compute = 1
                    if a_loops - n_loops * (m_loops - 1) > 0 and a_loops != n_loops * m_loops:
                        communication_internal += (m + Array_i - f.expansion(m, Array_i)) * Array_j * 3
                    elif a_loops % n_loops == 0 and a_loops != n_loops * m_loops:
                        communication_internal += Array_i * (n + Array_j - f.expansion(n, Array_j)) * 3
                    elif a_loops == n_loops * m_loops:
                        communication_internal += (m + Array_i - f.expansion(m, Array_i)) * (n + Array_j - f.expansion(n, Array_j)) * 3
                    else:
                        communication_internal += Array_i * Array_j * 3
                    A_interval_work = A_interval_spare
                    A_interval_spare = Array_i + Array_i + Array_j + k * compute_pipeline + 1
                    a_loops += 1
        # 其他正常读数的情况以及A_buf已用尽的情况
        else:
            # A_buf未用尽，正常读数
            if G_A1_ptr != A_buf_tile and counter_A_read <= 0:
                counter_A_read = compute_pipeline
                G_A1_ptr += Array_i
                # a读取最后一列
                if a_loops - n_loops * (m_loops - 1) > 0:
                    # Array_i个数据从A_buf读出
                    communication_on_chip += m + Array_i - f.expansion(m, Array_i)
                    # Array_i个数据从A_buf到DR FIFO数据重组经历的额外读写次数
                    communication_interval += m + Array_i - f.expansion(m, Array_i) - 1
                    # Array_i个数据在PE之间要传递Array_j-1次
                    communication_interval += (m + Array_i - f.expansion(m, Array_i)) * (Array_j - 1)
                else:
                    # Array_i个数据从A_buf读出
                    communication_on_chip += Array_i
                    # Array_i个数据从A_buf到DR FIFO数据重组经历的额外读写次数
                    communication_interval += Array_i - 1
                    # Array_i个数据在PE之间要传递Array_j-1次
                    communication_interval += Array_i * (Array_j - 1)
                new_compute = 1
                if a_loops - n_loops * (m_loops - 1) > 0 and a_loops != n_loops * m_loops:
                    communication_internal += (m + Array_i - f.expansion(m, Array_i)) * Array_j * 3
                elif a_loops % n_loops == 0 and a_loops != n_loops * m_loops:
                    communication_internal += Array_i * (n + Array_j - f.expansion(n, Array_j)) * 3
                elif a_loops == n_loops * m_loops:
                    communication_internal += (m + Array_i - f.expansion(m, Array_i)) * (n + Array_j - f.expansion(n, Array_j)) * 3
                else:
                    communication_internal += Array_i * Array_j * 3
            # A_buf已用尽
            if G_A1_ptr == A_buf_tile and a_loops == m_loops*n_loops and G_Cw_in_use == 1:
                G_A1_ptr = 0
                on_chip_alter = 1
                G_C0_ptr = 0
                G_A_in_use = 0

    counter_A_read -= 1
    # 当C_buf0为工作buf时**************************************
    if G_Cr_in_use == 0:
        # 当C收到信号读取小矩阵时
        if A_interval_work == Array_i:
            G_C_ptr_add += Array_i
            G_C_ptw_add += Array_i
        if A_interval_spare == Array_i:
            G_C_ptr_add += Array_i
            G_C_ptw_add += Array_i
        if G_C_ptr_add > 0 and G_C0_ptr != C_buf_tile_r and G_Cr_in_use == G_A_in_use:
            G_C0_ptr += Array_j
            # c在读最后一列
            if c_loops % n_loops == 0 and c_loops != n_loops * m_loops:
                # Array_j个数据从C_buf读出与写回C_buf
                communication_on_chip += 2 * (n + Array_j - f.expansion(n, Array_j))
                # Array_j个数在PE间传递的次数不同，从1到Array_i-1，为了简化计算同时不失正确性，这里用平均数
                communication_interval += (Array_i - 1) * (n + Array_j - f.expansion(n, Array_j))
            # c在读最后一行
            elif c_loops - (m_loops - 1) * n_loops > 0 and c_loops != n_loops * m_loops:
                # Array_j个数据从c_buf读出
                communication_on_chip += 2 * (Array_j * (m + Array_i - f.expansion(m, Array_i)) / Array_i)
                # Array_j个数在PE间传递的次数不同，从1到Array_i-1，为了简化计算同时不失正确性，这里用平均数
                communication_interval += (Array_i - 1) * (Array_j * (m + Array_i - f.expansion(m, Array_i)) / Array_i)
            # c在读最后一块
            elif c_loops == n_loops * m_loops:
                # Array_j个数据从C_buf读出与写回C_buf
                communication_on_chip += 2 * ((n + Array_j - f.expansion(n, Array_j)) * (m + Array_i - f.expansion(m, Array_i)) / Array_i)
                # Array_j个数在PE间传递的次数不同，从1到Array_i-1，为了简化计算同时不失正确性，这里用平均数
                communication_interval += (Array_i - 1) * ((n + Array_j - f.expansion(n, Array_j)) * (m + Array_i - f.expansion(m, Array_i)) / Array_i)
            else:
                # Array_j个数据从C_buf读出与写回C_buf
                communication_on_chip += 2 * Array_j
                # Array_j个数在PE间传递的次数不同，从1到Array_i-1，为了简化计算同时不失正确性，这里用平均数
                communication_interval += (Array_i - 1) * Array_j
            G_C_ptr_add -= 1
            if G_C0_ptr % (Array_i * Array_j) == 0:
                C_compute_in_SA += 1
                c_loops += 1
                C_SA_load += 1


        if G_C0_ptr == C_buf_tile_r and G_A_in_use == 1:
            G_Cr_in_use = 1
            G_C0_ptr = 0
            G_C0_ready = 0
            c_loops = 1

    # 当C_buf1为工作buf时，读取C_buf中的数据到SA中与A*B相加**************************************
    else:
        # 当C收到信号读取小矩阵时
        if A_interval_work == Array_i:
            G_C_ptr_add += Array_i
            G_C_ptw_add += Array_i
        if A_interval_spare == Array_i:
            G_C_ptr_add += Array_i
            G_C_ptw_add += Array_i
        if G_C_ptr_add > 0 and G_C1_ptr != C_buf_tile_r and G_Cr_in_use == G_A_in_use:
            G_C1_ptr += Array_j
            # c在读最后一列
            if c_loops % n_loops == 0 and c_loops != n_loops * m_loops:
                # Array_j个数据从C_buf读出与写回C_buf
                communication_on_chip += 2 * (n + Array_j - f.expansion(n, Array_j))
                # Array_j个数在PE间传递的次数不同，从1到Array_i-1，为了简化计算同时不失正确性，这里用平均数
                communication_interval += (Array_i - 1) * (n + Array_j - f.expansion(n, Array_j))
            # c在读最后一行
            elif c_loops - (m_loops - 1) * n_loops > 0 and c_loops != n_loops * m_loops:
                # Array_j个数据从c_buf读出
                communication_on_chip += 2 * (Array_j * (m + Array_i - f.expansion(m, Array_i)) / Array_i)
                # Array_j个数在PE间传递的次数不同，从1到Array_i-1，为了简化计算同时不失正确性，这里用平均数
                communication_interval += (Array_i - 1) * (Array_j * (m + Array_i - f.expansion(m, Array_i)) / Array_i)
            # c在读最后一块
            elif c_loops == n_loops * m_loops:
                # Array_j个数据从C_buf读出与写回C_buf
                communication_on_chip += 2 * ((n + Array_j - f.expansion(n, Array_j)) * (m + Array_i - f.expansion(m, Array_i)) / Array_i)
                # Array_j个数在PE间传递的次数不同，从1到Array_i-1，为了简化计算同时不失正确性，这里用平均数
                communication_interval += (Array_i - 1) * ((n + Array_j - f.expansion(n, Array_j)) * (m + Array_i - f.expansion(m, Array_i)) / Array_i)
            else:
                # Array_j个数据从C_buf读出与写回C_buf
                communication_on_chip += 2 * Array_j
                # Array_j个数在PE间传递的次数不同，从1到Array_i-1，为了简化计算同时不失正确性，这里用平均数
                communication_interval += (Array_i - 1) * Array_j
            G_C_ptr_add -= 1
            if G_C1_ptr % (Array_i * Array_j) == 0:
                C_compute_in_SA += 1
                c_loops += 1
                C_SA_load += 1
        if G_C1_ptr == C_buf_tile_r and G_A_in_use == 0:
            G_Cr_in_use = 0
            G_C1_ptr = 0
            G_C1_ready = 0
            c_loops = 1

    A_interval_work -= 1
    A_interval_spare -= 1

    # 当B_buf_0为工作buf时的过程****************************************************************
    if G_B_in_use == 0:
        # 正常更新
        if new_compute == 1:
            G_B0_ptr += Array_j
            # b读取最后一列
            if a_loops % n_loops == 0:
                # Array_j个数据从C_buf读出
                communication_on_chip += n + Array_j - f.expansion(n, Array_j)
                # Array_j个数据从C_buf到DR FIFO数据重组经历的额外读写次数
                communication_interval += n + Array_j - f.expansion(n, Array_j) - 1
                # Array_j个数据在PE之间要传递Array_i-1次
                communication_interval += (n + Array_j - f.expansion(n, Array_j)) * (Array_i - 1)
            # b读取其他列
            else:
                # Array_j个数据从C_buf读出
                communication_on_chip += Array_j
                # Array_j个数据从C_buf到DR FIFO数据重组经历的额外读写次数
                communication_interval += Array_j - 1
                # Array_j个数据在PE之间要传递Array_i-1次
                communication_interval += Array_j * (Array_i - 1)
        # 当B_buf已经全部读过一遍时
        if G_B0_ptr == B_buf_tile:
            if b_loops != m_loops:
                b_loops += 1
                G_B0_ptr = 0
            else:
                if G_A_in_use == 1:
                    G_B_in_use = 1
                    G_B0_ptr = 0
    # 当B_buf_1为工作buf时的过程****************************************************************
    else:
        # 正常更新
        if new_compute == 1:
            G_B1_ptr += Array_j
            # b读取最后一列
            if a_loops % n_loops == 0:
                # Array_j个数据从C_buf读出
                communication_on_chip += n + Array_j - f.expansion(n, Array_j)
                # Array_j个数据从C_buf到DR FIFO数据重组经历的额外读写次数
                communication_interval += n + Array_j - f.expansion(n, Array_j) - 1
                # Array_j个数据在PE之间要传递Array_i-1次
                communication_interval += (n + Array_j - f.expansion(n, Array_j)) * (Array_i - 1)
            # b读取其他列
            else:
                # Array_j个数据从C_buf读出
                communication_on_chip += Array_j
                # Array_j个数据从C_buf到DR FIFO数据重组经历的额外读写次数
                communication_interval += Array_j - 1
                # Array_j个数据在PE之间要传递Array_i-1次
                communication_interval += Array_j * (Array_i - 1)
        # 当B_buf已经全部读过一遍时
        if G_B1_ptr == B_buf_tile:
            if b_loops != m_loops:
                b_loops += 1
                G_B1_ptr = 0
            else:
                if G_A_in_use == 0:
                    G_B_in_use = 0
                    G_B1_ptr = 0
    return G_A_in_use, G_A0_ready, G_A1_ready, G_A0_ptr, G_A0_ptw, G_A1_ptr, G_A1_ptw, a_loops,\
            G_B0_ready, G_B1_ready, G_B0_ptr, G_B0_ptw, G_B1_ptr, G_B1_ptw, G_B_in_use,\
            C_SA_load, A_interval_work, A_interval_spare, G_C0_ready, G_C0_ptw, G_C0_ptr,\
            G_C1_ready, G_C1_ptw, G_C1_ptr, G_Cw_in_use, G_Cr_in_use, b_loops,\
            on_chip_alter, bw_in_use, communication_Y_cycle, G_C_ptr_add, C_compute_in_SA, communication_on_chip,\
            communication_interval, communication_internal, counter_A_read, G_C_ptw_add, c_loops
