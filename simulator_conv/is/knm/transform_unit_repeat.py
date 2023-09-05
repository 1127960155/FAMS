# encoding: utf-8
from __future__ import division
from __future__ import print_function
import functions as f
import numpy as np
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
                m_before,
                n_before,
                k_before,
                n_next,
                k_next,
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
                ):
    new_compute = 0
    k_loops = f.roundup(k, Array_i)
    n_loops = f.roundup(n, Array_j)
    k_loops_before = f.roundup(k_before, Array_i)
    n_loops_before = f.roundup(n_before, Array_j)
    A_buf_tile, B_buf_tile = m * f.expansion(k, Array_i), f.expansion(k, Array_i) * f.expansion(n, Array_j)
    C_buf_tile_r, C_buf_tile_w = m * f.expansion(n, Array_j), m * f.expansion(n, Array_j)
    C_buf_tile_w_before = m_before * f.expansion(n_before, Array_j)
    B_buf_tile_next = f.expansion(k_next, Array_i) * f.expansion(n_next, Array_j)
    if G_Cw_in_use != G_Cr_in_use:
        C_buf_tile_w = C_buf_tile_w_before
    if G_B_in_use != G_A_in_use:
        B_buf_tile = B_buf_tile_next

    if G_Cw_in_use == 0:
        if bw_in_use == 'C' and G_Cw_in_use == G_Cr_in_use:
            if C_load_in == 1 and C_load_out == 1:
                if G_C1_ready == 0:
                    if G_C1_ptr != Y_buf_tile_trans:
                        G_C1_ptr += bandwidth
                        communication_Y_cycle += 1
                    else:
                        if G_C1_ptw != C_buf_tile_trans:
                            G_C1_ptw += bandwidth
                            communication_C_cycle += 1
                        else:
                            G_C1_ready = 1
                            G_C1_ptr = 0
                            G_C1_ptw = 0
                            bw_in_use = 'B'
            elif C_load_in == 1 and C_load_out == 0:
                if G_C1_ready == 0:
                    if G_C1_ptw != C_buf_tile_trans:
                        G_C1_ptw += bandwidth
                        communication_C_cycle += 1
                    else:
                        G_C1_ready = 1
                        G_C1_ptr = 0
                        G_C1_ptw = 0
                        bw_in_use = 'B'
            else:
                G_C1_ready = 1
                bw_in_use = 'B'

        if DR_list[Array_j-2] == 1:
            G_C0_ptw += Array_j
        if G_C0_ptw == C_buf_tile_w:
            if G_Cw_in_use != G_Cr_in_use:
                if cw_loops != k_loops_before:
                    cw_loops += 1
                    G_C0_ptw = 0
                else:
                    if G_C1_ready == 1:
                        cw_loops = 1
                        G_Cw_in_use = 1
                        G_C0_ptw = 0
                        G_C0_ready = 0
            else:
                if cw_loops != k_loops:
                    cw_loops += 1
                    G_C0_ptw = 0
                else:
                    if G_C1_ready == 1:
                        cw_loops = 1
                        G_Cw_in_use = 1
                        G_C0_ptw = 0
                        G_C0_ready = 0

    else:
        if bw_in_use == 'C' and G_Cw_in_use == G_Cr_in_use:
            if C_load_in == 1 and C_load_out == 1:
                if G_C0_ready == 0:
                    if G_C0_ptr != Y_buf_tile_trans:
                        G_C0_ptr += bandwidth
                        communication_Y_cycle += 1
                    else:
                        if G_C0_ptw != C_buf_tile_trans:
                            G_C0_ptw += bandwidth
                            communication_C_cycle += 1
                        else:
                            G_C0_ready = 1
                            G_C0_ptr = 0
                            G_C0_ptw = 0
                            bw_in_use = 'B'
            elif C_load_in == 1 and C_load_out == 0:
                if G_C0_ready == 0:
                    if G_C0_ptw != C_buf_tile_trans:
                        G_C0_ptw += bandwidth
                        communication_C_cycle += 1
                    else:
                        G_C0_ready = 1
                        G_C0_ptr = 0
                        G_C0_ptw = 0
                        bw_in_use = 'B'
            else:
                G_C0_ready = 1
                bw_in_use = 'B'

        if DR_list[Array_j-2] == 1:
            G_C1_ptw += Array_j
        if G_C1_ptw == C_buf_tile_w:
            if G_Cw_in_use != G_Cr_in_use:
                if cw_loops != k_loops_before:
                    cw_loops += 1
                    G_C1_ptw = 0
                else:
                    if G_C0_ready == 1:
                        cw_loops = 1
                        G_Cw_in_use = 0
                        G_C1_ptw = 0
                        G_C1_ready = 0
            else:
                if cw_loops != k_loops:
                    cw_loops += 1
                    G_C1_ptw = 0
                else:
                    if G_C0_ready == 1:
                        cw_loops = 1
                        G_Cw_in_use = 0
                        G_C1_ptw = 0
                        G_C1_ready = 0

    for i in range(1,Array_j-1):
        DR_list[Array_j-1-i] = DR_list[Array_j-2-i]
    DR_list[0] = SA_list[-1]


    if G_A_in_use == 0:
        if bw_in_use == 'A':
            if A_load == 1:
                if G_A1_ready == 0:
                    G_A1_ptw += bandwidth
                    communication_A_cycle += 1
                    if G_A1_ptw == A_buf_tile_trans:
                        G_A1_ready = 1
                        G_A1_ptw = 0
                        bw_in_use = 'C'
            else:
                bw_in_use = 'C'
                G_A1_ready = 1

        if G_A0_ptr == 0 and a_loops == 1:
            if B_SA_load >= 1:
                B_SA_load -= 1
                G_A0_ptr += Array_i
                if a_loops - n_loops * (k_loops - 1) > 0:
                    communication_on_chip += k + Array_i - f.expansion(k, Array_i)
                    communication_interval += k + Array_i - f.expansion(k, Array_i) - 1
                    communication_interval += (k + Array_i - f.expansion(k, Array_i)) * (Array_j - 1)
                else:
                    communication_on_chip += Array_i
                    communication_interval += Array_i - 1
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

        elif G_A0_ptr%(m*Array_i)==0 and a_loops != n_loops*k_loops:
            if a_loops % n_loops == 0:
                if B_SA_load >= 1:
                    B_SA_load -= 1
                    G_A0_ptr += Array_i
                    if a_loops - n_loops * (k_loops - 1) > 0:
                        communication_on_chip += k + Array_i - f.expansion(k, Array_i)
                        communication_interval += k + Array_i - f.expansion(k, Array_i) - 1
                        communication_interval += (k + Array_i - f.expansion(k, Array_i)) * (Array_j - 1)
                    else:
                        communication_on_chip += Array_i
                        communication_interval += Array_i - 1
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
            else:
                if B_SA_load >= 1:
                    B_SA_load -= 1
                    G_A0_ptr -= (m - 1) * Array_i
                    if a_loops - n_loops * (k_loops - 1) > 0:
                        communication_on_chip += k + Array_i - f.expansion(k, Array_i)
                        communication_interval += k + Array_i - f.expansion(k, Array_i) - 1
                        communication_interval += (k + Array_i - f.expansion(k, Array_i)) * (Array_j - 1)
                    else:
                        communication_on_chip += Array_i
                        communication_interval += Array_i - 1
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
        else:
            if G_A0_ptr != A_buf_tile:
                G_A0_ptr += Array_i
                if a_loops - n_loops * (k_loops - 1) > 0:
                    communication_on_chip += k + Array_i - f.expansion(k, Array_i)
                    communication_interval += k + Array_i - f.expansion(k, Array_i) - 1
                    communication_interval += (k + Array_i - f.expansion(k, Array_i)) * (Array_j - 1)
                else:
                    communication_on_chip += Array_i
                    communication_interval += Array_i - 1
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
            if G_A0_ptr == A_buf_tile and a_loops == n_loops*k_loops:
                if G_A1_ready == 1 and G_C1_ready == 1:
                    G_A_in_use = 1
                    G_A0_ptr = 0
                    G_A0_ready = 0
                    a_loops = 1
                    on_chip_alter = 1


    else:
        if bw_in_use == 'A':
            if A_load == 1:
                if G_A0_ready == 0:
                    G_A0_ptw += bandwidth
                    communication_A_cycle += 1
                    if G_A0_ptw == A_buf_tile_trans:
                        G_A0_ready = 1
                        G_A0_ptw = 0
                        bw_in_use = 'C'
            else:
                bw_in_use = 'C'
                G_A0_ready = 1

        if G_A1_ptr == 0 and a_loops == 1:
            if B_SA_load >= 1:
                B_SA_load -= 1
                G_A1_ptr += Array_i
                if a_loops - n_loops * (k_loops - 1) > 0:
                    communication_on_chip += k + Array_i - f.expansion(k, Array_i)
                    communication_interval += k + Array_i - f.expansion(k, Array_i) - 1
                    communication_interval += (k + Array_i - f.expansion(k, Array_i)) * (Array_j - 1)
                else:
                    communication_on_chip += Array_i
                    communication_interval += Array_i - 1
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

        elif G_A1_ptr%(m*Array_i)==0 and a_loops != n_loops*k_loops:
            if a_loops % n_loops == 0:
                if B_SA_load >= 1:
                    B_SA_load -= 1
                    G_A1_ptr += Array_i
                    if a_loops - n_loops * (k_loops - 1) > 0:
                        communication_on_chip += k + Array_i - f.expansion(k, Array_i)
                        communication_interval += k + Array_i - f.expansion(k, Array_i) - 1
                        communication_interval += (k + Array_i - f.expansion(k, Array_i)) * (Array_j - 1)
                    else:
                        communication_on_chip += Array_i
                        communication_interval += Array_i - 1
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
            else:
                if B_SA_load >= 1:
                    B_SA_load -= 1
                    G_A1_ptr -= (m - 1) * Array_i
                    if a_loops - n_loops * (k_loops - 1) > 0:
                        communication_on_chip += k + Array_i - f.expansion(k, Array_i)
                        communication_interval += k + Array_i - f.expansion(k, Array_i) - 1
                        communication_interval += (k + Array_i - f.expansion(k, Array_i)) * (Array_j - 1)
                    else:
                        communication_on_chip += Array_i
                        communication_interval += Array_i - 1
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
        else:
            if G_A1_ptr != A_buf_tile:
                G_A1_ptr += Array_i
                if a_loops - n_loops * (k_loops - 1) > 0:
                    communication_on_chip += k + Array_i - f.expansion(k, Array_i)
                    communication_interval += k + Array_i - f.expansion(k, Array_i) - 1
                    communication_interval += (k + Array_i - f.expansion(k, Array_i)) * (Array_j - 1)
                else:
                    communication_on_chip += Array_i
                    communication_interval += Array_i - 1
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
            if G_A1_ptr == A_buf_tile and a_loops == n_loops*k_loops:
                if G_A0_ready == 1 and G_C0_ready == 1:
                    G_A_in_use = 0
                    G_A1_ptr = 0
                    G_A1_ready = 0
                    a_loops = 1
                    on_chip_alter = 1

    if G_B_in_use == 0:
        if bw_in_use == 'B':
            if B_load == 1:
                if G_B1_ready == 0:
                    G_B1_ptw += bandwidth
                    communication_B_cycle += 1
                    if G_B1_ptw == B_buf_tile_trans:
                        G_B1_ready = 1
                        G_B1_ptw = 0
                        bw_in_use = 'A'
                        B_load = 0
            else:
                bw_in_use = 'A'
                G_B1_ready = 1

        if A_interval_work == Array_i:
            G_B_ptr_add += Array_i
        if A_interval_spare == Array_i:
            G_B_ptr_add += Array_i
        if G_B_ptr_add > 0 and G_B0_ptr != B_buf_tile:
            G_B0_ptr += Array_j
            if b_loops % n_loops == 0 and b_loops != n_loops * k_loops:
                communication_on_chip += n + Array_j - f.expansion(n, Array_j)
                communication_interval += (Array_i - 1) * (n + Array_j - f.expansion(n, Array_j))
            elif b_loops - (k_loops - 1) * n_loops > 0 and b_loops != n_loops * k_loops:
                communication_on_chip += Array_j * (k + Array_i - f.expansion(k, Array_i)) / Array_i
                communication_interval += (Array_i - 1) * (Array_j * (k + Array_i - f.expansion(k, Array_i)) / Array_i)

            elif b_loops == n_loops * k_loops:
                communication_on_chip += (n + Array_j - f.expansion(n, Array_j)) * (k + Array_i - f.expansion(k, Array_i)) / Array_i

                communication_interval += (Array_i - 1) * ((n + Array_j - f.expansion(n, Array_j)) * (k + Array_i - f.expansion(k, Array_i)) / Array_i)
            else:
                communication_on_chip += Array_j
                communication_interval += (Array_i - 1) * Array_j
            G_B_ptr_add -= 1
            if G_B0_ptr % (Array_i*Array_j) == 0:
                B_SA_load += 1
                b_loops += 1
        A_interval_work -= 1
        A_interval_spare -= 1
        if G_B0_ptr == B_buf_tile:
            if G_B1_ready == 1 and G_A_in_use == 0:
                G_B_in_use = 1
                G_B0_ptr = 0
                G_B0_ready = 0
                b_loops = 1

    else:
        if bw_in_use == 'B':
            if B_load == 1:
                if G_B0_ready == 0:
                    G_B0_ptw += bandwidth
                    communication_B_cycle += 1
                    if G_B0_ptw == B_buf_tile_trans:
                        G_B0_ready = 1
                        G_B0_ptw = 0
                        bw_in_use = 'A'
                        B_load = 0
            else:
                bw_in_use = 'A'
                G_B0_ready = 1

        if A_interval_work == Array_i:
            G_B_ptr_add += Array_i
        if A_interval_spare == Array_i:
            G_B_ptr_add += Array_i
        if G_B_ptr_add > 0 and G_B1_ptr != B_buf_tile:
            G_B1_ptr += Array_j
            if b_loops % n_loops == 0 and b_loops != n_loops * k_loops:
                communication_on_chip += n + Array_j - f.expansion(n, Array_j)
                communication_interval += (Array_i - 1) * (n + Array_j - f.expansion(n, Array_j))
            elif b_loops - (k_loops - 1) * n_loops > 0 and b_loops != n_loops * k_loops:
                communication_on_chip += Array_j * (k + Array_i - f.expansion(k, Array_i)) / Array_i
                communication_interval += (Array_i - 1) * (Array_j * (k + Array_i - f.expansion(k, Array_i)) / Array_i)

            elif b_loops == n_loops * k_loops:
                communication_on_chip += (n + Array_j - f.expansion(n, Array_j)) * (k + Array_i - f.expansion(k, Array_i)) / Array_i

                communication_interval += (Array_i - 1) * ((n + Array_j - f.expansion(n, Array_j)) * (k + Array_i - f.expansion(k, Array_i)) / Array_i)
            else:
                communication_on_chip += Array_j
                communication_interval += (Array_i - 1) * Array_j
            G_B_ptr_add -= 1
            if G_B1_ptr % (Array_i*Array_j) == 0:
                B_SA_load += 1
                b_loops += 1
        A_interval_work -= 1
        A_interval_spare -= 1
        if G_B1_ptr == B_buf_tile:
            if G_B0_ready == 1 and G_A_in_use == 1:
                G_B_in_use = 0
                G_B1_ptr = 0
                G_B1_ready = 0
                b_loops = 1

    for i in range(1,Array_i*compute_pipeline):
        SA_list[Array_i*compute_pipeline-i] = SA_list[Array_i*compute_pipeline-i-1]
    SA_list[0] = new_compute

    if G_Cr_in_use == 0:

        if new_compute == 1:
            G_C0_ptr += Array_j
            if a_loops % n_loops == 0:
                communication_on_chip += 2 * (n + Array_j - f.expansion(n, Array_j))
                communication_interval += 2 * (n + Array_j - f.expansion(n, Array_j) - 1)
                communication_interval += (n + Array_j - f.expansion(n, Array_j)) * (Array_i - 1)
            else:
                communication_on_chip += 2 * Array_j
                communication_interval += 2 * (Array_j - 1)
                communication_interval += Array_j * (Array_i - 1)
        if G_C0_ptr == C_buf_tile_r:
            if cr_loops != k_loops:
                cr_loops += 1
                G_C0_ptr = 0
            else:
                if G_C1_ready == 1 and G_A_in_use == 1:
                    G_Cr_in_use = 1
                    cr_loops = 1
                    G_C0_ptr = 0
                    G_C0_ready = 0
    else:
        if new_compute == 1:
            G_C1_ptr += Array_j
            if a_loops % n_loops == 0:
                communication_on_chip += 2 * (n + Array_j - f.expansion(n, Array_j))
                communication_interval += 2 * (n + Array_j - f.expansion(n, Array_j) - 1)
                communication_interval += (n + Array_j - f.expansion(n, Array_j)) * (Array_i - 1)
            else:
                communication_on_chip += 2 * Array_j
                communication_interval += 2 * (Array_j - 1)
                communication_interval += Array_j * (Array_i - 1)
        if G_C1_ptr == C_buf_tile_r:
            if cr_loops != k_loops:
                cr_loops += 1
                G_C1_ptr = 0
            else:
                if G_C0_ready == 1 and G_A_in_use == 0:
                    G_Cr_in_use = 0
                    cr_loops = 1
                    G_C1_ptr = 0
                    G_C1_ready = 0

    return G_A_in_use, G_A0_ready, G_A1_ready, G_A0_ptr, G_A0_ptw, G_A1_ptr, G_A1_ptw, a_loops, b_loops,\
            G_B0_ready, G_B1_ready, G_B0_ptr, G_B0_ptw, G_B1_ptr, G_B1_ptw, G_B_in_use,\
            B_SA_load, A_interval_work, A_interval_spare, G_C0_ready, G_C0_ptw, G_C0_ptr,\
            G_C1_ready, G_C1_ptw, G_C1_ptr, G_Cw_in_use, G_Cr_in_use, cw_loops, cr_loops,\
            on_chip_alter, bw_in_use, communication_A_cycle, communication_B_cycle,\
            communication_C_cycle, communication_Y_cycle, B_load, G_B_ptr_add, communication_on_chip,\
            communication_interval, communication_internal

def write_back( write_back_flag,
                m,
                n,
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
                communication_Y_cycle
                ):
    C_buf_tile_w = m * f.expansion(n, Array_j)
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
            if G_C0_ptw == C_buf_tile_w:
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
            if G_C1_ptw == C_buf_tile_w:
                G_C1_ptw = 0
        else:
            G_C1_ptr += bandwidth
            communication_Y_cycle += 1
            if G_C1_ptr == Y_buf_tile_trans:
                G_C1_ptr = 0
                write_back_flag = 2
    return write_back_flag, G_Cr_in_use, G_C0_ptr, G_C0_ptw, G_C1_ptr, G_C1_ptw, communication_Y_cycle
