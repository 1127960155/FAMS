# encoding: utf-8
from __future__ import division
from __future__ import print_function
# functions
import numpy as np
import img2col_coordinate as ic

def roundup(X,x):
    temp = X // x
    reminder = X % x
    if reminder != 0:
        temp += 1
    return temp

def ceil(X, bw):
    quotient = X // bw
    remainder = X % bw
    if remainder != 0:
        quotient += 1
    result = quotient * bw
    return result

def expansion(A, Array):
    A_quotient = A // Array
    A_remainder = A % Array
    if A_remainder != 0:
        A = (A_quotient + 1) * Array
    return A

def next_num_kmn(m_serial, k_serial, n_serial, m_loop_times, k_loop_times, n_loop_times):
    if n_serial != n_loop_times-1:
        n_serial += 1
    elif n_serial == n_loop_times-1 and m_serial != m_loop_times-1:
        m_serial += 1
        n_serial = 0
    elif m_serial == m_loop_times-1 and n_serial == n_loop_times-1 and k_serial != k_loop_times-1:
        k_serial += 1
        n_serial = 0
        m_serial = 0
    else:
        pass
    return m_serial, k_serial, n_serial

def before_num_kmn(m_serial, k_serial, n_serial, m_loop_times, k_loop_times, n_loop_times):
    if n_serial != 0:
        n_serial -= 1
    elif n_serial == 0 and m_serial != 0:
        m_serial -= 1
        n_serial = n_loop_times-1
    elif m_serial == 0 and n_serial == 0 and k_serial != 0:
        k_serial -= 1
        n_serial = n_loop_times-1
        m_serial = m_loop_times-1
    else:
        pass
    return m_serial, k_serial, n_serial

def next_num_knm(m_serial, k_serial, n_serial, m_loop_times, k_loop_times, n_loop_times):
    if m_serial != m_loop_times-1:
        m_serial += 1
    elif m_serial == m_loop_times-1 and n_serial != n_loop_times-1:
        n_serial += 1
        m_serial = 0
    elif m_serial == m_loop_times-1 and n_serial == n_loop_times-1 and k_serial != k_loop_times-1:
        k_serial += 1
        n_serial = 0
        m_serial = 0
    else:
        pass
    return m_serial, k_serial, n_serial

def before_num_knm(m_serial, k_serial, n_serial, m_loop_times, k_loop_times, n_loop_times):
    if m_serial != 0:
        m_serial -= 1
    elif m_serial == 0 and n_serial != 0:
        n_serial -= 1
        m_serial = m_loop_times-1
    elif m_serial == 0 and n_serial == 0 and k_serial != 0:
        k_serial -= 1
        n_serial = n_loop_times-1
        m_serial = m_loop_times-1
    else:
        pass
    return m_serial, k_serial, n_serial

def next_num_mkn(m_serial, k_serial, n_serial, m_loop_times, k_loop_times, n_loop_times):
    if n_serial != n_loop_times-1:
        n_serial += 1
    elif n_serial == n_loop_times-1 and k_serial != k_loop_times-1:
        k_serial += 1
        n_serial = 0
    elif n_serial == n_loop_times-1 and k_serial == k_loop_times-1 and m_serial != m_loop_times-1:
        m_serial += 1
        k_serial = 0
        n_serial = 0
    else:
        pass
    return m_serial, k_serial, n_serial

def before_num_mkn(m_serial, k_serial, n_serial, m_loop_times, k_loop_times, n_loop_times):
    if n_serial != 0:
        n_serial -= 1
    elif n_serial == 0 and k_serial != 0:
        k_serial -= 1
        n_serial = n_loop_times-1
    elif n_serial == 0 and k_serial == 0 and m_serial != 0:
        m_serial -= 1
        k_serial = k_loop_times-1
        n_serial = n_loop_times-1
    else:
        pass
    return m_serial, k_serial, n_serial

def next_num_mnk(m_serial, k_serial, n_serial, m_loop_times, k_loop_times, n_loop_times):
    if k_serial != k_loop_times-1:
        k_serial += 1
    elif k_serial == k_loop_times-1 and n_serial != n_loop_times-1:
        n_serial += 1
        k_serial = 0
    elif k_serial == k_loop_times-1 and n_serial == n_loop_times-1 and m_serial != m_loop_times-1:
        m_serial += 1
        n_serial = 0
        k_serial = 0
    else:
        pass
    return m_serial, k_serial, n_serial

def before_num_mnk(m_serial, k_serial, n_serial, m_loop_times, k_loop_times, n_loop_times):
    if k_serial != 0:
        k_serial -= 1
    elif k_serial == 0 and n_serial != 0:
        n_serial -= 1
        k_serial = k_loop_times-1
    elif k_serial == 0 and n_serial == 0 and m_serial != 0:
        m_serial -= 1
        n_serial = n_loop_times-1
        k_serial = k_loop_times-1
    else:
        pass
    return m_serial, k_serial, n_serial

def next_num_nkm(m_serial, k_serial, n_serial, m_loop_times, k_loop_times, n_loop_times):
    if m_serial != m_loop_times-1:
        m_serial += 1
    elif m_serial == m_loop_times-1 and k_serial != k_loop_times-1:
        k_serial += 1
        m_serial = 0
    elif m_serial == m_loop_times-1 and k_serial == k_loop_times-1 and n_serial != n_loop_times-1:
        n_serial += 1
        k_serial = 0
        m_serial = 0
    else:
        pass
    return m_serial, k_serial, n_serial

def before_num_nkm(m_serial, k_serial, n_serial, m_loop_times, k_loop_times, n_loop_times):
    if m_serial != 0:
        m_serial -= 1
    elif m_serial == 0 and k_serial != 0:
        k_serial -= 1
        m_serial = m_loop_times-1
    elif m_serial == 0 and k_serial == 0 and n_serial != 0:
        n_serial -= 1
        k_serial = k_loop_times-1
        m_serial = m_loop_times-1
    else:
        pass
    return m_serial, k_serial, n_serial

def next_num_nmk(m_serial, k_serial, n_serial, m_loop_times, k_loop_times, n_loop_times):
    if k_serial != k_loop_times-1:
        k_serial += 1
    elif k_serial == k_loop_times-1 and m_serial != m_loop_times-1:
        m_serial += 1
        k_serial = 0
    elif m_serial == m_loop_times-1 and k_serial == k_loop_times-1 and n_serial != n_loop_times-1:
        n_serial += 1
        k_serial = 0
        m_serial = 0
    else:
        pass
    return m_serial, k_serial, n_serial

def before_num_nmk(m_serial, k_serial, n_serial, m_loop_times, k_loop_times, n_loop_times):
    if k_serial != 0:
        k_serial -= 1
    elif k_serial == 0 and m_serial != 0:
        m_serial -= 1
        k_serial = k_loop_times-1
    elif m_serial == 0 and k_serial == 0 and n_serial != 0:
        n_serial -= 1
        k_serial = k_loop_times-1
        m_serial = m_loop_times-1
    else:
        pass
    return m_serial, k_serial, n_serial


def mkn_num(m_serial, k_serial, n_serial, m_loop_times, k_loop_times, n_loop_times, M, K, N, m, k, n):
    if m_serial != m_loop_times - 1:
        m_num = m
    else:
        m_num = M - m * m_serial
    if k_serial != k_loop_times - 1:
        k_num = k
    else:
        k_num = K - k * k_serial
    if n_serial != n_loop_times - 1:
        n_num = n
    else:
        n_num = N - n * n_serial
    return m_num, k_num, n_num

def transmission_ifmap(FH, FW, OX, OY, IX, IY, stride, m_coordinate, k_coordinate, m_num, k_num, m, k, ifmap_img2col0_new, ifmap_img2col1_new, ifmap_img2col0_old, ifmap_img2col1_old, C_channel, G_A_in_use, Batchsize, dataflow_M, dataflow_K):
    # 比较片上是否有可重复使用的数据
    transmission_on_chip = 0
    transmission_local = 0
    transmission_zero = 0
    if G_A_in_use == 1:
        for m0 in range(m_coordinate, m_coordinate+m):
            for k0 in range(k_coordinate, k_coordinate+k):
                zero_padding = 0
                local_reuse = 0
                neighbor_reuse = 0
                flag = True
                # 计算目标传输子矩阵原地址
                # 在小分块取数范围内过程的计算
                if m0-m_coordinate < m_num and k0-k_coordinate < k_num:
                    ifmap_img2col0_new[m0-m_coordinate][k0-k_coordinate] = ic.img2col_coordinates(m0,k0,OY,OX,stride,FH,FW,Batchsize,C_channel,dataflow_M,dataflow_K)
                    # 计算源地址是否为ifmap中same模式下补的零
                    if ifmap_img2col0_new[m0-m_coordinate][k0-k_coordinate][0]>(IY-1) or ifmap_img2col0_new[m0-m_coordinate][k0-k_coordinate][1]>(IX-1) or ifmap_img2col0_new[m0-m_coordinate][k0-k_coordinate][2]>(C_channel-1):
                        ifmap_img2col0_new[m0-m_coordinate][k0-k_coordinate] = (-1, -1, -1, -1)
                        zero_padding = 1
                    # 计算在片上的缓存中是否有数据重复，可以不需从片外搬移
                    else:
                        for m00 in range(m):
                            for k00 in range(k):
                                # 在本buf中已存在所需数据
                                if ifmap_img2col0_new[m0-m_coordinate][k0-k_coordinate] == ifmap_img2col0_old[m00][k00]:
                                    local_reuse = 1
                                    flag = False
                                    break
                                # 在此次传输中有重复数据
                                if ifmap_img2col0_new[m0-m_coordinate][k0-k_coordinate] == ifmap_img2col0_new[m00][k00]:
                                    if not(m00==m0-m_coordinate and k00==k0-k_coordinate):
                                        local_reuse = 1
                                        flag = False
                                        break
                                # 在工作buf中存在所需数据
                                if ifmap_img2col0_new[m0-m_coordinate][k0-k_coordinate] == ifmap_img2col1_old[m00][k00]:
                                    neighbor_reuse = 1
                            if not flag:
                                break
                # 不需要取数的小分块地址记为-1
                # 超出此次小分块取数范围，也就是不用取数，统统设为-1
                else:
                    ifmap_img2col0_new[m0-m_coordinate][k0-k_coordinate] = (-1, -1, -1, -1)
                # 记录该数据的复用情况
                if zero_padding == 1:
                    transmission_zero += 1
                elif local_reuse == 1:
                    transmission_local += 1
                elif neighbor_reuse == 1:
                    transmission_on_chip += 1

    else:
        for m0 in range(m_coordinate, m_coordinate+m):
            for k0 in range(k_coordinate, k_coordinate+k):
                zero_padding = 0
                local_reuse = 0
                neighbor_reuse = 0
                flag = True
                # 计算目标传输子矩阵原地址
                if m0-m_coordinate < m_num and k0-k_coordinate < k_num:
                    ifmap_img2col1_new[m0-m_coordinate][k0-k_coordinate] = ic.img2col_coordinates(m0,k0,OY,OX,stride,FH,FW,Batchsize,C_channel,dataflow_M,dataflow_K)
                    # 计算源地址是否为ifmap中same模式下补的零
                    if ifmap_img2col1_new[m0-m_coordinate][k0-k_coordinate][0]>(IY-1) or ifmap_img2col1_new[m0-m_coordinate][k0-k_coordinate][1]>(IX-1) or ifmap_img2col1_new[m0-m_coordinate][k0-k_coordinate][2]>(C_channel-1):
                        ifmap_img2col1_new[m0-m_coordinate][k0-k_coordinate] = (-1, -1, -1, -1)
                        zero_padding = 1
                    # 计算在片上的缓存中是否有数据重复，可以不需从片外搬移
                    else:
                        for m00 in range(m):
                            for k00 in range(k):
                                # 在本buf中已存在所需数据
                                if ifmap_img2col1_new[m0-m_coordinate][k0-k_coordinate] == ifmap_img2col1_old[m00][k00]:
                                    local_reuse = 1
                                    flag = False
                                    break
                                # 在此次传输中有重复数据
                                if ifmap_img2col1_new[m0-m_coordinate][k0-k_coordinate] == ifmap_img2col1_new[m00][k00]:
                                    if not(m00==m0-m_coordinate and k00==k0-k_coordinate):
                                        local_reuse = 1
                                        flag = False
                                        break
                                # 在工作buf中存在所需数据
                                if ifmap_img2col1_new[m0-m_coordinate][k0-k_coordinate] == ifmap_img2col0_old[m00][k00]:
                                    neighbor_reuse = 1
                            if not flag:
                                break
                # 不需要取数的小分块地址记为-1
                else:
                    ifmap_img2col1_new[m0-m_coordinate][k0-k_coordinate] = (-1, -1, -1, -1)
                # 记录该数据的复用情况
                if zero_padding == 1:
                    transmission_zero += 1
                elif local_reuse == 1:
                    transmission_local += 1
                elif neighbor_reuse == 1:
                    transmission_on_chip += 1

    return ifmap_img2col0_new, ifmap_img2col1_new, transmission_zero, transmission_local, transmission_on_chip
