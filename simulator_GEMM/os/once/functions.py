# encoding: utf-8
from __future__ import division
from __future__ import print_function
# functions
import numpy as np


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

