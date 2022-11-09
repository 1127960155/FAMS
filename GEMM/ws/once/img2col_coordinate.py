# encoding: utf-8
from __future__ import division
from __future__ import print_function
import sys
# (A, B, C, D) == (IY, IX, C_channel, Batchsize)
# 'Batch_OX_OY', 'C_FH_FW'后者在循环的内层
def img2col_coordinates(m0,k0,OY,OX,stride,FH,FW,Batchsize,C_channel,dataflow_M,dataflow_K):
    if dataflow_M == 'Batch_OY_OX':
        A_base = ((m0%(OX*OY))//OX)*stride
        B_base = stride*(m0%OX)
        D = m0//(OX*OY)
    elif dataflow_M == 'Batch_OX_OY':
        A_base = (m0%OY)*stride
        B_base = (m0%(OX*OY)//OY)*stride
        D = m0//(OX*OY)
    elif dataflow_M == 'OY_OX_Batch':
        A_base = ((m0//Batchsize)//OX)*stride
        B_base = stride*((m0//Batchsize)%OX)
        D = m0%Batchsize
    elif dataflow_M == 'OX_OY_Batch':
        A_base = ((m0//Batchsize)%OY)*stride
        B_base = stride*((m0//Batchsize)//OY)
        D = m0%Batchsize
    elif dataflow_M == 'OY_Batch_OX':
        A_base = (m0//(Batchsize*OX))*stride
        B_base = (m0%OX)*stride
        D = (m0//OX)%Batchsize
    elif dataflow_M == 'OX_Batch_OY':
        A_base = (m0%OY)*stride
        B_base = (m0//(OY*Batchsize))*stride
        D = (m0//OY)%Batchsize
    else:
        print('img2col transfer error!')
        sys.exit()

    if dataflow_K == 'C_FH_FW':
        A_bias = (k0%(FH*FW))//FW
        B_bias = k0%FW
        C = k0//(FH*FW)
    elif dataflow_K == 'C_FW_FH':
        A_bias = k0%FH
        B_bias = (k0%(FH*FW))//FH
        C = k0//(FH*FW)
    elif dataflow_K == 'FH_C_FW':
        A_bias = k0//(C_channel*FW)
        B_bias = k0%FW
        C = (k0//FW)%C_channel
    elif dataflow_K == 'FW_C_FH':
        A_bias = k0%FH
        B_bias = k0//(FH*C_channel)
        C = (k0//FH)%C_channel
    elif dataflow_K == 'FH_FW_C':
        A_bias = k0//(FW*C_channel)
        B_bias = (k0//C_channel)%FW
        C = k0%C_channel
    elif dataflow_K == 'FW_FH_C':
        A_bias = (k0//C_channel)%FH
        B_bias = k0//(FH*C_channel)
        C = k0%C_channel
    else:
        print('img2col transfer error!')
        sys.exit()

    A = A_base + A_bias
    B = B_base + B_bias
    return (A, B, C, D)
