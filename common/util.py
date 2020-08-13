# coding: utf-8
import numpy as np


def shuffle_dataset(x, t) :
    '''

    :param x: 训练数据
    :param t: 监督数据
    :return: x(打乱数据），t（监督数据）
    '''
    permulation = np.random.permutation(x.shape[0])
    x = x[permulation,:] if x.ndim == 2 else x[permulation, :, : ,:]
    t = t[permulation]
    return x, t

def conv_output_size(input_size, kernel_size, stride=1, pad=0):
    return (input_size + 2 * pad - kernel_size) / stride + 1


def im2col(input_data, kernel_h, kernel_w, stride = 1, pad = 0):
    '''
    :param input_data: 由（数据量，通道，高，长）的4维数据构成的输入数据
    :param kernel_h: 卷积核的高
    :param kernel_w: 卷积核的长
    :param stride: 步幅
    :param pad: 填充
    :return: col（二维数据，input_data的展开形式）
    '''
    N, C, H, W = input_data.shape

    # 输出的高和宽
    out_h = (H + 2 * pad - kernel_h) // stride + 1
    out_w = (W + 2 * pad - kernel_w) // stride + 1

    # 增加填充
    img = np.pad(input_data, [(0,0),(0,0),(pad,pad),(pad,pad)], 'constant')

    # 返回的二维数据
    col = np.zeros((N, C, kernel_h, kernel_w, out_h, out_w))

    for x in range(kernel_h):
        x_max = x + stride * out_h
        for y in range(kernel_w):
            y_max = y + stride * out_w
            col[:, :, x, y, :, :] = img[:, :, x:x_max:stride, y:y_max:stride]
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col


def col2im(col, input_shape, kernel_h, kernel_w, stride=1, pad=0):
    '''

    :param col: 输入数据（即展开后的矩形）
    :param intput_shape: 输出数据的形状 (例如：（10，1，2，3）)
    :param kernel_h: 卷积核的高
    :param kernel_w: 卷积核的长
    :param stride: 步幅
    :param pad: 填充
    :return: 返回一个4维数组
    '''
    N, C, H, W = input_shape

    out_h = (H + 2 * pad - kernel_h) // stride + 1;
    out_w = (W + 2 * pad - kernel_w) // stride + 1;

    col = col.reshape(N, out_h, out_w, C, kernel_h, kernel_w).transpose(0, 3, 4, 5 , 1 ,2);
    img = np.zeros((N, C, H + 2 * pad, W + 2 * pad))
    for x in range(kernel_h) :
        x_max = x + stride * out_h
        for y in range(kernel_w):
            y_max = y + stride * out_w
            img[:, :, x:x_max:stride, y:y_max:stride] += col[:, :, x, y, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]