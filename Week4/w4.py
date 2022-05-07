import math
import numpy as np


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def tanh_sigmoid(x):
    """
    AKA symmetric tangent sigmoid (i.e. Tansigmoid) function
    This function is equal to `np.tanh(x)`
    """
    return 2 * sigmoid(2 * x) - 1


def d_tanh_sigmoid(x):
    """derivative of tanh(x)"""
    return 1 - math.pow(tanh_sigmoid(x), 2)


def y(xp=[1, 1], c=[0, 0], sigma=(1/math.sqrt(2))):
    """
    y_i(x_p) = ...
    
    sigma is calculated by using Gaussian function
    """
    xpc = 0
    for i in range(len(xp)):
        xpc += math.pow((xp[i] - c[i]), 2)
    return np.exp(-xpc / (2 * math.pow(sigma, 2)))


def RBF_y():
    """
    给定c 和 sigma, 输入x, 计算y
    
    Tutorial中让我们设计一个RBF来解决`XOR`，使用的pattern是1: ((0,0) -> 0)和4: ((1,1) -> 0)
    考试中猜测可能出现使用pattern2和3，或者解决`AND`或`OR`
        - 有几个pattern/center，就有几个hidden unit
        - 猜测考试中应该不会让计算weights w, 因为计算量巨大，需要使用least squares method
    整个RBF NN的设计和训练分为两部：
        1. choose c and sigma (manually or randomly)
        2. determine weights at the output layer using training dataset and least squares method
    """
    # TODO
    pass


if __name__ == '__main__':
    print(np.tanh(1.5))
    print(tanh_sigmoid(1.5))
