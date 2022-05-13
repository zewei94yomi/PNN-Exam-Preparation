import numpy as np


def DX(x=[1, 2],
       theta=[0.1, 0.2]):
    power = x[0] * theta[0] - x[1] * theta[1] - 2
    res = 1 / (1 + np.exp(-power))
    print(f"D({x}) = {res}")
    return res
    
    
def VDG_real(x=[[1, 2], [3, 4]],
             theta=[0.1, 0.2]):
    num = len(x)
    res = 0
    for i in range(num):
        res += 0.5 * np.log(DX(x=x[i], theta=theta))
    print(f"Real Data, Ex~pdata(x)[logD(x)] = {res}")
    return res


def VDG_fake(x=[[5, 6], [7, 8]],
             theta=[0.1, 0.2]):
    num = len(x)
    res = 0
    for i in range(num):
        res += 0.5 * (np.log(1 - DX(x=x[i], theta=theta)))
    print(f"Fake Data, Ex~pz(z)[ln(1 - D(G(z))] = {res}")
    return res


def VDG(x_real=[[1, 2], [3, 4]],
        x_fake=[[5, 6], [7, 8]],
        theta=[0.1, 0.2]):
    """Tutorial 6.3"""
    res = VDG_real(x=x_real, theta=theta) + VDG_fake(x=x_fake, theta=theta)
    print(f"V(D, G) = {res}")
    return res
    

if __name__ == '__main__':
    VDG()
