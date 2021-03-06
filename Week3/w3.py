import numpy as np
from commons.utils import has_change, add_list, Heaviside


def neuron_Heaviside(w=[0.1, -0.5, 0.4],
                     threshold=0,
                     X=[0.1, -0.5, 0.4]):
    """
    Tutorial 3.2
    Neuron has a transfer function:
        1. linear weighted sum of its inputs
        2. activation function: Heaviside function
    """
    wx = np.dot(w, X)
    res = wx - threshold
    y = Heaviside(res)
    print(f"wx - theta = {res}")
    print(f"y = H(wx - theta) = {y}")


def delta_learning(theta=-2,
                   w=[0.5, 1],
                   lr=1,
                   X=[[1, 0, 2], [1, 2, 1], [1, -3, 1], [1, -2, -1], [1, 0, -1]],
                   t=[1, 1, 0, 0, 0],
                   learn_type="Sequential"):
    """
    Tutorial 3.3('OR': Sequential), 3.4('OR': Batch),
             3.5('AND': Sequential), 3.6(2d data: Sequential)
    It doesn't matter which type of operation is being learning,
    the only important things are input `X` and target `y`.
    
    Sequential learning:
        w = w + lr * (t - y) * xt
            where t is the class (i.e. y)
    
    Batch learning:
        w = w + sum(lr * (t - y)) * xt
            where t is the class (i.e. y), and sum is the sum of all delta in a batch
    """
    
    print(f"{learn_type} learning: ")
    W = [-theta] + w    # Augmented notation
    print(f"Initial: W = {W}, theta = {-W[0]}, w = {W[1:]}")
    need_update = True
    epoch = 0
    iteration = 0
    while need_update:
        epoch += 1
        need_update = False
        total_delta = [0 for i in range(len(W))]
        for i in range(len(X)):
            iteration += 1
            res = np.dot(W, X[i])
            y = Heaviside(res)
            ty = t[i] - y
            delta = np.multiply(lr * ty, X[i])
            if learn_type == "Sequential":
                if ty != 0:
                    need_update = True
                # add_list(W, delta)
                W = np.add(W, delta)
                print(f"Iteration {iteration}, \t x = {X[i]}, \t t = {t[i]}, \t y = H(wx) = {y}, \t t-y = {ty}, "
                      f"\t update = {delta}, \t W = {W}")
            else:
                delta = np.multiply(lr * ty, X[i])
                # add_list(total_delta, delta)
                total_delta = np.add(total_delta, delta)
        if learn_type == "Batch":
            if has_change(total_delta):
                need_update = True
                # add_list(W, total_delta)
                W = np.add(W, total_delta)
            print(f"Epoch {epoch}, \t x = {X[i]}, \t t = {t[i]}, \t y = H(wx) = {y}, \t t-y = {ty}, "
                  f"\t update = {total_delta}, \t W = {W}")
    print(f"Converged: W = {W}, theta = {-W[0]}, w = {W[1:]}")


def negative_feedback_network(X=[1, 1, 0],
                              W=[[1, 1, 0], [1, 1, 1]],
                              alpha=0.25,
                              y=[0, 0],
                              ite=4):
    for i in range(ite):
        W = np.array(W)
        e = X-np.dot(W.T, y)
        We = np.dot(W, e)
        y = np.add(y, alpha*We)
        WY = np.dot(W.T, y)
        print(f"Iteration{i+1}, \t e = {e}, \t We = {We}, \t y = {y}, \t WY = {WY}")
    print(f"final output:{y}")


if __name__ == '__main__':
    # print("Tutorial 3.2")
    # neuron_Heaviside()
    # neuron_Heaviside(X=[0.1, 0.5, 0.4])

    # print("Tutorial 3.3(Sequential), 3.4(Batch), 3.5(Sequential), 3.6(Sequential)")
    # delta_learning(theta=-1, w=[0, 0], lr=1, X=[[1, 0, 2], [1, 1, 2], [1, 2, 1], [1, -3, 1], [1, -2, -1], [1, -3, -2]], t=[1, 1, 1, 0, 0, 0],learn_type='Sequential')
    delta_learning()

    # negative_feedback_network()

