import numpy as np


def covariance(zero_mean_X=[[-1., -1., 0], [0, 0, 0], [1., 2., 0], [0, -1., 0]]):
    C = 0
    for x in zero_mean_X:
        x_temp = np.transpose(np.array([x]))
        C += np.multiply(x_temp, x_temp.T)
    C /= len(zero_mean_X)
    return C


def KLT(X=[[1, 2, 1], [2, 3, 1], [3, 5, 1], [2, 2, 1]],
        V = [[0.4719, 0.8817, 0], [-0.8817, 0.4719, 0]]):
    """
    Tutorial 7.4
    Use KLT to project n-dimension data onto the first i principal components
    X: all data points
    V: selected i largest eigenvectors.
    """
    mean = np.mean(X, axis=0)
    print("Mean of X:")
    print(mean)
    zero_mean_X = X - mean
    print("Zero mean data:")
    print(zero_mean_X)
    
    C = covariance(zero_mean_X=zero_mean_X)
    print("Covariance Matrix:")
    print(C)
    # E, V = np.linalg.eig(cm)
    # print(E)
    # print(V)
    Y = []
    V = np.array(V)
    print("Eigenvectors:")
    print(V)
    for i in range(len(zero_mean_X)):
        newY = np.dot(V, zero_mean_X[i].T)
        print(f"y_{i}")
        print(newY)
        Y.append(newY)
    # print(np.array(Y))
    # cm = np.cov(np.array(Y).T, bias=True)
    # print(np.around(cm, 4))
    C2 = covariance(Y)
    print("Covariance matrix of new data:")
    print(C2)
    

def Ojas_learning_rule(X=[[-5, -4], [-2, 0], [0, -1], [0, 1], [3, 2], [4, 2]],
                       lr=0.01,
                       learn_type="Batch",
                       w=[-1, 0],
                       epoch=2):
    """
    Tutorial 7.7
    Apply Oja's learning rule to zero-mean data and find weights to project to
    the first principal component.
    w = w + lr * sum(y * (x_i - yw))

    X: all zero-mean data
    lr: learning rate
    learn_type: choose Batch or Sequential
    w: initial weight vector
    epoch: total epoch
    """

    print(f"Initial: w = {w}")
    for i in range(epoch):
        print(f"Epoch{i+1}:")
        print("----------------------------------------------------------------")
        total_change = [0 for _ in range(len(w))]
        for x in X:
            y = np.dot(w, x)
            weight_change = x - (y * np.array(w))
            final_weight_change = lr * y * weight_change
            if learn_type=='Batch':
                total_change = np.add(total_change, final_weight_change)
                print(
                    f"y=wx:{y}, \t x-yw:{weight_change}, \t lr*y*(x-yw):{final_weight_change}")
            else:
                w = np.add(w, final_weight_change)
                print(
                    f"y=wx:{y}, \t x-yw:{weight_change}, \t lr*y*(x-yw):{final_weight_change}, \t updated weight:{np.around(w, 4)}")
        if learn_type == "Batch":
            w = np.add(w, total_change)
            print(f"total weight change:{np.around(total_change, 4)}, \t updated weight:{np.around(w, 4)}")

def fisher_method_LDA(X=[[1, 2], [2, 1], [3, 3], [6, 5], [7, 8]],
                      y = [1, 1, 1, 2, 2],
                      w = [[-1, 5], [2, -3]]):
    """
    Tutorial 7.10
    Fisher's LDA calculate to maximise the cost function sb/sw.

    X: all data
    y: class labels
    w: two weights waiting to be compared
    """

    list1, list2 = [], []
    for i in range(len(X)):
        if y[i] == 1:
           list1.append(X[i])
        elif y[i] == 2:
           list2.append(X[i])
    mean1 = np.mean(list1, axis=0)
    mean2 = np.mean(list2, axis=0)
    costs = []
    print(f"mean1:{mean1}, \t mean2:{mean2}")
    for i in range(len(w)):
        sb = np.dot(w[i], (mean1 - mean2).T) ** 2
        sw = 0
        for ele in list1:
            sw += np.dot(w[i], (ele-mean1).T) ** 2
        for ele in list2:
            sw += np.dot(w[i], (ele-mean2).T) ** 2
        costs.append(round(sb/sw, 4))
        print(f"w{i+1}:\t sb:{sb}, \t sw:{sw}, Cost J(w):{costs[i]}")

def extreme_learning_machine(X=[[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]],
                             V=[[-0.62, 0.44, -0.91], [-0.81, -0.09, 0.02], [0.74, -0.91, -0.60], [-0.82, -0.92, 0.71], [-0.26, 0.68, 0.15], [0.80, -0.94, -0.83]],
                             w=[0, 0, 0, -1, 0, 0, 2]):
    """
    Tutorial 7.10
    Apply extreme learning machine to calculate the response of the output neuron.
    Y = H[VX] => Z = wY
    X: all data in Augmented Notion. (1, x1, x2)
    V: weights for hidden neuron
    w: weights for output neuron
    """

    vx = np.dot(V, np.array(X).T)
    y = np.heaviside(vx, 0.5)
    tmp = [[1] * len(y[0])]
    new_y = np.concatenate((tmp, y)) # Append a list of 1 to y for augmented notion.
    z = np.dot(w, new_y)
    print(f"VX={vx}, \n Y=H(VX)={y}, \n Z=wY={z}")


def sparse_coding(x=[-0.05, -0.95],
                  Yt=[[1, 0, 0, 0, 1, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0, -1, 0]],
                  Vt=[[0.4, 0.55, 0.5, -0.1, -0.5, 0.9, 0.5, 0.45],
                      [-0.6, -0.45, -0.5, 0.9, -0.5, 0.1, 0.5, 0.55]]):
    """
    Tutorial 7.12
    Calculate the best sparse code for signal x
    ||x-Vt*y||_2

    X: signal x
    Yt: all sparse codes
    Vt: dictionary of weights
    """

    errors = []
    cnt = 0
    for y in Yt:
        cnt += 1
        tmp = np.dot(Vt, np.array(y).T)
        res = x-tmp
        normval = round(np.linalg.norm(res), 3)
        errors.append(normval)

        print(f"Vt*y_{cnt}:{tmp}, \t x-Vt*y_{cnt}:{res}, \t error of y_{cnt}: {normval}")


if __name__ == '__main__':
    # KLT to project n-dimensional data onto the first two principal components
    # KLT(X=[[5, 5, 4.4, 3.2], [6.2, 7.0, 6.3, 5.7], [5.5, 5.0, 5.2, 3.2], [3.1, 6.3, 4.0, 2.5], [6.2, 5.6, 2.3, 6.1]],
    #     V=[[0.58, 0.12, -0.04, 0.81],[0.11, 0.25, 0.96, -0.07]])
    KLT()

    # Apply to Tutorial 7.6
    # KLT(X=[[0, 1], [3, 5], [5, 4], [5, 6], [8, 7], [9, 7]],
    #     V=[-0.8309, -0.5564])

    # Apply Oja's learning rule to find the first principal component.
    # Ojas_learning_rule(X=[[-0.2, -0.78, -0.04, -0.94], [1, 1.22, 1.86, 1.56], [0.3, -0.78, 0.76, -0.94], [-2.1, 0.52, -0.44, -1.64], [1, -0.18, -2.14, 1.96]],
    #                    w=[-0.2, -0.2, 0.2, 0],
    #                    epoch=1,
    #                    learn_type='Sequential')

    # fisher_method_LDA()

    # extreme_learning_machine()

    # sparse_coding()