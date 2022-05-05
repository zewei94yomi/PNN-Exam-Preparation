import numpy as np
from utils import add_list, sub_list, my_argmax_plus1
from sklearn.neighbors import KNeighborsClassifier


def dichotomizer(W=[2, 1],
                 w0=-5,
                 x=[3, 3]):
    """
    Tutorial 2.1
    g(x) = w^t * x + w_0
    判断：
        - g(x) > 0, class 1
        - otherwise, class 2
    If g(x) > 0, Class 1; otherwise Class 2
    """
    gx = np.dot(W, x) + w0
    cls = 1 if gx > 0 else 2
    print(f"g(x) = w^t * x + w_0 = {gx}, Class {cls}")
    return gx


def dichotomizer_augmented_vectors(at=[-3, 1, 2, 2, 2, 4],
                                   x=[1, 1, 1, 1, 1, 1],
                                   print_res=True):
    """
    Tutorial 2.2, 2.5
    g(x) = a^t * y
        where y = (1, x)
    Augmented vectors: 本质上和第一种是一样的，无非是调换了一下参数的位置
    判断：
        - g(x) > 0, class 1
        - otherwise, class 2
    """
    gx = np.dot(at, x)
    cls = 1 if gx > 0 else 2
    if print_res:
        print(f"g(x) = at * y = {gx}, Class {cls}")
    return gx


def two_d_quadratic_discrimination(x=[[1], [1]],
                                   A=[[-2, 5], [5, -8]],
                                   b=[[1], [2]], c=-3):
    """
    Tutorial 2.4
    g(x) = xt * A * x + xt * b + c
    """
    xt = np.transpose(x)
    gx = np.matmul(xt, A)
    gx = np.dot(gx, x) + np.dot(xt, b) + c
    gx = np.sum(gx)
    print(f"g(x) = xt * A * x + xt * b + c = {gx}")
    return gx


def perceptron_augmented_notion(at=[1, 0, 0],
                                Y=[[1, 0, 2], [1, 1, 2], [1, 2, 1], [-1, 3, -1], [-1, 2, 1], [-1, 3, 2]],
                                lr=1,
                                learn_type="Sequential"):
    """
    Tutorial 2.6, 2.7, 2.9（和提供的解法略有不同，但是原理是一样的，只不过使用的）, 2.10
    Augmented notion + Sample normalization
    Batch
    Sequential
    Sequential判断：
        - g(x) > 0  -> no mismatch (no update)
        - g(x) <= 0 -> mismatch (update)
    """
    has_mismatch = True
    epoch = 0
    iteration = 0
    print(f"Initial: at = {at}")
    while has_mismatch:
        epoch += 1
        has_mismatch = False
        total_change = [0 for i in range(len(at))]
        for y in Y:
            iteration += 1
            gx = dichotomizer_augmented_vectors(at, y, print_res=False)
            if gx <= 0:
                has_mismatch = True
                if learn_type == "Batch":
                    add_list(total_change, lr * y)
                else:
                    add_list(at, lr * y)
            if learn_type == "Sequential":
                print(f"Iteration {iteration}, \t y^t = {y}, \t g(x) = {gx}, \t at = {at}")
        if learn_type == "Batch":
            add_list(at, total_change)
            print(f"Epoch {epoch}, \t sum of delta = {total_change}, \t at = {at}")
    print(f"Converged: at = {at}")
    return at


def multiclass_sequential_perceptron_augmented_notion(
        Yt=[[1, 1, 1], [1, 2, 0], [1, 0, 2], [1, -1, 1], [1, -1, -1]],
        w=[1, 1, 2, 2, 3],
        At=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        lr=1):
    """
    Tutorial 2.11
    Augmented notation: yt第一位全都为1
    w: target, 代表class类别: class 1 -> 1, class 2 -> 2, class 3 -> 3
    预测的class: j = argmax(a^t_j * y_k)
    判断：
        if j != w_k:
            a_wk = a_wk + lr * y_k  # 真实的class
            a_j = a_j - lr * y_k    # 预测出的错误class
    """
    need_update = True
    iteration = 0
    print(f"Initial: At = {At}")
    while need_update:
        need_update = False
        for i in range(len(Yt)):
            iteration += 1
            g = []
            for at in At:
                g.append(np.dot(Yt[i], at))
            g_max = my_argmax_plus1(g)
            if g_max != w[i]:
                need_update = True
                add_list(At[w[i] - 1], lr * Yt[i])
                sub_list(At[g_max - 1], lr * Yt[i])
            print(f"Iteration {iteration}, \t g = {g}, \t w = {w[i]}, \t At = {At}")
    print(f"Converged: At = {At}")
    return At


def sequential_widrow_hoff_iter(at=[1, 0, 0],
                                bt=[1, 1, 1, 1, 1, 1],
                                Yt=[[1, 0, 2], [1, 1, 2], [1, 2, 1], [-1, 3, -1], [-1, 2, 1], [-1, 3, 2]],
                                lr=0.1,
                                iterations=12):
    """
    Tutorial 2.14
    * Widrow_Hoff learning: a = a + lr * (bk - at * yk) * yk
      Perceptron learning:  a = a + lr * yk
    :param bt: margin, 有多少个数据，就有多少个margin
    """
    print(f"Initial: At = {at}")
    iteration = 0
    while iteration < iterations:
        i = iteration % len(Yt)
        aty = np.dot(at, Yt[i])
        if bt[i] != aty:
            delta = np.multiply(lr * (bt[i] - aty), Yt[i])
            add_list(at, delta)
        iteration += 1
        print(
            f"Iteration {iteration}, \t yt = {Yt[i]}, \t at * y = {np.round_(aty, decimals=4)}, "
            f"\t\t at = {np.round_(at, decimals=4)}")
    print(f"After {iterations} iterations: at = {np.round_(at, decimals=4)}")
    return at


def sequential_widrow_hoff_epoch(at=[1, 0, 0],
                                bt=[1, 1, 1, 1, 1, 1],
                                Yt=[[1, 0, 2], [1, 1, 2], [1, 2, 1], [-1, 3, -1], [-1, 2, 1], [-1, 3, 2]],
                                lr=0.1,
                                epochs=2):
    """
    Tutorial 2.14
    """
    print(f"Initial: At = {at}")
    epoch = 0
    while epoch < epochs:
        for i in range(len(Yt)):
            aty = np.dot(at, Yt[i])
            if bt[i] != aty:
                delta = np.multiply(lr * (bt[i] - aty), Yt[i])
                add_list(at, delta)
        epoch += 1
        print(f"Epoch {epoch}, \t at = {np.round_(at, decimals=4)}")
    print(f"After {epochs} epochs: at = {np.round_(at, decimals=4)}")
    return at


def knn(X=[[0.15, 0.35], [0.15, 0.28], [0.12, 0.2], [0.1, 0.32], [0.06, 0.25]],
        y=[1, 2, 2, 3, 3],
        k=3,
        x=[[0.1, 0.25]]):
    """
    Tutorial 2.15
    Predict the K-nearest neighbor, not distances to all neighbors
    class start from `1`, `2`, `3` to ...
    """
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X, y)
    cls = neigh.predict(x)
    print(f"K neighbors = {k}, point {x} is class = {cls}")
    return cls


if __name__ == '__main__':
    # Linear discriminant
    # dichotomizer()
    
    # Linear discriminant: Augmented vectors
    # dichotomizer_augmented_vectors()
    
    # 2d feature quadratic discrimination function
    # two_d_quadratic_discrimination()
    
    # Perceptron Learning
    # perceptron_augmented_notion()
    
    # Multiclass sequential perceptron learning
    # multiclass_sequential_perceptron_augmented_notion()
    
    # Sequential Widrow-Hoff
    # sequential_widrow_hoff_iter()
    sequential_widrow_hoff_epoch()
    
    # KNN
    # knn()
