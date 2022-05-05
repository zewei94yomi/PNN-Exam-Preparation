import numpy as np
from utils import add_list, sub_list, my_argmax_plus1
from sklearn.neighbors import KNeighborsClassifier


def dichotomizer(W=[2, 1], w0=-5, x=[3, 3]):
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


def dichotomizer_augmented_vectors(at=[-3, 1, 2, 2, 2, 4], x=[1, 1, 1, 1, 1, 1], print_res=True):
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


def two_d_quadratic_discrimination(x=[[1], [1]], A=[[-2, 5], [5, -8]],
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
    Tutorial 2.6, 2.7, 2.9（迭代步数略有不同，但最终收敛的答案一样）, 2.10（和2.9一模一样）
    Batch Perceptron Learning: 对整个batch的数据更新一次，直至拟合
    Sequential Perceptron Learning: 对每一个mismatch的数据实时更新，其他和Batch一样，直至拟合
    Sequential判断：
        - g(x) > 0  -> no mismatch (no update)
        - g(x) <= 0 -> mismatch (update)
    """
    has_mismatch = True
    epoch = 0
    iteration = 0
    print(f"Initial at = {at}")
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
                print(f"Iteration {iteration}, at = {at}")
        if learn_type == "Batch":
            add_list(at, total_change)
            print(f"Epoch {epoch}, at = {at}")
    print(f"at = {at}")
    return at


def multiclass_sequential_perceptron_augmented_notion(Yt, At, lr):
    """
    Tutorial 2.11
    3 classes
    yt第一位全都为1: Augmented notation
    yt最后一位代表class，代表class类别: class 1 -> 1, class 2 -> 2, class 3 -> 3
    """
    need_update = True
    while need_update:
        need_update = False
        for yt in Yt:
            g = []
            for at in At:
                g.append(np.dot(yt[:-1], at))
            g_max = my_argmax_plus1(g)
            if g_max != yt[-1]:
                need_update = True
                # update at[g_max] and at[yt[-1]]:
                add_list(At[yt[-1] - 1], lr * yt[:-1])
                sub_list(At[g_max - 1], lr * yt[:-1])
    print(f"At = {At}")
    return At


def sequential_widrow_hoff_iter(Yt, at, bt, lr, iterations):
    """
    Tutorial 2.14
    Widrow_Hoff: a = a + lr * (bk - at * yk) * yk
    Perceptron: a = a + lr * yk
    :param at:
    :param bt: margin, 有多少个数据，就有多少个margin
    :param lr:
    :param iterations: the number of iterations
    :return:
    """
    epochs = iterations // len(Yt)
    rest_iter = iterations % len(Yt)
    sequential_widrow_hoff_epoch(Yt, at, bt, lr, epochs)
    if rest_iter != 0:
        for i in range(rest_iter):
            aty = np.dot(at, Yt[i])
            if bt[i] != aty:
                delta = np.multiply(lr * (bt[i] - aty), Yt[i])
                add_list(at, delta)
    return at


def sequential_widrow_hoff_epoch(Yt, at, bt, lr, epochs):
    """
    Tutorial 2.14
    """
    for i in range(epochs):
        for i in range(len(Yt)):
            aty = np.dot(at, Yt[i])
            if bt[i] != aty:
                delta = np.multiply(lr * (bt[i] - aty), Yt[i])
                add_list(at, delta)
    return at


def knn(X, y, k, x):
    """
    Tutorial 2.15
    """
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X, y)
    return neigh.predict(x)


if __name__ == '__main__':
    # Linear discriminant
    # dichotomizer()
    
    # Linear discriminant: Augmented vectors
    # dichotomizer_augmented_vectors()
    
    # 2d feature quadratic discrimination function
    # two_d_quadratic_discrimination()
    
    # Perceptron Learning
    perceptron_augmented_notion()
    
    # Multiclass sequential perceptron learning
    # 3 classes, 2d features
    # yt第一位全都为1: Augmented notation
    # yt最后一位代表class，代表class类别: class 1 -> 1, class 2 -> 2, class 3 -> 3
    # t_Yt = [[1, 1, 1, 1], [1, 2, 0, 1], [1, 0, 2, 2], [1, -1, 1, 2], [1, -1, -1, 3]]
    # t_At = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]    # all initialize to 0
    # t_lr = 1
    # multiclass_sequential_perceptron_augmented_notion(Yt=t_Yt, At=t_At, lr=t_lr)
    
    # Sequential Widrow-Hoff
    # t_Yt = [[1, 0, 2], [1, 1, 2], [1, 2, 1], [-1, 3, -1], [-1, 2, 1], [-1, 3, 2]]
    # t_at = [1, 0, 0]
    # t_bt = [1, 1, 1, 1, 1, 1]
    # t_lr = 0.1
    # t_epochs = 2
    # t_iterations = 12
    # sequential_widrow_hoff_epoch(Yt=t_Yt, at=t_at, bt=t_bt, lr=t_lr, epochs=t_epochs)
    # sequential_widrow_hoff_iter(Yt=t_Yt, at=t_at, bt=t_bt, lr=t_lr, iterations=t_iterations)
    # print(t_at)
    
    # KNN
    # t_X = [[0.15, 0.35], [0.15, 0.28], [0.12, 0.2], [0.1, 0.32], [0.06, 0.25]]
    # t_y = [1, 2, 2, 3, 3]
    # t_k = 3
    # t_x = [[0.1, 0.25]]
    # res = knn(X=t_X, y=t_y, k=t_k, x=t_x)
    # print(f"K neighbors = {t_k}, class = {res}")
