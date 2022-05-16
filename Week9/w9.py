import math
import numpy as np


def calculate_alpha(epsilon):
    return round(0.5 * math.log((1 - epsilon) / epsilon), 4)


def update_weights(initial_weight=0.25,
                   alpha=0.5493,
                   real_class=1,
                   predict_class=1):
    update_weights = initial_weight * math.exp(-alpha * real_class * predict_class)
    return round(update_weights, 4)


def entropy_impurity(w1=6, w2=8):
    if w1 == 0 or w2 == 0:
        print(f"entropy impurity i(n) = 0")
        return 0
    else:
        i = - (w1 / (w1 + w2)) * np.log2(w1 / (w1 + w2)) - (w2 / (w1 + w2)) * np.log2(w2 / (w1 + w2))
        print(f"entropy impurity i(n) = {i}")
        return i


def delta_entropy_impurity(root=[6, 8], left=[6, 0], right=[2, 6]):
    res = entropy_impurity(w1=root[0], w2=root[1]) - \
          (root[0] / (root[0] + root[1])) * entropy_impurity(w1=left[0], w2=left[1]) - \
          (root[1] / (root[0] + root[1])) * entropy_impurity(w1=right[0], w2=right[1])
    print(f"The drop in impurity = {res}")
    return res


# 要再经过regularisation才能最后更新权重
# 即得到的return值再除以总的weight
if __name__ == '__main__':
    # print(calculate_alpha(0.1666))
    # print(update_weights(alpha=0.805, initial_weight=0.5001))
    # entropy_impurity()
    delta_entropy_impurity()