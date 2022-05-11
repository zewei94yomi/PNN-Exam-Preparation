import math
def calculate_alpha(epsilon):
    return round(0.5 * math.log((1-epsilon)/epsilon), 4)

def update_weights(initial_weight = 0.25,
                   alpha = 0.5493,
                   real_class = 1,
                   predict_class = -1):
    update_weights = initial_weight * math.exp(-alpha*real_class*predict_class)
    return round(update_weights, 4)
#要再经过regularisation才能最后更新权重
#即得到的return值再除以总的weight
if __name__ == '__main__':
    print(calculate_alpha(0.3))
    print(update_weights())