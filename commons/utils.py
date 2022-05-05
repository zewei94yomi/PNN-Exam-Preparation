def add_list(l1, l2):
    for i in range(len(l1)):
        l1[i] += l2[i]
        
        
def sub_list(l1, l2):
    for i in range(len(l1)):
        l1[i] -= l2[i]
        
        
def my_argmax(gx):
    max_index = 0
    for i in range(len(gx)):
        if gx[i] >= gx[max_index]:
            max_index = i
    return max_index


def my_argmax_plus1(gx):
    return my_argmax(gx) + 1


def has_change(delta):
    for d in delta:
        if d != 0:
            return True
    return False


def Heaviside(ans):
    return 1 if ans > 0 else 0


if __name__ == '__main__':
    print("Test function here...")