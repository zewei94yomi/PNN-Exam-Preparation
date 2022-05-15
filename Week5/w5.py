import numpy as np


def relu(x=[[1, 0.5, 0.2], [-1, -0.5, -0.2], [0.1, -0.1, 0]]):
    """
    Tutorial 5.4 - ReLU
    f(x_i) = y
    判断：
        - x_i >= 0, y = x_i
        - otherwise, y=0
    """
    x = np.array(x)
    print("relu result:")
    print(np.maximum(0, x))
    return np.maximum(0, x)


def lrelu(x=[[1, 0.5, 0.2], [-1, -0.5, -0.2], [0.1, -0.1, 0]],
          a=0.1):
    """
    Tutorial 5.4 - LReLU
    f(x_i) = y
    判断：
        - x_i >= 0, y = x_i
        - otherwise, y=a*x_i
    """
    
    x = np.array(x)
    print("lrelu result:")
    print(np.where(x > 0, x, x * a))
    return np.where(x > 0, x, x * a)


def tanh(x=[[1, 0.5, 0.2], [-1, -0.5, -0.2], [0.1, -0.1, 0]]):
    """
    Tutorial 5.4 - tanh
    """
    
    x = np.array(x)
    print("tanh result:")
    print(np.tanh(x))
    return np.tanh(x)


def heaviside(x=[[1, 0.5, 0.2], [-1, -0.5, -0.2], [0.1, -0.1, 0]], t=0.1):
    """
    Tutorial 5.4 - heaviside function where each neuron has a threshold
    """
    x = np.array(x)
    x = x - t
    print("heaviside result:")
    print(np.heaviside(x, 0.5))
    return np.heaviside(x, 0.5)


def batch_normalization(beta=0,
                        gamma=1,
                        epsilon=0.1,
                        X=[[[1, 0.5, 0.2], [-1, -0.5, -0.2], [0.1, -0.1, 0]],
                           [[1, -1, 0.1], [0.5, -0.5, -0.1], [0.2, -0.2, 0]],
                           [[0.5, -0.5, -0.1], [0, -0.4, 0], [0.5, 0.5, 0.2]],
                           [[0.2, 1, -0.2], [-1, -0.6, -0.1], [0.1, 0, 0.1]]]):
    """
    Tutorial 5.5 - batch normalization given dataset
    bn(x) = beta + (gamma * (x-mean) / (var+epsilon)^1/2)
    """
    X = np.array(X)
    mean = X.sum(axis=0) / len(X)
    var = ((X - mean) ** 2).sum(axis=0) / len(X)
    print("E(x):")
    print(mean)
    print("Var(x):")
    print(var)
    result = beta + (gamma * (X - mean) / np.sqrt(var + epsilon))
    result = np.around(result, 4)
    for i in range(len(X)):
        print(f"BN(X{i + 1}):")
        print(result[i])
    return result


def calculate_outputdim(input=[11, 15],
                        mask=[3, 3],
                        padding=0,
                        stride=2):
    """
    Tutorial 5.9 - calculate the output width and height
    outputDim = 1 + (inputDim - maskDim + 2 * padding) / stride
    """
    print(f"outputHeight:{1 + (input[0] - mask[0] + 2 * padding) / stride}")
    print(f"outputWidth:{1 + (input[1] - mask[1] + 2 * padding) / stride}")


if __name__ == '__main__':
    # Relu function
    # relu()
    
    # Leaky relu function
    # lrelu()
    
    # Tanh function
    # tanh()
    
    # Heavyside function with threshold
    # heaviside()
    
    # Batch normalization
    batch_normalization()
    
    # Average Pooling with kernel size and stride
    # avg_pooling()
    
    # Calculate output dimensions
    # calculate_outputdim()
    # calculate_outputdim(input=[200, 200],
    #                     mask=[5, 5],
    #                     padding=0,
    #                     stride=1)
    # calculate_outputdim(input=[196, 196],
    #                     mask=[2, 2],
    #                     padding=0,
    #                     stride=2)
    # calculate_outputdim(input=[98, 98],
    #                     mask=[4, 4],
    #                     padding=1,
    #                     stride=2)
