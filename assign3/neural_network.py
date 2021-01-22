import numpy as np
import os
from math import inf
import sys

def sigmoid(z):
    return 1/(1+np.exp(-1*np.clip(z, -500, 500)))

def relu(z):
    return z*(z>0)

def train_nn(x, y, r, M, hidden_layers_arch, xi, adaptive, use_relu, eta=0.001):
    """
    Parameters
    ----------
    x, y: train data and labels (encoded)
    r: no of output classes
    M: mini-batch size
    hidden_layers_arch: array containing no of units in each layer e.g. [100,50]
    xi: convergence criteria: converge if absolute difference of loss <= xi between consecutive epochs
    adaptive: whether to use an adaptive learning rate
    use_relu: whether to use relu in hidden layers
    eta: learning rate (if adaptive=False)
    """
    m, n = x.shape
    layers_arch = [n] + hidden_layers_arch + [r]
    num_layers = len(layers_arch)
    
    # initialising thetas
    thetas = []
    for l in range(num_layers - 2):
        # thetas[l][j][k] is connection between unit j in layer l+1 and unit k in layer l
        if use_relu:
            # initialise weights via He initialisation
            thetas.append(np.random.randn(layers_arch[l+1] + 1, layers_arch[l] + 1) * np.sqrt(2/layers_arch[l])) # +1 for the intercept term
        else:    
            # initialise weights via xavier initialisation
            thetas.append(np.random.randn(layers_arch[l+1] + 1, layers_arch[l] + 1) * np.sqrt(1/layers_arch[l])) # +1 for the intercept term
        # initialise biases to zero
        thetas[-1][:,0] = 0
    # initialise weights via xavier initialisation
    thetas.append(np.random.normal(0, np.sqrt(1/layers_arch[-2]), (r, layers_arch[-2] + 1)))
    thetas[-1][:,0] = 0

    delJs = [None for i in range(num_layers - 1)] # gradient of J corresponding to each parameter

    t = 0
    epochs = 1
    converged = False
    J_avg_last, J_avg = -inf, 0
    while not converged and epochs < 101:
        for b in range(m//M): # batch b
            s = b*M
            e = (b+1)*M
            
            # --> compute values of all units
            # input layer
            units = []
            units.append(np.hstack((np.ones((e-s,1)), x[s:e,:])))
            # hidden layers
            for l in range(len(hidden_layers_arch)):
                if use_relu:
                    units.append(relu(np.matmul(units[-1], thetas[l].T)))
                else:
                    units.append(sigmoid(np.matmul(units[-1], thetas[l].T)))
                units[-1][:,0] = 1
            # output layer
            units.append(sigmoid(np.matmul(units[-1], thetas[-1].T)))

            # --> cost
            J_avg = J_avg + np.sum((y[s:e,:] - units[-1])**2)/(2*M)

            # --> update gradients
            # for output layer
            delJ_delnetT = (y[s:e,:] - units[-1])*units[-1]*(1 - units[-1]) # will have a shape of (m, r)  
            delJs[-1] = (-1/M)*np.sum(np.matmul(delJ_delnetT[:,:,np.newaxis], units[-2][:,np.newaxis,:]), axis=0) # will have a shape of (r, 1 + no of hidden units in 2nd last layer)
            delJ_delnet_last = delJ_delnetT # has shape (m, r)
            # delJ_delnet for layer j+1 is calculated using delJ_delnet for layer j+2
            # delJs[j] is calculated using delJ_delnet for layer j+1

            # for hidden layers
            j = num_layers - 2
            while j>=1:
                if use_relu:
                    delJ_delnetj = np.matmul(delJ_delnet_last, thetas[j])*np.heaviside(units[j], 0.5)
                else:
                    delJ_delnetj = np.matmul(delJ_delnet_last, thetas[j])*units[j]*(1 - units[j])
                delJs[j-1] = (-1/M)*np.sum(np.matmul(delJ_delnetj[:,:,np.newaxis], units[j-1][:,np.newaxis,:]), axis=0)
                j -= 1
                delJ_delnet_last = delJ_delnetj

            # update parameters
            if adaptive:
                eta = 0.5/np.sqrt(epochs)
            for l in range(len(thetas)):
                thetas[l] -= eta * delJs[l]
      
            t += 1
        
        # Average cost over last epoch
        J_avg = J_avg/(m//M)
        if abs(J_avg - J_avg_last) <= xi:
            converged = True
        J_avg_last = J_avg
        J_avg = 0
        epochs += 1
    return thetas

def predict_nn(x, hidden_layers_arch, thetas, use_relu, label_index_map):
    layer = np.hstack((np.ones((x.shape[0],1)), x))
    # hidden layers
    for l in range(len(hidden_layers_arch)):
        if use_relu:
            layer = relu(np.matmul(layer, thetas[l].T))
        else:
            layer = sigmoid(np.matmul(layer, thetas[l].T))
        layer[:,0] = np.ones(layer.shape[0])
    # output layer
    layer = sigmoid(np.matmul(layer, thetas[-1].T))
    prediction_indices = np.argmax(layer, axis=1) 
    predictions = np.zeros(x.shape[0], dtype='int8')
    for i in range(x.shape[0]):
      predictions[i] = label_index_map[prediction_indices[i]]
    return predictions


def write_predictions(fname, arr):
    np.savetxt(fname, arr, fmt="%d", delimiter="\n")


def run(x_train_path, y_train_path, x_test_path, batch_size, hidden_layer_list, activation):
    x_train = np.load(x_train_path)
    x_test = np.load(x_test_path)
    y_train = np.load(y_train_path)

    x_train = x_train.reshape((x_train.shape[0], 784))/255
    x_test = x_test.reshape((x_test.shape[0], 784))/255

    r = 0
    m = len(y_train)
    label_map = {}
    label_index_map = {}
    for i in range(m):
        if y_train[i] not in label_map:
            label_map[y_train[i]] = r
            label_index_map[r] = y_train[i]
            r += 1

    y_train_encoded = np.zeros((m, r), dtype='int8')
    for i in range(m):
        y_train_encoded[i][label_map[y_train[i]]] = 1

    use_relu = True
    if activation=='softmax' or activation=='sigmoid':
        use_relu = False
    thetas = train_nn(x_train, y_train_encoded, r, batch_size, hidden_layer_list, 1e-3, False, use_relu, 0.1)
    predictions = predict_nn(x_test, hidden_layer_list, thetas, use_relu, label_index_map)
    return predictions


def main():
    x_train = sys.argv[1]
    y_train = sys.argv[2]
    x_test = sys.argv[3]
    output_file = sys.argv[4]
    batch_size = int(sys.argv[5])
    hidden_layer_list = [int(i) for i in sys.argv[6].split()]
    activation = sys.argv[7]

    output = run(x_train, y_train, x_test, batch_size, hidden_layer_list, activation)
    write_predictions(output_file, output)


if __name__ == '__main__':
    main()


