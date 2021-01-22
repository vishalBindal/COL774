import matplotlib
matplotlib.use('Agg')
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

input_dir = sys.argv[1]
output_dir = sys.argv[2]
qn_part = sys.argv[3]

# For printing to file
console_output = []
def printn(str):
    print(str)
    console_output.append(str)
    console_output.append('\n')

def Hessian(x, theta, m):
    '''
    Return the hessian matrix
    H = - sum_{i=1}^{m} (x_j^(i) x_k^(i) e^(- theta.T x^(i)))/(1 + e^(- theta.T x^(i)))^2
    '''
    n = theta.size # no of attributes
    h = np.zeros((n,n))
    for j in range(n):
        for k in range(n):
            for i in range(m):
                ethetaTx = np.exp(-1*np.matmul(theta.T, x[:,i]))
                h[j,k] = h[j,k] - x[j,i]*x[k,i]*(ethetaTx)/ ((1 + ethetaTx)**2)
    return h

def gradient(y, x, theta, m):
    '''
    Return the gradient of log-likelihood wrt theta
    gradient = - sum_{i=1}^{m} (y^(i) - h_theta(x^(i))) x^(i)
    '''
    htheta = 1/(1+np.exp(-1*np.matmul(theta.T, x)))
    diff = y - htheta
    gradient = np.sum( diff * x, axis=1)
    return gradient

def newtons_method(x, y, m):
    '''
    Apply newton's method for logistic regression on input data (x,y) with no of features = m
    Return parameters theta
    '''
    theta = np.zeros(3)
    converged = False
    
    theta_prev = theta
    convergence_criteria = 0.001*np.ones(3)
    t = 0
    while not converged:
        Hessian_inverse = np.linalg.inv(Hessian(x, theta, m))
        grad = gradient(y, x, theta, m)
        # Theta update        
        theta = theta - np.matmul(Hessian_inverse, grad)
        # Check convergence
        if np.all(abs(theta - theta_prev) <= convergence_criteria):
            converged = True
        theta_prev = theta
        t = t + 1

    return theta

def plot_data(x, y, m, theta, savepath, x_orig, u, stddev):
    '''
    Plot the decision boundary for logistic regression
    Save the plot to savepath
    '''
    plt.figure()
    plt.title('Logistic regression using Newton\'s method')
    plt.xlabel('x1')
    plt.ylabel('x2')

    lzero, lone = [], []
    for i in range(m):
        if y[i] == 1:
            # points with label 1
            lone.append([x_orig[0,i], x_orig[1,i]])
        else:
            # points with label 0
            lzero.append([x_orig[0,i], x_orig[1,i]])
    
    zero_labels = np.array(lzero)
    one_labels = np.array(lone)
    # Plot input points
    plt.scatter(zero_labels[:,0], zero_labels[:,1], marker='_', label='y = 0')
    plt.scatter(one_labels[:,0], one_labels[:,1], marker='+', label = 'y = 1')
    plt.legend()

    # Create a meshgrid of sample points
    x1 = np.linspace(1,9,50)
    x2 = np.linspace(1,9,50)
    X1, X2 = np.meshgrid(x1, x2)
    # Find theta.T x for points, after normalising them
    ThetaTX = theta[0] + theta[1]*(X1 - u[0])/stddev[0] + theta[2]*(X2 - u[1])/stddev[1]
    # Plot the decision boundary
    plt.contour(X1, X2, ThetaTX, levels=[0])
    plt.savefig(savepath)

# Read data
x_orig = np.genfromtxt(os.path.join(input_dir, 'logisticX.csv'), delimiter=',')
x_orig = x_orig.T
y = np.genfromtxt(os.path.join(input_dir, 'logisticY.csv'))

# No of examples
m = y.size

# Normalising data
u = np.mean(x_orig, axis=1).reshape((2,1))
var = np.var(x_orig, axis=1).reshape((2,1))
x = (x_orig - u) / np.sqrt(var)

# Introducing intercept term x0
x = np.vstack([np.ones((1, m)), x])

# Learning parameters
theta = newtons_method(x,y,m)

if qn_part == 'a':
    printn('Paramaters obtained (Theta): '+ str(theta))
else:
    plot_data(x, y, m, theta, os.path.join(output_dir, 'plot.png'), x_orig, u, np.sqrt(var))

if len(console_output) > 0:
    console_out = open(os.path.join(output_dir, '3'+qn_part+'-console.txt'), 'w')
    console_out.writelines(console_output)
    console_out.close()