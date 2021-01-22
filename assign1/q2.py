import matplotlib
matplotlib.use('Agg')
import sys
import os
import numpy as np
from time import time
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# System arguments
input_dir = sys.argv[1]
output_dir = sys.argv[2]
qn_part = sys.argv[3]

# For printing to file
console_output = []
def printn(str):
    print(str)
    console_output.append(str)
    console_output.append('\n')

def sample_points(m, write_to_file=False):
    '''
    Generate m sample (x, y) points
    with x(i) = (1, x1, x2) where
    x1 is normally distributed as N(3,4)
    x2 is normally distributed as N(-1,4)
    y = 3 + x1 + 2*x2 + epsilon, epsilon distributed as N(0,2)
    '''
    x1 = np.random.normal(3, 4, m)
    x2 = np.random.normal(-1, 4, m)
    x = np.vstack([np.ones(m), x1, x2])

    epsilon = np.random.normal(0, 2, m)
    y = 3 + x1 + 2*x2 + epsilon

    if write_to_file:
        np.savetxt(os.path.join(output_dir, 'x.csv'), x.T)
        np.savetxt(os.path.join(output_dir, 'y.csv'), y.T)
    return x, y


def J(theta, x, y, m):
    '''
    ------ Parameters -------
    theta : parameters, shape (2,)
    x, y : input batch (shape (2,m)) and batch labels (shape (m,))
    m : batch size
    ------ Returns ---------
    cost J = (1/2m) * sum_{i=1}^m (y - theta.T X)^2
    '''
    diff = y - np.matmul(theta.T, x)
    cost = (np.sum(diff**2))/(2*m)
    return cost


def delJ(theta, x, y, m):
    '''
    Parameters same as in J()
    ------ Returns ---------
    gradient del_theta(J) = (-1/m) * sum_{i=1}^m (y - theta.T X)
    '''
    diff = y - np.matmul(theta.T, x)
    gradient = (-1/m)*(np.sum(diff * x, axis=1))
    return gradient


def sgd(x, y, m, r, delta, k):
    '''
    ------Parameters----
    x, y, m: input features, labels, no of examples
    r: batch size
    delta, k: convergence criteria on theta
    -------Returns------
    theta: parameters learned, shape (3,)
    thetas: list of theta sampled once every 10 iterations
    '''
    # Parameters
    theta = np.zeros(3)
    # Learning rate
    eta = 0.001

    # Split data into batches
    xbatch = np.split(x, m//r, axis=1)
    ybatch = np.split(y, m//r)

    epoch = 0
    t = 0
    converged = 0
    start_time = time()
    thetas = []
    # Initialise theta_last to a large value
    theta_last = 100*np.ones(3)

    while converged < k:
        for l in range(int(m/r)):
            theta = theta - eta * delJ(theta, xbatch[l], ybatch[l], r)
            if t % 10 == 0:
                thetas.append(theta)
            t = t + 1
            
            # Checking convergence
            c = True 
            for i in range(3):
                if abs(theta_last[i] - theta[i]) > delta:
                    c = False
                    break
            if c:
                converged = converged + 1
                if converged >= k:
                    break
            else:
                converged = 0

            theta_last = theta    
        
        epoch = epoch + 1
    
    time_taken = time() - start_time
    printn(f'------For no of batches: {r} -------')
    printn(f'Delta: {delta}\tk: {k}')
    printn(f'Iterations: {t}\tEpochs: {epoch}\tTime: {time_taken}s')
    printn(f'Theta: {theta}\t')
    printn(f'-------------------------------------')
    return theta, thetas


def plot_theta(thetas, r):
    '''
    Plot movement of theta in the 3d space
    -------Parameters-------
    thetas: list of theta after every 10 iterations, to be plotted
    r: batch size, to be mentioned in the plot title
    '''
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_title(r'Movement of $\theta$ for batch size '+str(r))
    ax.set_xlabel(r'$\theta_0$')
    ax.set_ylabel(r'$\theta_1$')
    ax.set_zlabel(r'$\theta_2$')

    T = np.array(thetas)
    ax.scatter3D(T[:,0], T[:,1], T[:,2])
    fig.savefig(os.path.join(output_dir, 'theta-'+str(r)+'.png'))

# Generate data
m = 10**6
x, y = sample_points(m, write_to_file=True)

if qn_part != 'a':
    # Train for different batch sizes
    theta4, thetas4 = sgd(x, y, m, 1, 1e-1, 20000)
    
    theta3, thetas3 = sgd(x, y, m, 100, 5*1e-3, 10000)

    theta2, thetas2 = sgd(x, y, m, 10000, 1e-3, 5000)

    theta1, thetas1 = sgd(x, y, m, 1000000, 1e-4, 1)

    if qn_part == 'c':
        # Read test data
        data_test = np.genfromtxt(os.path.join(input_dir, 'q2test.csv'), delimiter=',', skip_header=1)

        x1_test = data_test[:, 0]
        x2_test = data_test[:, 1]

        y_test = data_test[:, 2]
        m_test = y_test.size
        # Adding intercept term
        x_test = np.vstack([np.ones(m_test), x1_test, x2_test])

        # Error wrt original parameters
        printn('\nError with original hypothesis: '+ str(J(np.array([3,1,2]), x_test, y_test, m_test)))

        # Error wrt learned parameters
        printn('\nFor batch size 1:')
        printn('Error with learned hypothesis: ' + str(J(theta4, x_test, y_test, m_test)))

        printn('\nFor batch size 100:')
        printn('Error with learned hypothesis: ' + str(J(theta3, x_test, y_test, m_test)))

        printn('\nFor batch size 10000:')
        printn('Error with learned hypothesis: ' + str(J(theta2, x_test, y_test, m_test)))

        printn('\nFor batch size 1000000:')
        printn('Error with learned hypothesis: ' + str(J(theta1, x_test, y_test, m_test)))

    if qn_part == 'd':
        # Plot movement of theta for different batch sizes
        plot_theta(thetas1, 1000000)
        plot_theta(thetas2, 10000)
        plot_theta(thetas3, 100)
        plot_theta(thetas4, 1)

if len(console_output) > 0:
    console_out = open(os.path.join(output_dir, '2'+qn_part+'-console.txt'), 'w')
    console_out.writelines(console_output)
    console_out.close()