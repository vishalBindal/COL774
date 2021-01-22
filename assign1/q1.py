import matplotlib
matplotlib.use('Agg')
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from time import time
from math import sqrt

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

def J(theta, x, y, m):
    '''
    ------ Parameters -------
    theta : parameters, shape (2,)
    x, y : input features (shape (2,m)) and labels (shape (m,))
    m : no of examples
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
    gradient = (-1/m)*(np.sum( diff * x, axis=1)) 
    return gradient


def linear_regression(x, y, eta = 0.01, epsilon = 1e-12, printInfo= False):
    '''
    ------ Parameters -------
    x, y : input features and labels
    eta: learning rate
    epsilon: convergence criteria in terms of change in J
    printInfo: whether stats are to be printed to console 
    ------- Returns -------
    theta: Parameters learned, shape (2,1)
    Js, theta0s, theta1s: list, values of cost and parameters at each iteration
    '''
    # Parameters
    theta = np.zeros(2)

    # Cost before linear regression
    Jlast = J(theta, x, y, m)

    Js, theta1s, theta0s = [], [], []
    converged = False
    i, start_time = 0, time()
    
    while not converged:
        theta = theta - eta * delJ(theta, x, y, m)
        Jcur = J(theta, x, y, m)
        if abs(Jcur - Jlast) <= epsilon:
            converged = True
        Jlast = Jcur
        
        Js.append(Jcur)
        theta1s.append(theta[1])
        theta0s.append(theta[0])
        
        i = i+1
    
    end_time = time()
    if printInfo:
        printn(f'Learning rate (eta): {eta}')
        printn(f'Stopping criteria: epsilon = {epsilon}')
        printn(f'Iterations: {i}\nTime taken: {round(end_time-start_time,5)}s')
        printn(f'Parameters obtained (Theta):\n{theta}')
    
    return theta, Js, theta0s, theta1s 


def plot_hypothesis(x, y, theta, save_path, x_orig):
    '''
    Plot the hypothesis function
    ------ Parameters -------
    x, y, theta : input features, input labels, parameters
    save_path: path where plot is to be saved
    x_orig: original un-normalised input features, shape (m,)
    '''
    fig, ax = plt.subplots()
    ax.set_title('Density of wine v/s its acidity')
    ax.set_xlabel('Acidity')
    ax.set_ylabel('Density')

    # Plotting 'original' input points
    ax.plot(x_orig, y, '.', color='red', label = 'Wine data')

    # Plotting hypothesis function
    h = theta[1]*x[1,:] + theta[0] 
    ax.plot(x_orig, h, linewidth=1, color='black', label = 'Hypothesis function')

    ax.legend()
    fig.savefig(save_path,dpi=300)


def Jmesh(T0, T1, x, y, m):
    '''
    Cost function similar to J()
    But here theta is passed as a matrix of T0 and T1,
    where theta[i,j] = (T0[i,j], T1[i,j])
    This is to compute the cost at every theta parameter in the 3D mesh
    '''
    cost = 0
    for i in range(m):
        cost = cost + (y[i] - x[1,i]*T1 - T0)**2
    cost = cost / (2*m)
    return cost

def get_sample_mesh(x, y, m):
    '''
    Return 3 matrices T0, T1, J to facilitate creation of 3d mesh
    T0: theta_0 points, shape (150,150)
    T1: theta_1 points, shape (150,150)
    Jz: Jz[i,j] = cost at theta = (T0[i,j], T1[i,j]) 
    '''
    theta1 = np.linspace(-1.1,1.1,150)
    theta0 = np.linspace(-0.1,2.1,150)
    T0, T1 = np.meshgrid(theta0, theta1)
    Jz = Jmesh(T0, T1, x, y, m)
    return T0, T1, Jz

def plot_3dmesh(theta0s, theta1s, Js, x, y, m, save_path):
    '''
    Plot 3d mesh alongwith J values at each iteration
    ------ Parameters -------
    x, y, m : input features, input labels, parameters
    save_path: path where plot is to be saved
    Js, theta0s, theta1s: 
    values of cost and parameters after each iteration, as returned by linear_regression()
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Error function')
    ax.set_xlabel(r'Intercept ($\theta_0$)')
    ax.set_ylabel(r'Slope ($\theta_1$)')

    # Get sample points for mesh
    T0, T1, Jz = get_sample_mesh(x, y, m)
    # Plot mesh
    ax.plot_surface(T0, T1, Jz, color='red', alpha=0.6)
    
    # Plot cost after each iteration
    for i in range(len(Js)):
        ax.scatter(theta0s[i], theta1s[i], Js[i], color='black', marker='.')
        # plt.pause(0.2) # For visualisation by human eye
    # plt.show()
    fig.savefig(save_path, dpi=300)
    # plt.close()


def plot_contour(theta0s, theta1s, Js, eta, x, y, m, save_path):
    '''
    Plot contour lines of the cost, mentioning learning rate eta in the plot title
    Also plot theta after every 10 iterations
    ------ Parameters -------
    eta: learning rate
    Other parameters same as plot_3dmesh()
    '''
    fig, ax = plt.subplots()
    ax.set_title('Contours of error function for learning rate = ' + str(eta))
    ax.set_xlabel(r'Intercept ($\theta_0$)')
    ax.set_ylabel(r'Slope ($\theta_1$)')

    # Plotting contours
    T0, T1, Jz = get_sample_mesh(x, y, m)
    ax.contour(T0, T1, Jz, 10)

    for i in range(len(Js)):
        # Plotting only one point every 10 iterations
        if i % 10 == 0:
            # To visualise contour circles
            # plt.contour(T0, T1, Jz, levels=sorted(Js))
            # plt.contour(T0, T1, Jz, levels=[Js[i]])
            
            # To visualise theta
            ax.scatter(theta0s[i], theta1s[i], color='black', marker='.')

            # plt.pause(0.2) # For visualisation by human eye
            # plt.clf()
    # plt.show()
    fig.savefig(save_path, dpi=300)
    # plt.close()

# Read data
x_orig = np.genfromtxt(os.path.join(input_dir, 'linearX.csv'))
y = np.genfromtxt(os.path.join(input_dir, 'linearY.csv'))

# No of examples
m = x_orig.size

# Normalising data
x = (x_orig- np.mean(x_orig)) / sqrt(np.var(x_orig))

# Introducing intercept term x0
x.reshape((1,m))
x = np.vstack([np.ones((1, m)), x])

### For eta = 0.01
if qn_part == 'a':
    # Linear regression
    theta, Js, theta0s, theta1s = linear_regression(x, y, printInfo=True)

elif qn_part == 'b':
    theta, Js, theta0s, theta1s = linear_regression(x, y)
    # Plot hypothesis function
    plot_hypothesis(x, y, theta, os.path.join(output_dir, 'hypothesis.png'), x_orig)

elif qn_part == 'c':
    theta, Js, theta0s, theta1s = linear_regression(x, y)
    # Plot 3D mesh
    plot_3dmesh(theta0s, theta1s, Js, x, y, m, os.path.join(output_dir, '3dmesh.png'))

elif qn_part == 'd':
    theta, Js, theta0s, theta1s = linear_regression(x, y)
    # Plot contour plot
    plot_contour(theta0s, theta1s, Js, 0.01, x, y, m, os.path.join(output_dir, 'contour_normal.png'))

else:
    ## For eta = 0.001
    theta, Js, theta0s, theta1s = linear_regression(x, y, eta=0.001)
    plot_contour(theta0s, theta1s, Js, 0.001, x, y, m, os.path.join(output_dir, 'contour_eta1.png'))
    ## For eta = 0.025
    theta, Js, theta0s, theta1s = linear_regression(x, y, eta=0.025)
    plot_contour(theta0s, theta1s, Js, 0.025, x, y, m, os.path.join(output_dir, 'contour_eta2.png'))
    ## For eta = 0.1
    theta, Js, theta0s, theta1s = linear_regression(x, y, eta=0.1)
    plot_contour(theta0s, theta1s, Js, 0.1, x, y, m, os.path.join(output_dir, 'contour_eta3.png'))

if len(console_output) > 0:
    console_out = open(os.path.join(output_dir, '1'+qn_part+'-console.txt'), 'w')
    console_out.writelines(console_output)
    console_out.close()