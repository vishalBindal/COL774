import matplotlib
from numpy.lib.npyio import save
matplotlib.use('Agg')
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

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

def getphi(y, m):
    '''
    Return value of phi
    '''
    return np.sum(y)/m


def getu0(x, y, m):
    '''
    Return value of u0
    '''
    num = np.sum((1-y)*x, axis=1)
    den = np.sum(1-y)
    return num/den


def getu1(x, y, m):
    '''
    Return value of u1
    '''
    num = np.sum(y*x, axis=1)
    den = np.sum(y)
    return num/den


def getsigma(x, y, m, u0, u1):
    '''
    Return value of Sigma (when Sigma1 = Sigma2 = Sigma)
    '''
    sigma = np.zeros((2, 2))
    for i in range(m):
        if y[i] == 0:
            sigma = sigma + np.outer(x[:, i] - u0, x[:, i] - u0)
        else:
            sigma = sigma + np.outer(x[:, i] - u1, x[:, i] - u1)
    sigma = sigma / m
    return sigma


def getsigma0(x, y, m, u0):
    '''
    Return value of Sigma0
    '''
    sigma0 = np.zeros((2, 2))
    count_zero = 0
    for i in range(m):
        if y[i] == 0:
            count_zero = count_zero + 1
            sigma0 = sigma0 + np.outer(x[:,i] - u0, x[:,i] - u0)
    sigma0 = sigma0 / count_zero
    return sigma0


def getsigma1(x, y, m, u1):
    '''
    Return value of Sigma1
    '''
    sigma1 = np.zeros((2, 2))
    count_one = 0
    for i in range(m):
        if y[i] == 1:
            count_one = count_one + 1
            sigma1 = sigma1 + np.outer(x[:,i] - u1, x[:,i] - u1)
    sigma1 = sigma1 / count_one
    return sigma1


def plot_points(ax, x, y, savePlot=False):
    '''
    Plot input points (x(i),y(i)) on axes ax
    Save figure to output directory if savePlot is True
    '''
    ax.set_title('Growth ring parameters in salmon')
    ax.set_xlabel('x1 (Fresh water)')
    ax.set_ylabel('x2 (Marine water)')

    lzero, lone = [], []
    for i in range(m):
        if y[i] == 1:
            # points with label 1
            lone.append([x[0, i], x[1, i]])
        else:
            # points with label 0
            lzero.append([x[0, i], x[1, i]])

    zero_labels = np.array(lzero)
    one_labels = np.array(lone)
    ax.scatter(zero_labels[:, 0], zero_labels[:, 1],
               marker='.', label='Alaska')
    ax.scatter(one_labels[:, 0], one_labels[:, 1], marker='1', label='Canada')
    ax.legend()
    if savePlot:
        fig.savefig(os.path.join(output_dir, 'points.png'))


def decision_rhs_linear(x, sigma_inv, u0, u1):
    '''
    Return RHS of the linear decision boundary expression
    i.e. (1/2)(-2 x.T Sigma_1^(-1) mu_1 + 2 x.T Sigma_0^(-1) mu_0 
    + mu_1.T Sigma_1^(-1) mu_1 - mu_0.T Sigma_0^(-1) mu_0)
    '''
    t1 = -2*np.matmul(x.T, np.matmul(sigma_inv, u1))
    t2 = 2*np.matmul(x.T, np.matmul(sigma_inv, u0))
    t3 = np.matmul(u1.T, np.matmul(sigma_inv, u1))
    t4 = -1*np.matmul(u0.T, np.matmul(sigma_inv, u0))
    return (t1 + t2 + t3 + t4)/2


def decision_lhs_linear(phi):
    '''
    Return LHS of the linear decision boundary expression
    i.e. log((phi/(1-phi))
    '''
    return np.log(phi / (1 - phi))


def plot_decision_boundary_linear(fig, ax, phi, u0, u1, sigma, u, stddev, savePlot=False):
    '''
    Plot the linear decision boundary on axes ax
    Save figure fig to output directory if savePlot is True
    '''
    # Obtain a meshgrid of points
    x1 = np.linspace(75, 175, 50)
    x2 = np.linspace(300, 500, 50)
    X1, X2 = np.meshgrid(x1, x2)

    RHS = np.zeros((50, 50))
    # Calculate sigma^-1
    sigma_inv = np.linalg.inv(sigma)
    for i in range(50):
        for j in range(50):
            # Calculate RHS corresponding to each point in the mesh
            # Normalise each point as (x - u)/stddev before plugging in the RHS equation
            RHS[i, j] = decision_rhs_linear(
                np.array([(x1[j] - u[0])/stddev[0], (x2[i] - u[1])/stddev[1]]), sigma_inv, u0, u1)
    # Calculate LHS of the decision boundary
    LHS = decision_lhs_linear(phi)
    # Plot the decision boundary using contour plot
    ax.contour(X1, X2, RHS, levels=[LHS])
    if savePlot:
        fig.savefig(os.path.join(output_dir, 'boundary_linear.png'))


def decision_rhs_quadratic(x, sigma0_inv, sigma1_inv, u0, u1):
    '''
    Return RHS of the quadratic decision boundary expression
    i.e (1/2)(-(x-mu_0).T Sigma_0^(-1) (x-mu_0) + (x-mu_1).T Sigma_1^(-1) (x-mu_1))
    '''
    t1 = -1*np.matmul((x - u0).T, np.matmul(sigma0_inv, x - u0))
    t2 = np.matmul((x - u1).T, np.matmul(sigma1_inv, x - u1))
    return (t1 + t2)/2


def decision_lhs_quadratic(phi, sigma0, sigma1):
    '''
    Return LHS of the quadratic decision boundary expression
    i.e. log((phi/(1-phi)) + (1/2)log(|Sigma_0|/|Sigma_1|)
    '''
    t1 = np.log(phi/(1-phi))
    t2 = (1/2)*np.log(np.linalg.det(sigma0) / np.linalg.det(sigma1))
    return t1 + t2


def plot_decision_boundary_quadratic(fig, ax, phi, u0, u1, sigma0, sigma1, u, stddev, savePlot=False):
    '''
    Plot the quadratic decision boundary on the axes ax
    Save figure fig to output directory if savePlot is True
    '''
    # Get sample points on the axes
    x1 = np.linspace(75, 175, 50)
    x2 = np.linspace(300, 500, 50)

    # Normalise these sample points
    x1n = (x1 - u[0])/stddev[0]
    x2n = (x2 - u[1])/stddev[1]
    
    # Create a meshgrid of the sample points
    X1, X2 = np.meshgrid(x1, x2)

    RHS = np.zeros((50, 50))

    # Calculate inverse of sigma0 and sigma1
    sigma0_inv = np.linalg.inv(sigma0)
    sigma1_inv = np.linalg.inv(sigma1)

    for i in range(50):
        for j in range(50):
            # Calculate RHS corresponding to each point in the mesh, putting normalised points in the RHS equation
            RHS[i, j] = decision_rhs_quadratic(np.array([x1n[j], x2n[i]]), sigma0_inv, sigma1_inv, u0, u1)
    
    # Calculate LHS in the quadratic decision boundary expression
    LHS = decision_lhs_quadratic(phi, sigma0, sigma1)
    
    # Plot the decision boundary using contour
    ax.contour(X1, X2, RHS, levels=[LHS])
    if savePlot:
        fig.savefig(os.path.join(output_dir, 'boundary_quadratic.png'))


# Read data
x_orig = np.genfromtxt(os.path.join(input_dir, 'q4x.dat'))
x_orig = x_orig.T
y_str = np.genfromtxt(os.path.join(input_dir, 'q4y.dat'), dtype='str')

# No of examples
m = y_str.size

# Getting int labels
y = np.zeros(m)
for i in range(m):
    if y_str[i] == 'Canada':
        y[i] = 1

# Normalising data
u = np.mean(x_orig, axis=1).reshape((2, 1))
var = np.var(x_orig, axis=1).reshape((2, 1))
stddev = np.sqrt(var)
x = (x_orig - u) / stddev

# Getting phi, u0, u1, sigma
phi = getphi(y, m)
u0 = getu0(x, y, m)
u1 = getu1(x, y, m)
sigma = getsigma(x, y, m, u0, u1)

if qn_part == 'a':
    printn(f'Phi: {phi}\nu0: {u0}\nu1: {u1}\nSigma:\n{sigma}')

# Plotting points and linear decision boundary
elif qn_part == 'b':
    fig, ax = plt.subplots()
    plot_points(ax, x_orig, y, savePlot=True)

elif qn_part == 'c':
    fig, ax = plt.subplots()
    plot_points(ax, x_orig, y)
    plot_decision_boundary_linear(fig, ax, phi, u0, u1, sigma, u, stddev, savePlot=True)

elif qn_part == 'd':
    # Getting sigma0 and sigma1
    sigma0 = getsigma0(x, y, m, u0)
    sigma1 = getsigma1(x, y, m, u1)
    printn(f'u0: {u0}\nu1: {u1}\nSigma_0:\n{sigma0}\nSigma_1:\n{sigma1}')

elif qn_part == 'e':
    # Getting sigma0 and sigma1
    sigma0 = getsigma0(x, y, m, u0)
    sigma1 = getsigma1(x, y, m, u1)
    # Plotting quadratic decision boundary
    fig, ax = plt.subplots()
    plot_points(ax, x_orig, y)
    plot_decision_boundary_linear(fig, ax, phi, u0, u1, sigma, u, stddev)
    plot_decision_boundary_quadratic(fig, ax, phi, u0, u1, sigma0, sigma1, u, stddev, savePlot=True)

if len(console_output) > 0:
    console_out = open(os.path.join(output_dir, '4'+qn_part+'-console.txt'), 'w')
    console_out.writelines(console_output)
    console_out.close()