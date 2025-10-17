from classes import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn import datasets
mnist = datasets.load_digits()
X,y = mnist.data, mnist.target.astype(int)


SAMPLES = 500
MEAN1 = np.array([0, 0])
MEAN2 = np.array([5, 5])
MEAN3 = np.array([0, 5])
MEAN4 = np.array([5, 0])

COV = np.array([[0.5, 0], [0, 0.5]])

X1 = multivariate_normal.rvs(mean=MEAN1, cov=COV, size=SAMPLES)
X2 = multivariate_normal.rvs(mean=MEAN2, cov=COV, size=SAMPLES)
X3 = multivariate_normal.rvs(mean=MEAN3, cov=COV, size=SAMPLES)
X4 = multivariate_normal.rvs(mean=MEAN4, cov=COV, size=SAMPLES)

X_class0 = np.vstack((X1, X2))
y_class0 = np.zeros(len(X_class0))

X_class1 = np.vstack((X3, X4))
y_class1 = np.ones(len(X_class1))

X = np.vstack((X_class0, X_class1))
y = np.hstack((y_class0, y_class1))

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.7)
plt.title('XOR Dataset')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True)
plt.show()

