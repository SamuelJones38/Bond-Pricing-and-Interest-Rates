# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
    
np.random.seed(100)

def em_vasicek(a, b, sigma, Xzero, T, N, M):
        t, dt = np.linspace(0, T, N+1, retstep=True)
        dt = T / N
        dW = np.sqrt(dt) * np.random.randn(N, M)
        W = np.zeros((N+1, M))
        W[1:, :] = np.cumsum(dW, axis=0)
        Xem = np.zeros_like(W)
        Xem[0, :] = Xzero
        for j in range(1, N+1):
            Xem[j, :] = (Xem[j-1, :] + dt * a *(b - Xem[j-1, :]) + 
                         sigma * (W[j, :] - W[j-1, :]))
        return t, Xem
    
# Implementing the Vasicek model for our estimated Parameters
t, Xem = em_vasicek(0.008226, 0.023628, 0.007377, 0.01703, 1, 2**8, 10000)

# Plot the sample paths
plt.figure(figsize=(12,8))
plt.plot(t, Xem, alpha=0.1)
plt.xlabel("Time")
plt.ylabel("Interest Rate")
plt.title("Vasicek Model Simulation")
plt.show()

# Plot the interest rate as a histogram
plt.hist(Xem[-1,:])
plt.xlabel("Interest Rate")
plt.ylabel("Frequency")
plt.title("Vasicek Estimation of ^TNX Interest Rates")
plt.show()

# Calculating the error of our average responses
b = 0.023628
error_vasicek = b - Xem[-1,:].mean()
print("The average error for the Vasicek model is", error_vasicek)


def em_CIR(a, b, sigma, Xzero, T, N, M):
        t2, dt = np.linspace(0, T, N+1, retstep=True)
        dt = T / N
        dW = np.sqrt(dt) * np.random.randn(N, M)
        W = np.zeros((N+1, M))
        W[1:, :] = np.cumsum(dW, axis=0)
        Xem2 = np.zeros_like(W)
        Xem2[0, :] = Xzero
        for j in range(1, N+1):
            Xem2[j, :] = (Xem2[j-1, :] + dt * a *(b - Xem2[j-1, :]) + 
                         sigma * np.sqrt(Xem2[j-1, :]) * (W[j, :] - W[j-1, :]))
        return t2, Xem2   


# Implementing the CIR model for our estimated Parameters
t2, Xem2 = em_CIR(0.008226, 0.023628, 0.007377, 0.01703, 1, 2**8, 10000)

# Plot the sample paths
plt.figure(figsize=(12,8))
plt.plot(t2, Xem2, alpha=0.1)
plt.xlabel("Time")
plt.ylabel("Interest Rate")
plt.title("CIR Model Simulation")
plt.show()    

# Plot the interest rate as a histogram
plt.hist(Xem2[-1,:])
plt.xlabel("Interest Rate")
plt.ylabel("Frequency")
plt.title("CIR Estimation of ^TNX Interest Rates")
plt.show()

# Calculating the error of our average responses
error_CIR = b - Xem2[-1,:].mean()
print("The average error for the CIR model is", error_CIR)

# Reading in the data for ^TNX
my_csv= Path(r'C:/Users/samjo/OneDrive/Documents/Hand-in/Diss/Data_TNX.csv')

# Setting the interest rate as a decimal rather than a percentage
Data = pd.read_csv(my_csv.resolve())
Close = Data["Close"]/100

# Plot the histogram of the data
plt.hist(Close)
plt.xlabel("Interest Rate")
plt.ylabel("Frequency")
plt.title("^TNX Interest Rates")
plt.show()


# Wasserstein distances for both of the models
from scipy.stats import wasserstein_distance
Was_Vas = wasserstein_distance(Xem[-1,:], Close)
print("The Wasserstein Distances between the Vasicek Simulation and ^TNX is", Was_Vas)

Was_CIR = wasserstein_distance(Xem2[-1,:], Close)
print("The Wasserstein Distances between the CIR Simulation and ^TNX is", Was_CIR)

