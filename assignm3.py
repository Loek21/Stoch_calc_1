import numpy as np
import matplotlib.pyplot as plt
from hist_vol import historic_volatility
from scipy.stats import norm
from copy import deepcopy

#  closing 22-02-21
S_0 = 64.46
C_0 = 6.7
R = 0.0004
K = 65

# Maturity 17 september 2021, work, holiday, trading days until expiry respectively.
W_days = 161
H_days = 4
T_days = 157
T_years = T_days/255

# continuously compounded interest rate
r = np.log(1+R)
#r = 0

# historic volatility
sigma = historic_volatility()

def binomial_tree(S_0, K, T, r, sigma, N):
    """
    Builds the Binomial tree for the stock price.
    :param S_0: spot price at t=0
    :param K: strike price at t=T
    :param T: Time till expiry
    :param r: risk-free interest rate
    :param sigma: historic volatility
    :param N: steps in the tree
    :return: Lower-Triangular matrix representation of binomial tree containing stock prices
    """
    dt = T/N
    tree = np.zeros((N+1,N+1))

    u = np.exp(sigma*np.sqrt(dt))
    d = np.exp(-sigma*np.sqrt(dt))
    for i in range(N+1):
        for j in range(i+1):
            tree[i, j] = S_0*((u**j)*(d**(i-j)))
    return tree

def put_price_tree(tree, K, T, r, sigma, N, european):
    """
    Calculates the put option price in binomial tree using backwards induction.
    :param tree: Binomial tree containing stock prices
    :param K: strike price at t=T
    :param T: Time till expiry
    :param r: risk-free interest rate
    :param sigma: historic volatility
    :param N: steps in the tree
    :param type: True if European, False if American
    :returns: Lower-Triangular matrix representation of binomial tree containing put option prices
    """
    dt = T/N

    # up and down
    u = np.exp(sigma*np.sqrt(dt))
    d = np.exp(-sigma*np.sqrt(dt))

    # risk free probability
    q = (np.exp(r*dt)-d)/(u-d)

    # Binomial tree dimensions
    col = tree.shape[1]
    row = tree.shape[0]

    early_exercise_val = 0
    early_exercise_time = 0

    # Determines put price in the last row of the binomial tree
    for i in range(col):
        S_T = tree[row-1, i]
        tree[row-1, i] = max(0, K-S_T)

    # Backwards induction
    for i in range(row-1)[::-1]:
        for j in range(i+1):
            price_down = tree[i+1, j]
            price_up = tree[i+1, j+1]

            # if the option is european, calculate put value according to the usual formula
            if european == True:
                tree[i,j] = np.exp(-r*dt)*(q*price_up+(1-q)*price_down)
            
            # if the option is american, the put option value can also be simply the strike minus the spot (if it's in the money)
            else:
                if (K - tree[i][j] > np.exp(-r*dt)*(q*price_up+(1-q)*price_down)) and K - tree[i][j] > early_exercise_val:
                    early_exercise_val = K - tree[i][j]
                    early_exercise_time = i
                tree[i,j] = max(np.exp(-r*dt)*(q*price_up+(1-q)*price_down), K-tree[i,j])

    if (N == 3) and (early_exercise_val > 0):
        print(f"This option is exercised at step {early_exercise_time} for {early_exercise_val}")

    return tree

def put_price_BS(K, S, t, T, sigma, r):
    """
    Calculates the put option price using Black_Scholes formula
    :param K: strike price at t=T
    :param S: spot price
    :param t: current time
    :param T: time till expiry
    :param sigma: historic volatility
    :param r: risk-free interest rate
    :returns: put option price according to the Black-Scholes formula
    """
    d1 = (np.log(S/K)+(r+0.5*sigma**2)*(T-t))/(sigma*np.sqrt(T-t))
    d2 = d1 - sigma*np.sqrt(T-t)
    price = np.exp(-r*(T-t))*K*norm.cdf(-d2)-S*norm.cdf(-d1)
    
    return price

def binomial_tree_visual(stock_tree, option_tree):
    """
    Takes the stock and option binomial trees and combines them into a single print
    """
    rows = stock_tree.shape[0]
    #print the tree with stock values (option values)
    for i in range(rows):
        for j in range(0,i+1):
            print(f"{round(stock_tree[i][j],2)} ({round(option_tree[i][j],4)})", end=" ")
        print()

if __name__ == "__main__":
    N = 3

    # american version
    stock_tree = binomial_tree(S_0, K, T_years, r, sigma, N)
    put_tree = put_price_tree(deepcopy(stock_tree), K, T_years, r, sigma, N, False)
    blackscholes = put_price_BS(K, S_0, 0, T_years, sigma, r)
    binomial_tree_visual(stock_tree, put_tree)

    # european version
    put_tree = put_price_tree(deepcopy(stock_tree), K, T_years, r, sigma, N, True)
    blackscholes = put_price_BS(K, S_0, 0, T_years, sigma, r)
    binomial_tree_visual(stock_tree, put_tree)

    # get prices for different N and compare to Black-Scholes + plot
    put_price_list = []
    put_price_list_europ = []
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for N in range(1,101):

        # get the american option values
        stock_tree = binomial_tree(S_0, K, T_years, r, sigma, N)
        put_tree = put_price_tree(stock_tree, K, T_years, r, sigma, N, False)
        put_price_list.append(put_tree[0,0])

        # get the european data as well for good measure
        stock_tree = binomial_tree(S_0, K, T_years, r, sigma, N)
        put_tree = put_price_tree(stock_tree, K, T_years, r, sigma, N, True)
        put_price_list_europ.append(put_tree[0,0])

    blackscholes = put_price_BS(K, S_0, 0, T_years, sigma, r)
    blackscholes_list = list(blackscholes for i in range(100))
    x = np.linspace(1,100,100)
    ax.plot(x, put_price_list, label="Option value (American)", lw=0.9)
    ax.plot(x, put_price_list_europ, label="Option value (European)", lw=0.9)
    ax1 = ax.twinx()
    ax1.plot(x, abs((np.array(blackscholes_list) - np.array(put_price_list))/np.array(blackscholes_list))*100, "red", label="Relative error (American)", lw=0.9)
    ax1.plot(x, abs((np.array(blackscholes_list) - np.array(put_price_list_europ))/np.array(blackscholes_list))*100, "purple", label="Relative error (European)", lw=0.9)
    ax.set_xlabel("Binomial Tree steps N")
    ax.set_ylabel("Put option price $")
    ax.set_title("Convergence of the Binomial Tree method for determining option values \n and its relative error compared to the Black-Scholes model")
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax.transAxes)
    ax1.set_ylabel("Relative error in %")
    plt.show()


    # for more detail, plot the difference between american and european options
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, np.array(put_price_list)-np.array(put_price_list_europ), lw=0.9)
    ax.set_xlabel("Binomial Tree steps N")
    ax.set_ylabel("Price difference in $")
    ax.set_title("American and European put option price difference")
    plt.show()

    # for more detail, plot the difference between american and black-scholes
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, np.array(put_price_list)-np.array(blackscholes), lw=0.9)
    ax.set_xlabel("Binomial Tree steps N")
    ax.set_ylabel("Price difference in $")
    ax.set_title("American and European put option price difference")
    plt.show()

    # specifically get the value of early exercise by subtracting black scholes from american option, set N as you like
    N = 50
    stock_tree = binomial_tree(S_0, K, T_years, r, sigma, N)
    european_value = put_price_tree(deepcopy(stock_tree), K, T_years, r, sigma, N, True)[0,0]
    american_value = put_price_tree(deepcopy(stock_tree), K, T_years, r, sigma, N, False)[0,0]
    early_exercise_value = american_value - blackscholes
    diff = american_value - european_value
    print("Early exercise value:", early_exercise_value, american_value, european_value, diff)
