import numpy as np
import matplotlib.pyplot as plt
from hist_vol import historic_volatility
from scipy.stats import norm

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

def put_price_tree(tree, K, T, r, sigma, N):
    """
    Calculates the put option price in binomial tree using backwards induction.
    :param tree: Binomial tree containing stock prices
    :param K: strike price at t=T
    :param T: Time till expiry
    :param r: risk-free interest rate
    :param sigma: historic volatility
    :param N: steps in the tree
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

    # Determines put price in the last row of the binomial tree
    for i in range(col):
        S_T = tree[row-1, i]
        tree[row-1, i] = max(0, K-S_T)

    # Backwards induction
    for i in range(row-1)[::-1]:
        for j in range(i+1):
            price_down = tree[i+1, j]
            price_up = tree[i+1, j+1]
            tree[i,j] = np.exp(-r*dt)*(q*price_up+(1-q)*price_down)

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

if __name__ == "__main__":
    put_price_list = []
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for N in range(1,101):
        stock_tree = binomial_tree(S_0, K, T_years, r, sigma, N)
        put_tree = put_price_tree(stock_tree, K, T_years, r, sigma, N)
        put_price_list.append(put_tree[0,0])

    blackscholes = put_price_BS(K, S_0, 0, T_years, sigma, r)
    blackscholes_list = list(blackscholes for i in range(100))
    x = np.linspace(1,100,100)
    ax.plot(x, put_price_list, label="Option value (Binomial Tree)")
    ax1 = ax.twinx()
    ax1.plot(x, abs((np.array(blackscholes_list) - np.array(put_price_list))/np.array(blackscholes_list))*100, "orange", label="Relative error")
    ax.set_xlabel("Binomial Tree steps N")
    ax.set_ylabel("Put option price $")
    ax.set_title("Convergence of the Binomial Tree method for determining option values \n and its relative error compared to the Black-Scholes model")
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax.transAxes)
    ax1.set_ylabel("Relative error in %")
    fig.savefig("Combined.jpg")
