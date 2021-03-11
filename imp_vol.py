import matplotlib.pyplot as plt
import numpy as np
from tree import binomial_tree
from tree import put_price_tree
from tqdm import tqdm

#  closing 22-02-21
S_0 = 64.46
C_0 = 6.7
R = 0.0004
K = 65

# Maturity 17 september 2021, work, holiday, trading days until expiry respectively.
W_days = 161
H_days = 4
T_days = 157
T_years = T_days / 255

# continuously compounded interest rate
r = np.log(1 + R)


def fit_sigma(S_0, **kwargs):

    tree = binomial_tree(S_0, **kwargs)

    return put_price_tree(tree, **kwargs)[0, 0]


# Bisection method
condition = False
a = 0.0001
b = 1
m = a + (b - a) / 2
a_list = [a]
b_list = [b]

while not condition:

    f_a = fit_sigma(S_0, K=K, T=T_years, r=R, sigma=a, N=150) - C_0
    f_b = fit_sigma(S_0, K=K, T=T_years, r=R, sigma=b, N=150) - C_0
    f_m = fit_sigma(S_0, K=K, T=T_years, r=R, sigma=m, N=150) - C_0

    if np.sign(f_m) != np.sign(f_a):
        b = m
        b_list.append(b)
        m = a + (b - a) / 2

    if np.sign(f_m) != np.sign(f_b):
        a = m
        a_list.append(a)
        m = a + (b - a) / 2

    if abs(f_m) < 0.0000001:
        condition = True

plt.plot(a_list)
plt.plot(b_list)
plt.plot(len(a_list) + 2, m, "bo", label=f"Implied volatility = {round(m,3)}")
plt.title("Implied volatility by Bisection method", fontsize=25)
# plt.text(len(a_list) - 1, m + 0.1, f"Implied volatility = {round(m, 3)}", fontsize=18)
plt.legend(fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlim(0, 16)
plt.ylim(0, 1)
plt.ylabel("Implied volatility", fontsize=22)
plt.xlabel("Bisection steps", fontsize=22)
plt.show()
