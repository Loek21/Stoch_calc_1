import numpy as np
import csv

#  closing 22-02-21
S_0 = 64.46
C_0 = 6.7
R = 0.04 # %
K = 65

# Maturity 17 september 2021, work, holiday, trading days until expiry respectively.
W_days = 161
H_days = 4
T_days = 157

# continuously compounded interest rate
r = np.log(1+R)

def historic_volatility():
    """
    Determines the historical volatility based on daily returns
    :param NONE:
    :return: Historical volatility
    """
    closing = []
    with open('historicdata.csv', newline='') as f:
        file = csv.reader(f, delimiter=',')
        next(file)
        for row in file:
            closing.append(float(row[4]))

    returns_list = []
    for i in range(1, len(closing)):
        returns_list.append(np.log(closing[i]/closing[i-1]))
    hist_vol = np.std(returns_list)*np.sqrt(len(closing))

    return hist_vol
