
"""
Normal Distribution
Author: Balamurali M
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


alpha = 0.05

mu, sigma = 0, 1
samples = np.random.normal(mu, sigma, 1000)


w_log, p_stat = stats.shapiro(samples)
print("\nw:", w_log, "\np_stat:", p_stat)
if p_stat > alpha:
    print("Sample looks Gaussian (fail to reject H0)")
else:
    print("Sample does not look Gaussian (reject H0)")


np.random.seed(12345678)
x = stats.norm.rvs(loc=5, scale=3, size=100)


w_log, p_stat_log = stats.shapiro(x)
print("\nw:", w_log, "\np_stat:", p_stat_log)
if p_stat_log > alpha:
    print("Sample looks Gaussian (fail to reject H0)")
else:
    print("Sample does not look Gaussian (reject H0)")

stat, p = stats.shapiro(x)
print("Statistics=%.3f, p=%.3f" % (stat, p))
# interpret
if p > alpha:
    print("Sample looks Gaussian (fail to reject H0)")
else:
    print("Sample does not look Gaussian (reject H0)")
