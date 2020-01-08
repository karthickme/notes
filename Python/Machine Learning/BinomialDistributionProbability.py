# Binomial Example

# Question:
# 80 % of people who purchase pet insurance are women. If 9 pet insurance owners are randomly selected, find the probability that precisely 6 are women.

# Solution:
n=9
p=0.80
k=6
from scipy import stats
probability=stats.binom.pmf(k,n,p)
print(probability)
