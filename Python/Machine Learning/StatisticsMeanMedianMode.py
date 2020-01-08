from scipy import stats
import numpy as np
import statistics
dic ={"A": 90,"B": 86,"C":70,"D":95,"E":95,"F":95,"G":95}
print("Mean")
print(np.mean(dic.values()))
print("Median")
print(np.median(dic.values()))
print("Mode")
print(statistics.mode(dic.values()))
print("Standard Deviation")
print(np.std(dic.values()))
print("Variance")
print(np.var(dic.values()))
print("range")
print(stats.iqr(dic.values()))