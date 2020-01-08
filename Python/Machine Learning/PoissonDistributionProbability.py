# Poisson - Example

# Question:
# If the number of vehicles that pass through a junction on a busy
#  road is at an average rate of 300 per hour, 
# find the probability that no vehicle passes in a given minute.
# Answer
from scipy import stats

averagepass=300/60
probability=stats.poisson.pmf(0, averagepass)
print(probability)
