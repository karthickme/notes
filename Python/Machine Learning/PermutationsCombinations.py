# Permutations and Combinations in Python

# Question:
# Find the number of permutations and combinations that can be formed from the word HORSE taking two letters at a time.

# Answer

from itertools import combinations
from itertools import permutations 
import numpy as np
import math
arr=np.array(['H','O','R','S','E'])
print(len(list(combinations(arr, 2)) ))
print(len(list(permutations(arr,2) )))  
