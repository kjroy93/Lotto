"""Main of the proyect, in order to obtain the recommended stars. This data is going to append to each lottery ticket"""

# Standard Libraries of Python
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP, getcontext
getcontext().prec = 5

# Dependencies
import pandas as pd
import numpy as np
np.set_printoptions(precision=5)

# Libraries made for this Proyect
from src.parse import Criteria

euromillions = Criteria(is_star=True)

euromillions.apply_transformation(is_star=True)
euromillions.count_skips(is_star=True)
euromillions.get_natural_rotations(is_star=True)

print(euromillions.aprox_rotation)
print(euromillions.exact_rotation)