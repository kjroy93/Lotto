"""Main of the proyect, in order to obtain the recommended numbers. This data is the start point of for to make the lottery tickets"""

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

euromillions = Criteria()

euromillions.groups_info()
euromillions.apply_transformation()
euromillions.count_skips()
euromillions.skips_for_last_12_draws()
euromillions.get_natural_rotations()
euromillions.numbers_clasification()

euromillions.year_criterion()
euromillions.rotation_criterion()
euromillions.position_criterion()
euromillions.group_criterion()
euromillions.numbers_of_tomorrow()

print(euromillions.recommended_numbers)
print(euromillions.not_recommended_numbers)