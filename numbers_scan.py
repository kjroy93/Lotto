# Standard Libraries of Python
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP, getcontext
getcontext().prec = 5

# Dependencies
import pandas as pd
import numpy as np
np.set_printoptions(precision=5)

# Libraries made for this Proyect
from database.clean_database import database
from data_analisys.new_functions import draw_generator,Analysis,Criteria

euromillions = Analysis()
euromillions.groups_info()
euromillions.apply_transformation()
euromillions.count_skips()
euromillions.skips_for_last_12_draws()
euromillions.get_natural_rotations()
euromillions.numbers_clasification()

numbers_choice = Criteria()
numbers_choice.year_criterion()
numbers_choice.rotation_criterion()
numbers_choice.position_criterion()
numbers_choice.group_criterion()
numbers_choice.numbers_of_tomorrow()