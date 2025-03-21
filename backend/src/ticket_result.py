"""File in order to know the % of success and failure of the tickets"""

# Imports
# Standard Libraries of Python
import sys
import time
from collections import Counter

# Dependencies
import numpy as np

# Libraries made for this Proyect
from backend.src.functions import draw_generator
from backend.src.parse import Criteria
from backend.src import pick_numbers

# Test of Euromillions Analysis
start_time = time.time()

success = []
failure = []
results = []
euromillions = Criteria()
size = len(euromillions.scrap)

for draw in draw_generator(size):
	db_slice = euromillions.scrap.head(draw)
	euromillions.db = db_slice
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

	lotto = pick_numbers.Selection(euromillions)
	lotto.first_number()
	lotto.suggested_numbers()

	row = euromillions.scrap.loc[draw,['nro1','nro2','nro3','nro4','nro5']]
	column = lotto._selected_numbers
	
	result = np.isin(column,row).sum()
	success.append(result)
	failure.append(5 - result)

	sys.stdout.write(f"\ri = {draw}")
	sys.stdout.flush()
	
c = Counter(success)
x = Counter(failure)

# Print quantity of hits per draw, with recommended numbers, and random numbers aside
for i in range(0,6):
	hits = (success.count(i)/len(success))*100
	print(f"{i} hits: {c[i]}\n{round(hits,2)}%")