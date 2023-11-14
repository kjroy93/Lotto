"""This file gives you the numbers to be selected"""

# Libraries made for this Proyect
from .parse import Criteria

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