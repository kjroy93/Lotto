"""This file gives you the numbers to be selected"""

# Libraries made for this Proyect
from .parse import Criteria

def generate_numbers_criteria():
	euromillions = Criteria()

	euromillions.define_odd_even()
	euromillions.define_low_high()
	euromillions.define_combinations_skips()
	euromillions.groups_info()
	euromillions.apply_transformation()
	euromillions.count_skips()
	euromillions.skips_for_last_12_draws()
	euromillions.get_natural_rotations()
	euromillions.get_numbers_clasification()
	euromillions.draw_skips()
	euromillions.skips_evaluation()

	euromillions.year_criterion()
	euromillions.rotation_criterion()
	euromillions.position_criterion()
	euromillions.group_criterion()
	euromillions.numbers_of_tomorrow()

	print(euromillions.recommended_numbers)
	print(euromillions.not_recommended_numbers)
	print(euromillions.evaluation)

	return euromillions

if __name__ == "__main__":
	generate_numbers_criteria()