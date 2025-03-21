"""This file gives you information about the stars. Future selection pending"""

# Libraries made for this Proyect
from .parse import Criteria

def generate_stars_criteria():
	euromillions = Criteria(is_star=True)

	euromillions.apply_transformation(is_star=True)
	euromillions.count_skips(is_star=True)
	euromillions.get_natural_rotations(is_star=True)
	print(euromillions.year_history)

	print(euromillions.aprox_rotation)
	print(euromillions.exact_rotation)

if __name__ == "__main__":
	generate_stars_criteria()