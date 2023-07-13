"""Main of the proyect, in order to obtain the recommended stars. This data is going to append to each lottery ticket"""

# Libraries made for this Proyect
from src.parse import Criteria

euromillions = Criteria(is_star=True)

euromillions.apply_transformation(is_star=True)
euromillions.count_skips(is_star=True)
euromillions.get_natural_rotations(is_star=True)

print(euromillions.aprox_rotation)
print(euromillions.exact_rotation)