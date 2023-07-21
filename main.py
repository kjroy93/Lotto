"""This is the start point of for to make the lottery tickets"""

# Standard Libraries of Python
import itertools

# Libraries made for this Proyect
import numbers_scan
from src.parse import Tickets

euromillions = numbers_scan.euromillions
lotto = Tickets(euromillions)
lotto.draw_skips()
lotto.skips_evaluation()
lotto.first_number()
lotto.suggested_numbers()

print(lotto._selected_numbers)

tickets = itertools.combinations(lotto._selected_numbers,5)

for combination in tickets:
    print(combination)