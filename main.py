"""This is the start point of for to make the lottery tickets"""

# Standard Libraries of Python
import itertools

# Dependencies
import pandas as pd

# Libraries made for this Proyect
from backend.src import lotto_numbers, pick_numbers, pick_stars
from backend.src import stars
file_route = 'database/files/'

euromillions = lotto_numbers.generate_numbers_criteria()
euromillions.df.to_csv(f'{file_route}database.csv', index=False)
lotto = pick_numbers.Selection(euromillions)
# lotto.first_number()
lotto.suggested_numbers()

print(f'selected numbers: {lotto._selected_numbers}')

tickets = itertools.combinations(lotto._selected_numbers,5)
combinations = []

for combination in tickets:
	combinations.append(combination)

picking_low = []
for n, ticket in enumerate(combinations):
	value = filter(lambda x: x <= 25, ticket)
	amount = sum(1 for x in list(value) if x > 0)
	if amount == 2:
		picking_low.append(n)

picking_odd = []
for n, ticket in enumerate(combinations):
	value = filter(lambda x: x % 2 != 0, ticket)
	amount = sum(1 for x in list(value) if x > 0)
	if amount == 2:
		picking_odd.append(n)

buy = []
for ticket in picking_low:
    if ticket in picking_odd:
        buy.append(combinations[ticket])

print("these are the generated tickets")
print(buy)

df = pd.DataFrame(buy,columns=['1','2','3','4','5'])
df = df.iloc[:,0:6].apply(lambda x: pd.Series(sorted(x)),axis=1)
df.to_csv(f'{file_route}combinations.csv',index=False)