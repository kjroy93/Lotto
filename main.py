"""This is the start point of for to make the lottery tickets"""

# Standard Libraries of Python
import itertools

# Dependencies
import pandas as pd

# Libraries made for this Proyect
from src.backend.parse import Tickets
from src.backend import numbers
file_route = "data/files/"

euromillions = numbers.euromillions
lotto = Tickets(euromillions)
lotto.first_number()
lotto.suggested_numbers()

print(lotto._selected_numbers)

tickets = itertools.combinations(lotto._selected_numbers,5)
combinations = []

for combination in tickets:
    combinations.append(combination)

print("these are the generated tickets")
print(combinations)

df = pd.DataFrame(combinations,columns=['1','2','3','4','5'])
df = df.iloc[:,0:6].apply(lambda x: pd.Series(sorted(x)),axis=1)
df.to_csv(f'{file_route}combinations.csv',index=False)