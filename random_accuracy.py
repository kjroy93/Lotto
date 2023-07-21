"""File in order to know the % of random selection"""

# Imports
# Standard Libraries of Python
import sys
from collections import Counter

# Dependencies
import numpy as np

# Libraries made for this Proyect
from src.parse import draw_generator, Criteria
ruta_carpeta = 'data/simulation_result/'

euromillions = Criteria()
size = len(euromillions.scrap)

random_succes = []
random_failure = []
for draw in draw_generator(size):
    db_slice = euromillions.scrap.head(draw)
    db_resultados = db_slice.head(draw)
    row = euromillions.scrap.loc[draw,['nro1','nro2','nro3','nro4','nro5']]
    random_numbers = np.random.choice(range(1,51),size=25,replace=False)
    result = np.isin(row, random_numbers).sum()
    random_succes.append(result)
    random_failure.append(5-result)
    sys.stdout.write(f"\ri = {draw}")
    sys.stdout.flush()

y = Counter(random_succes)
z = Counter(random_failure)

# Print quantity of hits per draw, with recommended numbers, and random numbers aside
for e in range(0,6):
    nohits = (random_succes.count(e)/len(random_succes))*100
    print(f"{e} random hits: {y[e]}\n{round(nohits,2)}%")