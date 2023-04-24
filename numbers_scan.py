# Libraries
import sys
import pandas as pd
import numpy as np
import time
from data_analisys.data_functions import draw_generator
from data_analisys.numbers_analisys import analisys
from database.scrapping import euro_scraping
from database.clean_database import structure
from collections import Counter

db = euro_scraping()
db = structure(db)
db_slice = db
lenght = len(db)

start_time = time.time()

succes = []
failure = []
for draw in draw_generator(lenght):
    db_resultados = db_slice.head(draw)
    recommended_numbers, not_recommended_numbers = analisys(db_resultados)
    row = db.loc[draw,['Nro1','Nro2','Nro3','Nro4','Nro5']]
    column = recommended_numbers.loc[:, 'Numbers']
    result = column.isin(row).sum()
    succes.append(result)
    failure.append(5-result)
    sys.stdout.write(f"\ri = {draw}")
    sys.stdout.flush()
    
end_time = time.time()
print(f"Tiempo de ejecución: {end_time - start_time:.2f} segundos")

start_time = time.time()

random_succes = []
random_failure = []
for draw in draw_generator(lenght):
    db_resultados = db_slice.head(draw)
    row = db.loc[draw,['Nro1','Nro2','Nro3','Nro4','Nro5']]
    random_numbers = np.random.choice(range(1, 51), size=25, replace=False)
    result = np.isin(row, random_numbers).sum()
    random_succes.append(result)
    random_failure.append(5-result)
    sys.stdout.write(f"\ri = {draw}")
    sys.stdout.flush()

end_time = time.time()
print(f"Tiempo de ejecución: {end_time - start_time:.2f} segundos")

c = Counter(succes)
x = Counter(failure)
y = Counter(random_succes)
z = Counter(random_failure)

# Imprimir la cantidad de aciertos para cada número de aciertos posibles
for i in range(6):
    hits = (succes.count(i)/len(succes))*100
    print(f"{i} aciertos: {c[i]}\n{round(hits,2)}%")

for e in range(6):
    nohits = (random_succes.count(e)/len(random_succes))*100
    print(f"{e} aciertos: {y[e]}\n{round(nohits,2)}%")