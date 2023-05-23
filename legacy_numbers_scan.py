"""This is the main code to analize the entire game until current date. This is to observe if there is some difference to pick numbers for the tickets with this method, or selecting them random.
At this point, it is not doing tickets, just picking numbers
"""

# Standard Libraries of Python
import sys
import time

# Dependencies
import pandas as pd
import numpy as np

# Libraries proper of this Proyect
from database.clean_database import database
from data_analisys.legacy_data_functions import draw_generator, numbers_boolean, first_df_bool
from data_analisys.numbers_analisys import analisys
from collections import Counter

"""Constants of the main code for numbers analisys"""

# Main Database to calculate the first DataFrame main_df_counts
db = database()
db_slice = db
lenght = len(db)

# Array of numbers
total_numbers = np.arange(1,51)

# Create a template DataFrame with all values set to False
skip_winners_bool = pd.DataFrame(False, columns=[str(i) for i in range(1, 51)], index=range(len(db)))
    
# Fill in the True values
main_boolean_df = numbers_boolean(db, skip_winners_bool, total_numbers)

# Main DataFrame to be updated with new draws in function analisys
main_df_counts = first_df_bool(db, total_numbers)

start_time = time.time()

succes = []
failure = []
for draw in draw_generator(lenght):
    db_resultados = db_slice.head(draw)
    recommended_numbers, not_recommended_numbers, main_counts = analisys(db_resultados, main_boolean_df, main_df_counts)
    main_df_counts = main_counts
    row = db.loc[draw,['Nro1','Nro2','Nro3','Nro4','Nro5']]
    column = recommended_numbers.loc[:, 'Numbers']
    result = column.isin(row).sum()
    succes.append(result)
    failure.append(5-result)
    sys.stdout.write(f"\ri = {draw}")
    sys.stdout.flush()
    
end_time = time.time()
print(f"Tiempo de ejecución: {end_time - start_time:.2f} segundos")

succes

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

# Print quantity of hits per draw, with recommended numbers, and random numbers aside
for i in range(6):
    hits = (succes.count(i)/len(succes))*100
    print(f"{i} aciertos: {c[i]}\n{round(hits,2)}%")

for e in range(6):
    nohits = (random_succes.count(e)/len(random_succes))*100
    print(f"{e} aciertos: {y[e]}\n{round(nohits,2)}%")