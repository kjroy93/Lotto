"""File to clean the database"""

# Standard libraries of Python
import time
from datetime import datetime

# Dependencies
import pandas as pd

# Libraries proper of this proyect
from database.scraping import euro_scraping

def days_sum(day, change):
    date = day + change
    return date

def clean_df(df):
    df_clean = {
    'sorteo': "int64",
    'nro1': "int64",
    'nro2': "int64",
    'nro3': "int64",
    'nro4': "int64",
    'nro5': "int64",
    'star_1': "int64",
    'star_2': "int64"
    }
    df = df.astype(df_clean)
    return df

def structure(data):
    data.drop('dates', axis=1, inplace=True)
    data = data.reindex(columns=['sorteo', 'nro1', 'nro2', 'nro3', 'nro4', 'nro5', 'star_1', 'star_2'])
    data['sorteo'] = range(1,len(data)+1)
    update_dict = {'nro1': 3, 'nro2': 31, 'nro3': 41, 'nro4': 48, 'nro5': 50, 'star_1': 8, 'star_2': 11}
    data.loc[1088, update_dict.keys()] = update_dict.values()
    data = clean_df(data)
    data.iloc[:, 1:6] = data.iloc[:, 1:6].apply(lambda x: pd.Series(sorted(x)), axis=1)
    data.iloc[:, 6:8] = data.iloc[:, 6:8].apply(lambda x: pd.Series(sorted(x)), axis=1)

    # reference for math - first Friday of February 2004
    entry_point = time.mktime((2004, 2, 6, 0, 0, 0, 0, 0, 0))

    # reference for math with four days of difference
    tuesday = time.mktime((2004, 2, 10, 0, 0, 0, 0, 0, 0)) 

    # this is the first draws of Euro Millions
    first_draw = time.mktime((2004, 2, 13, 0, 0, 0, 0, 0, 0))

    # this is the Last draw of the old rule: one draw per week
    change_draw = time.mktime((2011, 5, 6, 0, 0, 0, 0, 0, 0)) 

    next_week = first_draw - entry_point
    next_tuesday = tuesday - entry_point
    tuesday_friday = first_draw - tuesday

    dates = []
    dates_1 = []

    for i in range(378):
        entry_point = days_sum(entry_point, next_week)
        dates.append(entry_point)

    for i in range(379,len(data)+1):
        if i % 2:
            change_draw = days_sum(change_draw, next_tuesday)
        else:
            change_draw = days_sum(change_draw, tuesday_friday)
        dates_1.append(change_draw)

    df = pd.DataFrame(list(map(datetime.fromtimestamp, dates)))
    df1 = pd.DataFrame(list(map(datetime.fromtimestamp, dates_1)))
    
    result = pd.concat([df, df1], ignore_index = True) \
        .rename(columns = {0: 'dates'})
    dataframe = pd.concat([result, data], axis = 1, join = 'inner')
    dataframe['dates'] = dataframe['dates'].dt.floor('d')

    return dataframe

def database():
    database = euro_scraping()
    database = structure(database)
    return database