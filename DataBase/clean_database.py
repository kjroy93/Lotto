"""File to clean the database"""

# Standard libraries of Python
import time
from datetime import datetime

# Dependencies
import pandas as pd
from pandas import DataFrame

# Libraries proper of this proyect
from database.scraping import euro_scraping

def days_sum(day,change):
    date = day + change
    return date

def clean_df(df: DataFrame) -> DataFrame:
    df_clean = {
    'draw': "int32",
    'nro1': "int32",
    'nro2': "int32",
    'nro3': "int32",
    'nro4': "int32",
    'nro5': "int32",
    'star_1': "int32",
    'star_2': "int32"
    }
    df = df.astype(df_clean)
    return df

def structure(data: DataFrame) -> DataFrame:
    # Droping the dates column because of code that generate dates below
    data.drop('dates',axis=1,inplace=True)

    # Reestucture of database
    data = data.reindex(
        columns=['draw', 'nro1', 'nro2', 'nro3', 'nro4', 'nro5', 'star_1', 'star_2']
        )
    
    # Adding the total of draws in the corresponding column
    data['draw'] = range(1,len(data)+1)

    # Manual fix of a draw, because of error in the webpage of euro millions
    update_dict = {
        'nro1': 3,
        'nro2': 31,
        'nro3': 41,
        'nro4': 48,
        'nro5': 50,
        'star_1': 8,
        'star_2': 11
        }
    
    # Locate and fix of the draw with the dictionary
    data.loc[1088,update_dict.keys()] = update_dict.values()

    # Function call to clean database, with the proper format per type of value
    data = clean_df(data)

    # Reordering the first five numbers in ascending order.
    data.iloc[:,1:6] = data.iloc[:,1:6].apply(lambda x: pd.Series(sorted(x)), axis=1)
    data.iloc[:,6:8] = data.iloc[:,6:8].apply(lambda x: pd.Series(sorted(x)), axis=1)

    # Reference for math. First Friday of February 2004
    entry_point = time.mktime((2004,2,6,0,0,0,0,0,0))

    # Reference for math with four days of difference
    tuesday = time.mktime((2004,2,10,0,0,0,0,0,0)) 

    # First draws of Euro Millions
    first_draw = time.mktime((2004,2,13,0,0,0,0,0,0))

    # Last draw of the old rule: one draw per week
    change_draw = time.mktime((2011,5,6,0,0,0,0,0,0)) 

    next_week = first_draw - entry_point
    next_tuesday = tuesday - entry_point
    tuesday_friday = first_draw - tuesday

    dates = []
    dates_1 = []

    for i in range(378):
        entry_point = days_sum(entry_point,next_week)
        dates.append(entry_point)

    for i in range(379,len(data)+1):
        if i % 2:
            change_draw = days_sum(change_draw,next_tuesday)
        else:
            change_draw = days_sum(change_draw,tuesday_friday)
        dates_1.append(change_draw)

    df = pd.DataFrame(list(map(datetime.fromtimestamp,dates)))
    df_1 = pd.DataFrame(list(map(datetime.fromtimestamp,dates_1)))
    
    result = pd.concat([df,df_1],ignore_index=True) \
        .rename(columns = {0: 'dates'})
    dataframe = pd.concat([result, data],axis=1,join='inner')
    dataframe['dates'] = dataframe['dates'].dt.floor('d')

    answer = int(input("Do you wish to save a .parquet file in order to do test without the scraping? (Enter 1 for yes, 0 for no): "))

    if answer == 1:
        # Save the .parquet file
        dataframe.to_parquet('database/db.parquet')
        # Perform further testing without scraping
    else:
        # Proceed with scraping and testing without saving the file
        pass
    
    return dataframe

def database() -> DataFrame:
    database = euro_scraping()
    database = structure(database)
    return database