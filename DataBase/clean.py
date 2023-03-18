from datetime import datetime
import time
import gspread
import pandas as pd
import numpy as np
from oauth2client.service_account import ServiceAccountCredentials
scope = ["https://spreadsheets.google.com/feeds",'https://www.googleapis.com/auth/spreadsheets',"https://www.googleapis.com/auth/drive.file","https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("creds.json", scope)
client = gspread.authorize(creds)

def days_sum(day, change):
    date = day + change
    return date

print("Limpiando base de datos de origen, Por favor, espere")

sh = client.open_by_key('1nJmpNVlyePB09EnMFET_CAsLccu6hs-IP3fsRBG0UXQ')
worksheet = sh.get_worksheet(2)
data = worksheet.get_all_values()
draws_df = pd.DataFrame(data)
draws_df = draws_df[[0, 1, 2, 3, 4, 5, 6, 7]]
draws_df.drop([0, 1, 2, 3], inplace = True)
draws_df.columns = ['Sorteos', 'Nro1', 'Nro2', 'Nro3', 'Nro4', 'Nro5', 'Star_1', 'Star_2']
draws_df = draws_df.replace(r'^s*$', float('NaN'), regex = True)
draws_df.dropna(how = 'all', inplace = True)
draws_df.reset_index(drop = True, inplace = True)

entry_point = time.mktime((2004, 2, 6, 0, 0, 0, 0, 0, 0)) # reference for math - first Friday of February 2004
Tuesday = time.mktime((2004, 2, 10, 0, 0, 0, 0, 0, 0)) # reference for math with four days of difference
first_draw = time.mktime((2004, 2, 13, 0, 0, 0, 0, 0, 0)) # this is the first draws of Euro Millions
change_draw = time.mktime((2011, 5, 6, 0, 0, 0, 0, 0, 0)) # this is the Last draw of the old rule: one draw per week
next_week = first_draw - entry_point
next_Tuesday = Tuesday - entry_point
Tuesday_Friday = first_draw - Tuesday

dates = []
dates1 = []

for i in range(378):
    entry_point = days_sum(entry_point, next_week)
    dates.append(entry_point)

for i in range(379, len(draws_df) + 1):
    if i % 2:
        change_draw = days_sum(change_draw, next_Tuesday)
    else:
        change_draw = days_sum(change_draw, Tuesday_Friday)
    dates1.append(change_draw)

df = pd.DataFrame(list(map(datetime.fromtimestamp, dates)))
df1 = pd.DataFrame(list(map(datetime.fromtimestamp, dates1)))
frames = [df, df1]
result = pd.concat(frames, ignore_index = True)
result = result.rename(columns = {0:'Dates'})

df_clean = {
    'Sorteos': "int64",
    'Nro1': "int64",
    'Nro2': "int64",
    'Nro3': "int64",
    'Nro4': "int64",
    'Nro5': "int64",
    'Star_1': "int64",
    'Star_2': "int64"
}
draws_df = draws_df.astype(df_clean)
draws_df = pd.concat([result, draws_df], axis = 1, join = 'inner')

print("¡Done!")

print("Creando base de datos actualizada")

print(draws_df)

print(draws_df.dtypes)

print(draws_df.describe())

draws_df.to_parquet('db.parquet', index = False)