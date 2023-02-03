"""Archivo para realizar scrapping de euromillones
El scrapping se realiza del año 2013 al 2024. Ajustable
Solo se debe ajustar el año de inicio y fin"""

import pandas as pd
import requests
from bs4 import BeautifulSoup

def redimensionado(dataframe):
    """Función para limpieza de data scrapeada.
    Recibe df con clumnas con numeros del 0 al 11,
    Se renombran del 0 al 9.
    Se elimina las dos primeras lineas ya que no cuentan con información"""
    return dataframe[0].rename(columns={
        0:"SEM.",
        1:"SORTEO",
        2:"DIA",
        3:"NRO.1",
        4:"NRO.2",
        5:"NRO.3",
        6:"NRO.4",
        7:"NRO.5",
        8:"ESTRELLA.1",
        9:"ESTRELLA.2"
    }).drop([0,1])


sorteos= pd.DataFrame()

for year in range(2004,2013):

    url = f'https://www.euromillones.com.es/historico/resultados-euromillones-{year}.html'
    r = requests.get(url)
    bs = BeautifulSoup(r.text)
    datos = [i.text.split('\n') for i in bs.find_all('tr')]
    dfs = pd.DataFrame(datos)[2:54].rename(columns={
            0:"SEM.",
            1:"SORTEO",
            2:"DIA",
            3:"NRO.1",
            4:"NRO.2",
            5:"NRO.3",
            6:"NRO.4",
            7:"NRO.5",
            8:"ESTRELLA.1",
            9:"ESTRELLA.2"
        }).drop(columns=[10])
    dfs['DIA'] = dfs['DIA'] + f"-{year}"
    sorteos= pd.concat([sorteos,dfs])

for year in range(2012,2024):                                                                                                              
    url = f'https://www.euromillones.com.es/historico/resultados-euromillones-{year}.html'
    dfs = pd.read_html(url)
    dfs = redimensionado(dfs)
    dfs['DIA'] = dfs['DIA'] + f"-{year}"
    sorteos= pd.concat([sorteos,dfs])
sorteos.drop(columns=[10,11],inplace=True)
sorteos.dropna(inplace=True)
sorteos.reset_index(inplace=True)
sorteos.to_csv("sorteos.csv",index=False)
