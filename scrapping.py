"""Archivo para realizar scrapping de euromillones
El scrapping se realiza del año 2013 al 2024. Ajustable
Solo se debe ajustar el año de inicio y fin"""

import pandas as pd

STARTYEAR = 2013
LASTYEAR = 2023


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
for year in range(STARTYEAR,LASTYEAR+1):                                                                                                              
    url = f'https://www.euromillones.com.es/historico/resultados-euromillones-{year}.html'
    dfs = pd.read_html(url)
    dfs = redimensionado(dfs)
    sorteos= pd.concat([sorteos,dfs])
sorteos.drop(columns=[10,11],inplace=True)
sorteos.dropna(inplace=True)
sorteos.to_csv("sorteos.csv",index=False)
