"""Scrapping functions"""

# Standard libraries of Python
from time import gmtime

# Dependencies
import pandas as pd
from pandas import DataFrame
import requests


def euro_scraping() -> DataFrame:
    """Return all euromillones data"""
    
    assert requests.get("https://www.euromillones.com.es/").status_code == 200, 'Hay un problema con la página de euromillones'
    
    
    def rename_sorteo(df: DataFrame) -> DataFrame:
        return df.rename(columns={
                    "SORTEO":'draw',
                    "FECHA":'dates',
                    "DIA":'dates',
                    "COMBINACION GANADORA":'nro1',
                    "COMBINACION GANADORA.1":'nro2',
                    "COMBINACION GANADORA.2":'nro3',
                    "COMBINACION GANADORA.3":'nro4',
                    "COMBINACION GANADORA.4":'nro5',
                    "COMBINACION GANADORA.5":'star_1',
                    "COMBINACION GANADORA.6":'star_2',
                    })

    sorteo_euro_millones = pd.DataFrame()
    for age in range(2004,gmtime().tm_year+1):
        try:
            if age < 2009:
                new_data = rename_sorteo(pd.read_html(f'https://www.euromillones.com.es/historico/resultados-euromillones-{age}.html',header=0)[0].drop(0))
                sorteo_euro_millones = pd.concat([sorteo_euro_millones,new_data],ignore_index=True)
            elif age == 2009 or age == 2010:
                new_data = rename_sorteo(pd.read_html(f'https://www.euromillones.com.es/historico/resultados-euromillones-{age}.html',header=0)[0][:-1 if age == 2009 else -2].drop(0).drop(columns="COMBINACION GANADORA.7"))
                sorteo_euro_millones = pd.concat([sorteo_euro_millones,new_data],ignore_index=True)
            elif age == 2018:
                new_data = rename_sorteo(pd.read_html(f'https://www.euromillones.com.es/historico/resultados-euromillones-{age}.html',header=0)[0].drop([0,18,19]).drop(columns=['SEM.','EL MILLÓN','Unnamed: 11']))
                sorteo_euro_millones = pd.concat([sorteo_euro_millones,new_data],ignore_index=True)
            else:
                new_data = rename_sorteo(pd.read_html(f'https://www.euromillones.com.es/historico/resultados-euromillones-{age}.html',header=0)[0].drop(0).drop(columns=['SEM.'] if age <= 2015 else ['SEM.','EL MILLÓN']))
                sorteo_euro_millones = pd.concat([sorteo_euro_millones,new_data.dropna()],ignore_index=True)
        except:
            print(f'Error en el año: {age}')
    assert not sorteo_euro_millones.empty, "El dataframe no ha sido llenado"
    return sorteo_euro_millones