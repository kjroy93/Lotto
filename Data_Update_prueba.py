from datetime import datetime
import time

"""Bienvenido a tu base de datos. Dime, ¿quieres actualizar los datos que tienes?
1. Sí
2. No
"""

print("Por favor, introduce los datos")
year = int(input("Introduce el año en curso: "))
month = int(input())
day = int(input())

print("El sorteo corresponde al día " + str(day) + " del mes " + str(month) + " en el año " + str(year))

time_tuple = time.mktime((year, month, day, 0, 0, 0, 0, 0, 0))
date = datetime.fromtimestamp(time_tuple)

print(date)