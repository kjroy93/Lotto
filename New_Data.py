from datetime import datetime, timedelta
import time
import pandas as pd
import calendar

draws = pd.read_csv('base_de_datos.csv', dtype={'Sorteos': "string"}, parse_dates=['Dates'])

def new_date():
    year = int(input("Introduce el año en curso del nuevo sorteo: "))
    month = int(input("Introduce el mes del año en curso: "))
    day = int(input("Introduce el dia del sorteo: "))
    date = time.mktime((year, month, day, 0, 0, 0, 0, 0, 0))
    date = datetime.fromtimestamp(date)
    return date

def nd():
    draw = input("Introduce el número del sorteo: ")
    return draw

def numbers():
    numbers = []
    Nro1 = int(input("Introduce el primer número: "))
    Nro2 = int(input("Introduce el segundo número: "))
    Nro3 = int(input("Introduce el tercer número: "))
    Nro4 = int(input("Introduce el cuarto número: "))
    Nro5 = int(input("Introduce el quinto y último número: "))
    numbers.extend([Nro1, Nro2, Nro3, Nro4, Nro5])
    return numbers

def stars():
    stars = []
    Star_1 = int(input("Introduce la primera estrella: "))
    Star_2 = int(input("Introduce la segunda estrella: "))
    stars.extend([Star_1, Star_2])
    return stars

def audit_db(date, db, future):
    audit = []
    past_dates = date in set(draws['Dates'])
    audit.append(past_dates)
    future_dates = date in set(future['Dates'])
    audit.append(future_dates)
    return audit


print("""Bienvenido a tu base de datos. Dime, ¿quieres actualizar los datos que tienes?
1. Sí
2. No""") # Welcome message

answer = int(input())

while answer == 2:
    print("Estas a punto de salir del programa. ¿Estás seguro?")
    answer = input()
    if answer == "no":
        print("""Bienvenido a tu base de datos. Dime, ¿quieres actualizar los datos que tienes?
    1. Sí
    2. No""")
        answer = int(input())
    else:
        break

while answer == 1:

    df = pd.DataFrame(columns=['Dates', 'Sorteos', 'Nro1', 'Nro2', 'Nro3', 'Nro4', 'Nro5', 'Star_1', 'Star_2']) # first, we create an empty DataFrame

    CONSTANT = draws.shape[0] # this line obtain the number of register in the data base (draws)
    entry_point = draws.loc[CONSTANT - 1].iat[0] # entry point of the future dates. In other word, "next draw date"
    init = entry_point.timestamp()
    limit = time.mktime((2043, 1, 2, 0, 0, 0, 0, 0, 0)) # limit of the next draws DataFrame, up to twenty years
    future = []
    three_days = 3*24*60*60 # time library. Transfomer the days into seconds, in order to sum them to the original date
    four_days = 4*24*60*60 # same thing as above
    while init < limit:
        date_convert = datetime.fromtimestamp(init) # we obtain the date from the epoch (seconds)
        date_calendar = date_convert.isocalendar() # transformed into calendar, to obtain the day of the week
        if date_calendar[2] == 2:
            init += three_days
        else:
            init += four_days
        future.append(init)
    future = pd.DataFrame(list(map(datetime.fromtimestamp, future))) # we create the DataFrame, and in the same line, apply the function to convert the seconds, to a legible date
    future = future.rename(columns={0:'Dates'})
    future['Dates'] = pd.to_datetime(future['Dates']).dt.normalize() # since the date came with hours, we eliminated them. That is because, the boolean factor can not be obtained with the hours next to it.

    date = new_date()
    audit = audit_db(date, draws, future)

    while audit != [False, True]:
        if audit == [True, True]:
            print("La fecha que ingresas ya existe. Por favor, introduce una fecha nueva: ")
            date = new_date()
            audit = audit_db(date, draws, future)
        elif audit == [True, False]:
            print("La fecha que ingresas ya existe. Por favor, introduce una fecha nueva: ")
            date = new_date()
            audit = audit_db(date, draws, future)
        elif audit == [False, False]:
            print("La fecha que introduces, no corresponde con los sorteos siguientes. Por favor, introduce una fecha nueva: ")
            date = new_date()
            audit = audit_db(date, draws, future)
            
    draw = nd()
    draw_entry = draw
    new_draw = draw_entry in set(draws['Sorteos'])

    while new_draw != False:
        print("Este sorteo ya existe. Introduce otro número: ")
        draw = nd()
        new_draw = draw_entry in set(draws['Sorteos'])
        
    winners = numbers()
    duplicated = len(winners) != len(set(winners))

    while duplicated == True:
        print("No puede existir el mismo número dos veces en la misma serie. Por favor, introduce de nuevo los números ganadores")
        winners = numbers()
        duplicated = len(winners) != len(set(winners))

    stars = stars()
    stars_duplicated = len(stars) != len(set(stars))

    while stars_duplicated == True:
        print("No puede existir dos veces la misma estrella en la misma serie. Por favor, introduce de nuevo los números ganadores")
        stars = stars()
        stars_duplicated = len(stars) != len(set(stars))

    df = pd.DataFrame({
        'Dates': date,
        'Sorteos': draw,
        'Nro1': winners[0],
        'Nro2': winners[1],
        'Nro3': winners[2],
        'Nro4': winners[3],
        'Nro5': winners[4],
        'Star_1': stars[0],
        'Star_2': stars[1]
    }, index=[0])

    draws = pd.concat([draws, df], ignore_index=True)

    df_clean = {
        'Sorteos': "string",
        'Nro1': "int64",
        'Nro2': "int64",
        'Nro3': "int64",
        'Nro4': "int64",
        'Nro5': "int64",
        'Star_1': "int64",
        'Star_2': "int64"
    }

    draws = draws.astype(df_clean)
    draws.to_csv('base_de_datos.csv', index=False)
    print("Base de datos actualizada")
    draws

    print("¿Deseas registrar una nueva entrada? Si o no")

    answer = input()

    if answer == "Si":
        answer = 1
    else:
        break
