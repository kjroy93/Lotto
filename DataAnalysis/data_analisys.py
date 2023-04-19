# Libraries
import sys
sys.path.append('C:\Proyectos\Loteria\DataBase')
sys.path.append('C:\Proyectos\Loteria\DataAnalysis')
import datetime
import data_functions
import pandas as pd
import numpy as np
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP, getcontext
getcontext().prec = 5
np.set_printoptions(precision=5)

def analisys(db):
    # Load the data base and obtain the first DataFrame
    winning_numbers = db.iloc[:, 2:7]
    winning_stars = db.iloc[:, 7:9]
    total_numbers = np.arange(1,51)
    total_stars = np.arange(1,13)

    # Create a template DataFrame with all values set to False
    skip_winners_bool = pd.DataFrame(False, columns=[str(i) for i in range(1,51)], index=range(len(winning_numbers)))

    # Fill in the True values
    for e in range(1,6):
        for i in range(1,51):
            skip_winners_bool[f"{i}"] |= (winning_numbers[f"Nro{e}"] == i)

    # Add an extra row, in order to compare the number that did not appear for the first time in the game history
    d_0 = pd.DataFrame(columns = [str(i) for i in range(1,51)], index=[0]).fillna(True)

    # Create the final DataFrame
    skip_winners = pd.concat([d_0, skip_winners_bool]).reset_index(drop=True)

    # Hits of numbers
    numbers_hits = data_functions.get_hits(db, winning_numbers, total_numbers)

    # Year History for numbers and stars
    numbers_year_history = data_functions.year_hits(db, winning_numbers, total_numbers)
    ny_mean = pd.DataFrame(numbers_year_history.mean(), columns=['Average']).T.rename(index={'0': 'Average'})
    ny_median = pd.DataFrame(numbers_year_history.median(), columns=['Median']).T.rename(index={'0': 'Median'}).applymap(lambda x:(int(x+0.5)))
    numbers_year_history = pd.concat([ny_median, ny_mean, numbers_year_history])

    # Average of hits per numbers
    numbers_average = data_functions.average_hits(db, numbers_hits, total_numbers)

    # Natural rotation of the numbers and stars, using exact and aproximation values
    aprox_rotation_BN_low, aprox_rotation_BN_high, exact_rotation_BN_low, exact_rotation_BN_high = data_functions.get_rotations(db, numbers_hits, total_numbers, numbers_average, is_star=False)

    # It creates the list of draws, numbers and the dictionary to obtain the amount of skips per number if wins and looses.
    draws = list(np.arange(1,len(skip_winners)))
    numbers = [str(i) for i in range(1,51)]
    dicts = {draw: {key:[] for key in numbers} for draw in draws}

    for e in draws:
        df = skip_winners.loc[:e]
        counts = data_functions.count_skips(df, numbers)
        dicts[e].update(counts)

    skip_numbers = pd.DataFrame(dicts).T

    # Order the last draw for skips:
    last_draw = sorted(dicts[len(skip_numbers)].items(), key=lambda x: x[1])
    last_draw = pd.DataFrame({'Numero': [x[0] for x in last_draw], 'Skips': [int(x[1]) for x in last_draw]})

    # Select the last 12 draws
    last_12_draws = np.arange(len(skip_numbers) - 12, len(skip_numbers) + 1)
    sk_12 = skip_numbers.loc[last_12_draws]

    # This establish the skips of the last 12 draws
    skips = np.arange(0,19)
    aus_12 = [sk_12.loc[i - 1, str(column)] for i in last_12_draws[1:13] for column in sk_12 if sk_12.loc[i, str(column)] == 0]
    counter_7 = Counter(aus_12[25:60])
    counter_12 = Counter(aus_12)
    last_7 = [counter_7.get(i,0) for i in skips]
    last_12 = [counter_12.get(i,0) for i in skips]
    skips_7_12 = pd.DataFrame({'7': last_7, '12': last_12})

    groups = [list(range(i,i+10)) for i in range(1,51,10)]
    group_names = [tuple(range(i,i+10)) for i in range(1,51,10)]
    results = {i: {group_name: sum([1 for num in row if num in group]) for group_name, group in zip(group_names, groups)} for i, row in winning_numbers.iterrows()}
    results_df = pd.DataFrame.from_dict(results, orient='index')
    results_df.columns = ['{}_to_{}'.format(i,i+9) for i in range(1,51,10)]
    sg_10 = results_df.iloc[-10:]
    sg_5 = results_df.iloc[-5:]
    groups_df = pd.DataFrame({'10_games': (sg_10 > 0).sum(), '5_games': (sg_5 > 0).sum()}).T
    future_sg_10 = results_df.iloc[-9:]
    future_sg_5 = results_df.iloc[-4:]
    future_groups_df = pd.DataFrame({'10_games': (future_sg_10 > 0).sum(), '5_games': (future_sg_5 > 0).sum()}).T

    years = db['Dates'].dt.year.to_frame()
    years = years.rename(columns={'Dates': 'Years'})
    months = db['Dates'].dt.month.to_frame()
    months = months.rename(columns={'Dates': 'Month'})
    year_month = pd.concat([years, months, winning_numbers], axis=1)
    series_dates = year_month.melt(id_vars=['Years', 'Month'], var_name='Data', value_vars=['Nro1', 'Nro2', 'Nro3', 'Nro4', 'Nro5'])
    df_dates = series_dates.groupby(['Years', 'Month', 'Data'])['value'].value_counts().unstack(fill_value=0)
    cols = ['Nro1', 'Nro2', 'Nro3', 'Nro4', 'Nro5']
    month_history = df_dates[df_dates.index.get_level_values('Data').isin(cols)].groupby(level=['Years', 'Month']).sum()

    COMBINATIONS = [(3,2), (2,3), (1,4), (4,1), (0,5), (5,0)]
    low_high, odd_even = data_functions.combination_count(db, total_numbers, winning_numbers)
    low_high_df, odd_even_df = data_functions.combination_df(db, low_high, odd_even)
    low_high = pd.DataFrame(low_high)
    odd_even = pd.DataFrame(odd_even)
    LHOE = pd.concat([low_high, odd_even], axis=1)
    LHOE.columns = ['L', 'H', 'O', 'E']
    low_high = LHOE.iloc[:, 0:2]
    odd_even = LHOE.iloc[:, 2:4]

    if len(low_high) < 100 and len(odd_even) < 100:
        pass
    else:
        low_high = data_functions.count_100_combinations(low_high, ['L', 'H'], COMBINATIONS, 'low/high')
        odd_even = data_functions.count_100_combinations(odd_even, ['O', 'E'], COMBINATIONS, 'odd/even')

    last_year = db['Dates'].iloc[-1]
    current_year = last_year.year
    year_info = numbers_year_history.loc[['Median', 'Average', current_year], :50]
    year_criteria = {key: [] for key in total_numbers}
    df_median = year_info.loc['Median']
    df_average = year_info.loc['Average']
        
    for number in total_numbers:
        x = year_info.at[current_year, number]
        median_half = df_median[number] / 2
            
        if x == 0 or x <= median_half:
            y = round((1 - ((df_median[number] * 100) / df_average[number]) / 100), 2)
        else:
            x_percentage = round((x * 100 / df_average[number]) / 100, 2)
            y = round((1 - x_percentage if x_percentage > 1 else 1 - x_percentage), 2)
            
            
        if np.isnan(y):
            y = 0
            
        year_criteria[number] = float(y)

    year_criteria = pd.DataFrame.from_dict(year_criteria, orient='index', columns=['year_criteria'])

    df_aprox = pd.concat([aprox_rotation_BN_low, aprox_rotation_BN_high])
    df_exact = pd.concat([exact_rotation_BN_low, exact_rotation_BN_high])
    best_numbers = df_aprox[df_aprox['Hits'] > df_aprox['Hits_Needed']].index.tolist()
    normal_numbers = df_exact[(df_exact['Hits'] > df_exact['Hits_Needed']) & ~(df_exact.index.isin(best_numbers))].index.tolist()
    missing_numbers = sorted(list(set(range(1,51)) - set(best_numbers) - set(normal_numbers)))

    rotation_info = {}
    for number in total_numbers:
        if number in best_numbers:
            rotation_info[number] = 2
        elif number in normal_numbers:
            rotation_info[number] = 1
        elif number in missing_numbers:
            rotation_info[number] = 0

    current_hits_needed = data_functions.minimal_hits(db, numbers_hits, total_numbers, numbers_average, aprox=True)

    for draw in range(len(db['Sorteo'])+1, len(db['Sorteo'])+12):
        draw_until_rotation = draw * data_functions.minimal_hits(db, numbers_hits, total_numbers, numbers_average, aprox=True) / len(db)
        if draw_until_rotation > int(current_hits_needed) + 1:
            rotation_hit = draw
            break

    rotation_criteria = {key:[] for key in total_numbers}
    for i in total_numbers:
        category = rotation_info[i]
        if category == 1 and df_exact.at[i, 'Difference'] >= 0:
            rotation_criteria[i] = 0.50
        elif category == 1 and df_exact.at[i, 'Difference'] > -1 and df_exact.at[i, 'Difference'] <= 0.40:
            x = (abs(df_exact.at[i, 'Difference']) + (Decimal('0.5') * Decimal('100')) / Decimal('0.50')) / Decimal('100') + Decimal('0.50')
            rotation_criteria[i] = float(x.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))
        elif category == 2 and df_aprox.at[i, 'Hits'] > current_hits_needed:
            rotation_criteria[i] = 1
        elif category == 2 and df_aprox.at[i, 'Difference'] > -1 and rotation_hit - len(db) < 4 and len(best_numbers) < 17:
            x = (abs(df_exact.at[i, 'Difference']) * Decimal('100')) / Decimal('1.00') / Decimal('100') + Decimal('1')
            rotation_criteria[i] = float(x.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))
        else:
            rotation_criteria[i] = 0.25

    rotation_criteria = pd.DataFrame.from_dict(rotation_criteria, orient='index', columns=['rotation_criteria'])

    future_groups_df.columns = [1,2,3,4,5]

    gp10 = {}
    for column, gp in future_groups_df.loc['10_games'].items():
        if gp >= 2 and gp <= 4:
            gp10[tuple(range(1 + 10*(column-1), 11 + 10*(column-1)))] = 1
        elif gp == 5:
            gp10[tuple(range(1 + 10*(column-1), 11 + 10*(column-1)))] = 0.70
        elif gp == 6:
            gp10[tuple(range(1 + 10*(column-1), 11 + 10*(column-1)))] = 0.675
        elif gp == 7:
            gp10[tuple(range(1 + 10*(column-1), 11 + 10*(column-1)))] = 0.650
        elif gp == 8:
            gp10[tuple(range(1 + 10*(column-1), 11 + 10*(column-1)))] = 0.625
        elif gp == 9:
            gp10[tuple(range(1 + 10*(column-1), 11 + 10*(column-1)))] = 0.6
        elif gp == 10:
            gp10[tuple(range(1 + 10*(column-1), 11 + 10*(column-1)))] = 0.5

    gp10_df = pd.DataFrame.from_dict(gp10, orient='index').T
    gp10_df.columns = ['{}_to_{}'.format(i,i+9) for i in range(1,51,10)]
    gp10_df = gp10_df.rename(index={0: 'Criteria'})

    group_criteria_dict = {key:[] for key in total_numbers}
    for i, group in enumerate(groups):
        gn = group
        for number in total_numbers:
            if number in gn:
                group_criteria_dict[number] = gp10_df.iloc[0,i]

    group_criteria = pd.DataFrame.from_dict(group_criteria_dict, orient='index', columns=['group_criteria'])

    skips_7_12['values_7'] = skips_7_12['7'].apply(data_functions.games_7)
    skips_7_12['values_12'] = skips_7_12['12'].apply(data_functions.games_12)
    
    position_criteria = {key: [] for key in total_numbers}

    game0 = 0.5
    if skips_7_12.iloc[0,0] <= 5:
        skips_7_12.iloc[0,2] = game0
        
    for i, key in zip(last_draw.iloc[:,1], last_draw.iloc[:,0]):
        value_added = False
        for e, value in enumerate(skips_7_12.iloc[:,2]):
            if skips_7_12.iloc[e,0] <= 7 and e == i:
                position_criteria[int(key)] = value
                value_added = True
            elif skips_7_12.iloc[e,0] >= 8 and e == i:
                value = skips_7_12.iloc[e,3]
                position_criteria[int(key)] = value
                value_added = True
        if not value_added:
            if len(last_draw.loc[last_draw['Skips'] <= 12]) <= 38:
                position_criteria[int(key)] = 1
            else:
                position_criteria[int(key)] = 0.25

    position_criteria = pd.DataFrame.from_dict(position_criteria, orient='index', columns=['position_criteria'])

    df_numbers = pd.concat([year_criteria, rotation_criteria, group_criteria, position_criteria], axis=1)
    print(df_numbers)
    df_numbers['evaluation'] = df_numbers.apply(lambda row: row['year_criteria'] + row['rotation_criteria'] + row['group_criteria'] + row['position_criteria'], axis=1)
    tomorrow_numbers = df_numbers[['evaluation']]
    tomorrow_numbers = tomorrow_numbers.sort_values(by='evaluation', ascending=False)
    criterion = tomorrow_numbers['evaluation'].median()
    recommended_numbers = tomorrow_numbers.loc[tomorrow_numbers['evaluation'] >= criterion]
    not_recommended_numbers = tomorrow_numbers.loc[tomorrow_numbers['evaluation'] < criterion]

    return recommended_numbers, not_recommended_numbers