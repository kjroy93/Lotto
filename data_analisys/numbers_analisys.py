"""Main function to analize the data of the 5 numbers in the array of 50 numbers in Euro Millions lottery game, the Europe Lottery, based in historical records"""

# Standard libraries of Python
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP, getcontext

import legacy_data_functions
getcontext().prec = 5

# Dependencies
import pandas as pd
import numpy as np
np.set_printoptions(precision=5)

# Libraries made for the Proyect
from data_analisys import legacy_data_functions

# Array of numbers
total_numbers = np.arange(1,51)

# Array of skips
skips = np.arange(0,19)

def analisys(db, boolean_df, main_counts):
    # Load the data base and obtain the first DataFrame
    winning_numbers = db.iloc[:, 2:7]

    # Year History for numbers and stars
    numbers_year_history, total_hits = legacy_data_functions.year_hits(db, winning_numbers, total_numbers, legacy_data_functions.count_hits)
    ny_mean = pd.DataFrame(numbers_year_history.mean(), columns=['Average']).T.rename(index={'0': 'Average'})
    ny_median = pd.DataFrame(numbers_year_history.median(), columns=['Median']).T.rename(index={'0': 'Median'}).applymap(lambda x:(int(x+0.5)))
    numbers_year_history = pd.concat([ny_median, ny_mean, numbers_year_history])

    # Average of hits per numbers
    numbers_average = legacy_data_functions.average_hits(db, total_hits, total_numbers)

    # Get the natural rotation of the numbers
    df_aprox, df_exact = legacy_data_functions.get_rotations(db, total_hits, total_numbers, numbers_average, is_star=False)
    
    # Continue of the Main Counts Df
    mcd_last_row = main_counts.iloc[-1:].reset_index(drop=True)

    # Last row of Boolean Df in Current Flow
    bdcf_last_row = boolean_df.iloc[len(winning_numbers), :].to_frame().T.reset_index(drop=True)

    # Replace the values of last row to prepare for sum
    true_to_nan = bdcf_last_row.replace(True, np.nan)
    bdcf_last_row = true_to_nan.replace(False, 1)

    # Sum of rows
    new_row = mcd_last_row + bdcf_last_row
    new_row = new_row.replace(np.nan, 0).astype(int)

    # New version of Main Counts Df
    main_counts = pd.concat([main_counts, new_row], ignore_index=True)
    main_counts.index = main_counts.index + 1

    # Order the last draw for skips:
    last_draw = new_row.transpose().sort_values(by=0).reset_index()
    last_draw = last_draw.rename(columns={'index': 'Numero', 0: 'Skips'})
    
    # Select the last 12 draws
    if len(main_counts) - 12 == 0:
        last_12_draws = np.arange(1, len(main_counts))
    else:
        pass

    last_12_draws = np.arange(len(main_counts) - 11, len(main_counts) + 1)
    sk_12 = main_counts.loc[last_12_draws]

    # This establish the skips of the last 12 draws
    aus_12 = [sk_12.loc[i, str(column)] for i in last_12_draws[0:11] for column in sk_12 if sk_12.loc[i, str(column)] == 0]
    counter_7 = Counter(aus_12[25:60])
    counter_12 = Counter(aus_12)
    last_7 = [counter_7.get(i,0) for i in skips]
    last_12 = [counter_12.get(i,0) for i in skips]
    skips_7_12 = pd.DataFrame({'7': last_7, '12': last_12})
    
    groups = [list(range(i,i+10)) for i in range(1,51,10)]
    group_names = [tuple(range(i,i+10)) for i in range(1,51,10)]
    results = {i: {group_name: sum([1 for num in row if num in group]) for group_name, group in zip(group_names, groups)} for i, row in winning_numbers.iterrows()}
    results_df = pd.DataFrame.from_dict(results, orient='index')
    results_df.columns = [f'{i}_to_{i+9}' for i in range(1,51,10)]
    future_sg_10 = results_df.iloc[-9:]
    future_sg_5 = results_df.iloc[-4:]
    future_groups_df = pd.DataFrame({'10_games': (future_sg_10 > 0).sum(), '5_games': (future_sg_5 > 0).sum()}).T
    
    last_year = db['dates'].iloc[-1]
    current_year = last_year.year
    year_info = numbers_year_history.loc[['Median', 'Average', current_year], :50]
    year_criteria = {key: [] for key in total_numbers}
    df_median = year_info.loc['Median']
    df_average = year_info.loc['Average']

    for number in total_numbers:
        x = year_info.at[current_year, number]
        median_half = df_median[number] / 2

        if df_average[number] != 0 and not np.isnan(df_average[number]):
            if x == 0 or x <= median_half:
                y = round((1 - ((df_median[number] * 100) / df_average[number]) / 100), 2)
            else:
                x_percentage = round((x * 100 / df_average[number]) / 100, 2)
                y = round((1 - x_percentage if x_percentage > 1 else 1 - x_percentage), 2)
        else:
            y = 0

        year_criteria[number] = float(y)

    year_criteria = pd.DataFrame.from_dict(year_criteria, orient='index', columns=['year_criteria'])

   # Finding missing numbers
    best_numbers = df_aprox.loc[df_aprox['Hits'] > df_aprox['Hits_Needed']].index.to_numpy()
    normal_numbers = df_exact.loc[(df_exact['Hits'] > df_exact['Hits_Needed']) & ~(df_exact.index.isin(best_numbers))].index.to_numpy()

    # Rotation information
    rotation_info = {number: 0 for number in total_numbers}
    for number in best_numbers:
        rotation_info[number] = 2
    for number in normal_numbers:
        rotation_info[number] = 1

    # Finding current hits needed
    current_hits_needed = legacy_data_functions.minimal_hits(db, total_hits, total_numbers, numbers_average, aprox=True)

    # Finding rotation hit
    rotation_hit = next((draw for draw in range(len(db['draw'])+1, len(db['draw'])+12) if draw * legacy_data_functions.minimal_hits(db, total_hits, total_numbers, numbers_average, aprox=True) / len(db) > int(current_hits_needed) + 1), None)

    # Rotation criteria
    rotation_criteria = {}
    for i in total_numbers:
        category = rotation_info[i]
        diff_exact = df_exact.at[i, 'Difference']
        diff_aprox = df_aprox.at[i, 'Difference']
        hits_exact = df_exact.at[i, 'Hits']
        if category == 1 and diff_exact > 0:
            rotation_criteria[i] = 0.50
        elif category == 1 and -1 < diff_exact <= 0.40:
            x = (abs(diff_exact) + (Decimal('0.5') * Decimal('100')) / Decimal('0.50')) / Decimal('100') + Decimal('0.50')
            rotation_criteria[i] = float(x.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))
        elif category == 2 and hits_exact > current_hits_needed:
            rotation_criteria[i] = 1
        elif category == 2 and diff_aprox > -1 and rotation_hit is not None and rotation_hit - len(db) < 4 and len(best_numbers) < 17:
            x = (abs(diff_exact) * Decimal('100')) / Decimal('1.00') / Decimal('100') + Decimal('1')
            rotation_criteria[i] = float(x.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))
        else:
            rotation_criteria[i] = 0.25

    rotation_criteria = pd.DataFrame.from_dict(rotation_criteria, orient='index', columns=['rotation_criteria'])
    
    future_groups_df.columns = [1,2,3,4,5]
    gp10 = {}
    gp_values = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 0.70, 6: 0.675, 7: 0.650, 8: 0.625, 9: 0.6, 10: 0.5}

    for column, gp in future_groups_df.loc['10_games'].items():
        gp_range = tuple(range(1 + 10*(column-1), 11 + 10*(column-1)))
        gp10[gp_range] = gp_values.get(gp, 0)

    gp10_df = pd.DataFrame.from_dict(gp10, orient='index').T
    gp10_df.columns = [1,2,3,4,5]
    gp10_df = gp10_df.rename(index={0: 'Criteria'})

    group_criteria_dict = {key:[] for key in total_numbers}
    for i, group in enumerate(groups):
        gn = group
        for number in total_numbers:
            if number in gn:
                group_criteria_dict[number] = gp10_df.iloc[0,i]

    group_criteria = pd.DataFrame.from_dict(group_criteria_dict, orient='index', columns=['group_criteria'])
    
    skips_7_12[['values_7', 'values_12']] = skips_7_12[['7', '12']].applymap(lambda x: legacy_data_functions.games_7(x) if x <= 7 else legacy_data_functions.games_12(x))

    position_criteria = {key: [] for key in total_numbers}

    game0 = 0.5
    if skips_7_12.iloc[0, 0] <= 5:
        skips_7_12.iloc[0, 2] = game0

    for i, key in zip(last_draw.iloc[:,1], last_draw.iloc[:,0]):
        value_added = False
        for e in range(len(skips_7_12)):
            if skips_7_12.iloc[e, 0] <= 7 and e == i:
                position_criteria[int(key)] = skips_7_12.at[e, 'values_7']
                value_added = True
            elif skips_7_12.iloc[e, 0] >= 8 and e == i:
                position_criteria[int(key)] = skips_7_12.at[e, 'values_12']
                value_added = True
        if not value_added:
            if len(last_draw.loc[last_draw['Skips'] <= 12]) <= 38:
                position_criteria[int(key)] = 1
            else:
                position_criteria[int(key)] = 0.25

    position_criteria = pd.DataFrame.from_dict(position_criteria, orient='index', columns=['position_criteria'])

    df_numbers = pd.concat([year_criteria, rotation_criteria, group_criteria, position_criteria], axis=1)
    df_numbers['Criteria'] = df_numbers.apply(lambda row: row['year_criteria'] + row['rotation_criteria'] + row['group_criteria'] + row['position_criteria'], axis=1)
    tomorrow_numbers = df_numbers[['Criteria']]
    tomorrow_numbers = tomorrow_numbers.sort_values(by='Criteria', ascending=False)
    criterion = tomorrow_numbers['Criteria'].median()
    recommended_numbers = tomorrow_numbers.loc[tomorrow_numbers['Criteria'] >= criterion].reset_index().rename(columns={'index': 'Numbers'})
    not_recommended_numbers = tomorrow_numbers.loc[tomorrow_numbers['Criteria'] < criterion].reset_index().rename(columns={'index': 'Numbers'})

    return recommended_numbers, not_recommended_numbers, main_counts