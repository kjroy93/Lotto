import pandas as pd
import numpy as np

from data_analisys import data_functions
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP, getcontext
getcontext().prec = 5
np.set_printoptions(precision=5)

total_numbers = np.arange(1,51)
# array of skips
skips = np.arange(0,19)
# Extra row, in order to compare the number that did not appear for the first time in the game history
d_0 = pd.DataFrame(columns=[str(i) for i in range(1, 51)], index=[0]).fillna(True)

def analisys(db):
    # Load the data base and obtain the first DataFrame
    winning_numbers = db.iloc[:, 2:7]

    # Create a template DataFrame with all values set to False
    skip_winners_bool = pd.DataFrame(False, columns=[str(i) for i in range(1, 51)], index=range(len(db)))

    new_bool_df = pd.DataFrame((db[f"Nro{1}"].to_numpy()[:, None] == total_numbers), columns=[str(i) for i in range(1, 51)])

    for col_name in [f"Nro{e}" for e in range(1, 6)]:
        num = db[col_name].iloc[0]
        # actualizar valores de new_bool_df
        new_bool_df = pd.DataFrame((db[col_name].to_numpy()[:, None] == total_numbers), columns=[str(i) for i in range(1, 51)])
        skip_winners_bool = skip_winners_bool.shift(1, fill_value=False) | new_bool_df
        skip_winners_bool.iloc[0, skip_winners_bool.columns.get_loc(str(num))] = True
    skip_winners_bool = pd.concat([d_0, skip_winners_bool]).reset_index(drop=True)

    # Year History for numbers and stars
    numbers_year_history, total_hits = data_functions.year_hits(db, winning_numbers, total_numbers, data_functions.count_hits)
    ny_mean = pd.DataFrame(numbers_year_history.mean(), columns=['Average']).T.rename(index={'0': 'Average'})
    ny_median = pd.DataFrame(numbers_year_history.median(), columns=['Median']).T.rename(index={'0': 'Median'}).applymap(lambda x:(int(x+0.5)))
    numbers_year_history = pd.concat([ny_median, ny_mean, numbers_year_history])

    # Average of hits per numbers
    numbers_average = data_functions.average_hits(db, total_hits, total_numbers)

    # Get the natural rotation of the numbers
    df_aprox, df_exact = data_functions.get_rotations(db, total_hits, total_numbers, numbers_average, is_star=False)
    
    # It creates the list of draws, numbers and the dictionary to obtain the amount of skips per number if wins and looses.
    draws = np.arange(1, len(skip_winners_bool))
    dicts = {e: data_functions.count_skips(skip_winners_bool.iloc[:e], total_numbers) for e in draws}

    skip_numbers = pd.DataFrame.from_dict(dicts, orient='index')

    # Order the last draw for skips:
    last_draw = sorted(dicts[len(skip_numbers)].items(), key=lambda x: x[1])
    last_draw = pd.DataFrame({'Numero': [x[0] for x in last_draw], 'Skips': [int(x[1]) for x in last_draw]})

    # Select the last 12 draws
    if len(skip_numbers) == 12:
        last_12_draws = np.arange(len(skip_numbers) - 11, len(skip_numbers)+1)
    else:
        last_12_draws = np.arange(len(skip_numbers) - 12, len(skip_numbers))
        
    sk_12 = skip_numbers.loc[last_12_draws]

    # This establish the skips of the last 12 draws
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
    results_df.columns = [f'{i}_to_{i+9}' for i in range(1,51,10)]
    future_sg_10 = results_df.iloc[-9:]
    future_sg_5 = results_df.iloc[-4:]
    future_groups_df = pd.DataFrame({'10_games': (future_sg_10 > 0).sum(), '5_games': (future_sg_5 > 0).sum()}).T
    
    last_year = db['Dates'].iloc[-1]
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
    current_hits_needed = data_functions.minimal_hits(db, total_hits, total_numbers, numbers_average, aprox=True)

    # Finding rotation hit
    rotation_hit = next((draw for draw in range(len(db['Sorteo'])+1, len(db['Sorteo'])+12) if draw * data_functions.minimal_hits(db, total_hits, total_numbers, numbers_average, aprox=True) / len(db) > int(current_hits_needed) + 1), None)

    # Rotation criteria
    rotation_criteria = {}
    for i in total_numbers:
        category = rotation_info[i]
        diff_exact = df_exact.at[i, 'Difference']
        diff_aprox = df_aprox.at[i, 'Difference']
        hits_exact = df_exact.at[i, 'Hits']
        if category == 1 and diff_exact > 0.50:
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
    
    skips_7_12[['values_7', 'values_12']] = skips_7_12[['7', '12']].applymap(lambda x: data_functions.games_7(x) if x <= 7 else data_functions.games_12(x))

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

    return recommended_numbers, not_recommended_numbers