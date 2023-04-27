"""Functions to work with the data"""

# Standard libraries of Python
from collections import Counter
from decimal import Decimal, getcontext
getcontext().prec = 5

# Dependencies
import numpy as np
import pandas as pd
np.set_printoptions(precision=5)

def draw_generator(lenght):
    for i in range(12,lenght):
        yield i
        
def clean_df(df, columns_id, name):
    df.columns = columns_id
    df.columns.name = name
    df.index.name = 'Draws'
    return df
        
def numbers_boolean(database, boolean_df, numbers):
    row_0 = pd.DataFrame(columns=[str(i) for i in range(1, 51)], index=[0]).fillna(True)
    for e in range(1,6):
        col_name = f"Nro{e}"
        boolean_df = boolean_df | (database[col_name].to_numpy()[:, None] == numbers)
    boolean_df = pd.concat([row_0, boolean_df]).reset_index(drop=True)
    return boolean_df

def first_df_bool(database, numbers):
    
    def count_skips(df, list_numbers):
        counts = {str(key): 0 for key in list_numbers}
        for col in range(df.shape[1]):
            counter = 0
            for i in df.iloc[::-1, col]:
                if not i:
                    counter += 1
                else:
                    counts[str(col+1)] = counter
                    break
        return counts

    df = database.head(13).iloc[:, 2:7]
    main_winners_bool = pd.DataFrame(False, columns=[str(i) for i in range(1, 51)], index=range(len(df)))
    
    boolean_df = numbers_boolean(df, main_winners_bool, numbers)

    draws = np.arange(1, len(boolean_df))
    dic = {e: count_skips(boolean_df.iloc[:e], numbers) for e in draws}
    main_df = pd.DataFrame.from_dict(dic, orient='index')
    main_df.drop(main_df.index[0], inplace=True)
    main_df = main_df.reset_index(drop=True)
    main_df.index = main_df.index + 1

    return main_df

def count_hits(df_with_numbers, db_year, year_number):
    year, number = year_number
    filtered = df_with_numbers.loc[(db_year == year) & (df_with_numbers == number).any(axis=1), :]
    count = filtered.eq(number).sum().sum()
    return int(count)

def year_hits(database, df_with_numbers, numbers_quantity, count_hits_func):
    db_year = database['Dates'].dt.year

    year_history = np.zeros((len(numbers_quantity), db_year.max()-db_year.min()+1))
    
    for i, number in enumerate(numbers_quantity):
        for j, year in enumerate(range(db_year.min(), db_year.max()+1)):
            year_history[i, j] = count_hits_func(df_with_numbers, db_year, (year, number))

    year_history = pd.DataFrame(year_history.T, columns=numbers_quantity, index=np.arange(db_year.min(), db_year.max()+1))
    total_hits = year_history.sum(axis=0).to_frame().rename(columns={0: 'Hits'}).T
    total_hits.index.name = None
    total_hits = total_hits.astype('int32')
    
    return year_history, total_hits

def average_hits(database, hits_data, numbers, is_star=False):
    divide = 5 if not is_star else 2
    average = round(hits_data.iloc[0][numbers] / len(database) / divide, 6)
    result = pd.DataFrame({'Average_Hits': average})
    result.index.name = 'Numbers/Stars'
    return result.T

def total_average_hits(database, hits, numbers, is_star=False, aprox=False):
    divide = 2 if is_star else 5
    average_hits = hits.apply(lambda hits: hits / len(database) / divide)
    average_hits = average_hits.iloc[0].rename('Average').to_frame()
    average_hits.index.name = None
    if aprox:
        return Decimal(average_hits['Average'].sum()) / Decimal(int(numbers.max())) + Decimal(0.001)
    else:
        return Decimal(average_hits['Average'].sum()) / Decimal(int(numbers.max()))

def minimal_hits(database, hits, numbers, average, is_star=False, aprox=False):
    min_hits = total_average_hits(database, hits, numbers, is_star, aprox)
    return min_hits * Decimal(int(hits.iloc[0, 0])) / Decimal(float(average.iloc[0, 0]))

def natural_rotation(database, hits, numbers, data_average, index_start, index_end, is_star=False, aprox=False):
    average = total_average_hits(database, hits, numbers, is_star=is_star, aprox=aprox)
    m_hits = minimal_hits(database, hits, numbers, data_average, is_star, aprox)
    rotation = pd.DataFrame({'Hits': hits.iloc[0], 'Average_Numbers': data_average.iloc[0], 'Average': average, 'Hits_Needed': m_hits}, index=range(index_start, index_end + 1))
    rotation['Difference'] = rotation['Hits'] - rotation['Hits_Needed']
    return rotation

def get_rotations(database, hits, numbers, data_average, is_star=False):
    exact_rotation = natural_rotation(database, hits, numbers, data_average, 1, 50, is_star=is_star, aprox=False)
    aprox_rotation = natural_rotation(database, hits, numbers, data_average, 1, 50, is_star=is_star, aprox=True)
    return aprox_rotation, exact_rotation

def combination_count(database, numbers, df_with_numbers):
    def low_numbers(numbers):
        return set(range(1,26)).intersection(numbers)
    
    def make_list(list_of_numbers, count_type=None):
        if count_type == 'low_high':
            return [number in low_numbers(numbers) for number in list_of_numbers]
        elif count_type == 'odd_even':
            return [True if number % 2 != 0 else False for number in list_of_numbers]
        
    low_high = []
    odd_even = []
    for draw in range(len(database)):
        list_of_numbers = df_with_numbers.loc[draw, :].tolist()
        count = Counter(make_list(list_of_numbers, count_type='low_high'))
        low_high.append((count.get(True,0), count.get(False,0)))
        count = Counter(make_list(list_of_numbers, count_type='odd_even'))
        odd_even.append((count.get(True,0), count.get(False,0)))
    return low_high, odd_even

def combination_df(database, low_high_counts, odd_even_counts):
    COMBINATIONS = [(3,2), (2,3), (1,4), (4,1), (0,5), (5,0)]
    draws = set(range(0,len(database)))
    columns_id = ['3/2', '2/3', '1/4', '4/1', '0/5', '5/0']
    
    low_high = {}
    odd_even = {}
    for i in draws:
        counts_l_h = {}
        counts_o_e = {}
        for combination in COMBINATIONS:
            count_l_h = sum([1 for j in range(i-9,i+1) if combination[0] == low_high_counts[j][0] and combination[1] == low_high_counts[j][1]])
            counts_l_h[combination] = count_l_h
            count_o_e = sum([1 for j in range(i-9,i+1) if combination[0] == odd_even_counts[j][0] and combination[1] == odd_even_counts[j][1]])
            counts_o_e[combination] = count_o_e
        low_high[i] = counts_l_h
        odd_even[i] = counts_o_e
    
    low_high = clean_df(pd.DataFrame.from_dict(low_high, orient='index'), columns_id, 'L/H')
    odd_even = clean_df(pd.DataFrame.from_dict(odd_even, orient='index'), columns_id, 'O/E')
    return low_high, odd_even

def count_100_combinations(df, columns, combinations, name):
    count_dic = {i: {key: 0 for key in combinations} for i in range(1, len(df) - 99)}
    columns_id = ['3/2', '2/3', '1/4', '4/1', '0/5', '5/0']
    for i, _ in enumerate(range(1, len(df) - 99)):
        df_slice = df.iloc[i:i+100]
        counts = [df_slice[(df_slice[columns[0]] == combination[0]) & (df_slice[columns[1]] == combination[1])][columns[0]].count() for combination in combinations]
        count_dic[i+1] = dict(zip(combinations, counts))
    df = clean_df(pd.DataFrame.from_dict(count_dic, orient='index'), columns_id, name)
    return df

def games_7(column):
    games_dict = {0: 1, 1: 1, 2: 0.75, 3: 0.65, 4: 0.55, 5: 0.45, 6: 0.35, 7: 0.25}
    return games_dict.get(column, 0)

def games_12(column):
    games_dict = {8: 0.65, 9: 0.55, 10: 0.45, 11: 0.35, 12: 0.25}
    return games_dict.get(column, 0)
