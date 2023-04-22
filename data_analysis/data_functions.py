"""Function with data"""

from collections import Counter
from decimal import Decimal, getcontext

import numpy as np
import pandas as pd

getcontext().prec = 5
np.set_printoptions(precision=5)

def draw_generator(lenght):
    for i in range(12,lenght):
        yield i

def clean_df(df, columns_id, name):
    df.columns = columns_id
    df.columns.name = name
    df.index.name = 'Draws'
    return df

def count_skips(df, list_numbers):
    counts = {str(key): 0 for key in list_numbers}
    df_len = len(df)
    for col in range(df.shape[1]):
        counter = 0
        for i in df.iloc[::-1, col]:
            if not i:
                counter += 1
            else:
                counts[str(col+1)] = counter
                break
    return counts

def max_output(df):
    df = df.T
    max_numbers = df.sort_values(by=df.columns[0], ascending=False)
    return max_numbers

def year_hits(database, df_with_numbers, numbers_quantity):
    db_year = database['Dates'].dt.year
    
    def count_hits(year_number):
        year, number = year_number
        filtered = df_with_numbers.loc[(db_year == year) & (df_with_numbers == number).any(axis=1), :]
        count = filtered.eq(number).sum().sum()
        return pd.Series({'Year': year, 'Number': number, 'Count': int(count)})
    
    year_numbers = [(y, n) for y in range(db_year.min(), db_year.max()+1) for n in numbers_quantity]
    year_history = pd.DataFrame(year_numbers, columns=['Year', 'Number'])
    year_history = year_history.apply(count_hits, axis=1)
    year_history = year_history.pivot_table(index='Year', columns='Number', values='Count', fill_value=0)
    total_hits = year_history.sum().reset_index()
    total_hits = total_hits.rename(columns = {'Number': 'Numbers', 'Count': 'Hits'}).set_index('Numbers').T
    total_hits = total_hits.iloc[0].rename('Hits').to_frame()
    total_hits.index.name = None
    return year_history, total_hits.T

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

def average_hits(database, hits_data, numbers, is_star=False):
    divide = 2 if is_star else 5
    hits_by_num = hits_data.apply(lambda hits: hits / len(database) / divide)
    hits_by_num = hits_by_num.iloc[0].rename('Average').to_frame()
    hits_by_num.index.name = None
    averages = []
    
    for num in numbers:
        avg = hits_by_num.loc[num]['Average'].sum()
        averages.append(round(avg, 6))
        
    result = pd.DataFrame({'Numbers/Stars': numbers, 'Average_Hits': averages})
    result = result.T
    result.columns = result.iloc[0].astype(int)
    result = result[1:]
    return result

def natural_rotation(database, hits, numbers, data_average, index_start, index_end, is_star=False, aprox=False):
    average = total_average_hits(database, hits, numbers, is_star, aprox)
    m_hits = minimal_hits(database, hits, numbers, data_average, is_star, aprox)
    rotation = pd.DataFrame({'Hits': hits.iloc[0], 'Average_Numbers': data_average.iloc[0], 'Average': average, 'Hits_Needed': m_hits}, index = range(index_start, index_end + 1))
    rotation['Difference'] = rotation['Hits'] - rotation['Hits_Needed']
    return rotation

def get_rotations(database, hits, numbers, data_average, is_star=False):
    aprox_rotations = []
    exact_rotations = []
    if is_star:
        index_ranges = [(1,6,'_low'), (7, 12, '_high')]
    else:
        index_ranges = [(1, 25, '_low'), (26, 50, '_high')]
    
    for start, end, suffix in index_ranges:
        # exact rotations
        exact_rotation = natural_rotation(database, hits, numbers, data_average, start, end, is_star=is_star, aprox=False)
        exact_rotations.append(exact_rotation)
        # approx rotations
        aprox_rotation = natural_rotation(database, hits, numbers, data_average, start, end, is_star=is_star, aprox=True)
        aprox_rotations.append(aprox_rotation)
    return tuple(aprox_rotations + exact_rotations)

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
    if column == 0 or column == 1:
        return 1
    elif column == 2:
        return 0.75
    elif column == 3:
        return 0.65
    elif column == 4:
        return 0.55
    elif column == 5:
        return 0.45
    elif column == 6:
        return 0.35
    elif column == 7:
        return 0.25

def games_12(column):
    if column == 8:
        return 0.65
    elif column == 9:
        return 0.55
    elif column == 10:
        return 0.45
    elif column == 11:
        return 0.35
    elif column == 12:
        return 0.25
    else:
        return 0