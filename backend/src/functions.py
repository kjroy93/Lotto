# Standard Libraries of Python
from collections import Counter

# Dependencies
import pandas as pd
import numpy as np
from pandas import DataFrame
from pandas import Series
np.set_printoptions(precision=5)

# Cache class for faster calculations in the loop that simulates results
class Memoize:
	def __init__(self,func):
		self.func = func
		self.cache = {}

	def __get__(self,instance,owner):
		bound_func = self.func.__get__(instance,owner)
		return self.__class__(bound_func)

	def __call__(self,*args,**kwargs):
		if (self.func, args, tuple(kwargs.items())) in self.cache:
			return self.cache[(self.func, args, tuple(kwargs.items()))]
		else:
			result = self.func(*args,**kwargs)
			self.cache[(self.func, args, tuple(kwargs.items()))] = result
			return result

def draw_generator(size:int) -> int:
	for draw in range(12,size):
		yield draw

def games_7(column: Series) -> Series:
	games_dict = {
		0: 1,
		1: 1,
		2: 1,
		3: 1,
		4: 1,
		5: 0.65,
		6: 0.45,
		7: 0.25
	}
	return games_dict.get(column,0)

def games_12(column: Series) -> Series:
	games_dict = {
		8: 0.65,
		9: 0.55,
		10: 0.45,
		11: 0.35,
		12: 0.25
	}
	return games_dict.get(column,0)

def clean_df_skips(df: DataFrame,columns_id,name) -> DataFrame:
	df.columns = columns_id
	df.columns.name = name
	df.index.name = 'Draws'
	return df

def combination_count(database,numbers,df_with_numbers):
	def low_numbers(numbers):
		return set(range(1,26)).intersection(numbers)
	
	def make_list(list_of_numbers,count_type=None):
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

def combination_df(database,low_high_counts,odd_even_counts):
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
	
	low_high = clean_df_skips(pd.DataFrame.from_dict(low_high, orient='index'), columns_id, 'L/H')
	odd_even = clean_df_skips(pd.DataFrame.from_dict(odd_even, orient='index'), columns_id, 'O/E')
	return low_high, odd_even

def count_100_combinations(df, columns, combinations, name):
	count_dic = {i: {key: 0 for key in combinations} for i in range(1, len(df) - 99)}
	columns_id = ['3/2', '2/3', '1/4', '4/1', '0/5', '5/0']
	for i, _ in enumerate(range(1, len(df) - 99)):
		df_slice = df.iloc[i:i+100]
		counts = [df_slice[(df_slice[columns[0]] == combination[0]) & (df_slice[columns[1]] == combination[1])][columns[0]].count() for combination in combinations]
		count_dic[i+1] = dict(zip(combinations, counts))
	df = clean_df_skips(pd.DataFrame.from_dict(count_dic, orient='index'), columns_id, name)
	return df