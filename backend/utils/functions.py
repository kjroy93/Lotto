# Standard Libraries of Python
from collections import Counter

# Dependencies
import pandas as pd
import numpy as np
from pandas import DataFrame
from pandas import Series
# from backend.src.pick_numbers import Selection
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

# def find_number(sources:list | DataFrame = None, column_name:str = None, number: int = None):
# 	"""
#     Searches for a number in different data structures.

#     Parameters:
#         sources (list | DataFrame): A list of DataFrames or a list to search in.
#         column_name (str, optional): The column name if a DataFrame is provided.
#         number (int, optional): The number to search for.

#     Returns:
#         DataFrame | list | None: The matching rows (if DataFrame), the found number (if list), or None if not found.
#     """

# 	for source in sources:
# 		if isinstance(source, DataFrame):
# 			if source.loc[source[column_name] == number].empty:

# def answers_to_inputs(instance: Selection, numbers: list[int]):
# 	recommended_numbers = instance.recommended_numbers
# 	not_recommended = instance.not_recommended_numbers

# 	def prints(df: DataFrame, df_1: DataFrame, column_name, number):
# 		if df.loc[df[column_name] == number].empty:
# 			print(df_1.loc[df_1[column_name] == number])
# 		else:
# 			print(df.loc[df[column_name] == number])

# 	for i in numbers:
# 		prints(recommended_numbers, not_recommended, 'numbers', i)