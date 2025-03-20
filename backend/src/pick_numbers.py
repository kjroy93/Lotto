# Standard Libraries of Python
import random
from typing import Union, Literal
from decimal import Decimal, ROUND_HALF_UP

# Dependencies
import pandas as pd
import numpy as np
from pandas import DataFrame, Index
from pandas import Series
np.set_printoptions(precision=5)

# Libraries made for this Proyect
from .parse import Criteria

# Class that select the winning numbers and construct the tickets. Initial population for Genetic Algorithm
class Selection:
	def __df_numbers(self) -> Series:
		self._df_values = self.euromillions.last_draw.groupby('skips')['number'].apply(lambda x: list(x)).reset_index().set_index('skips')
		self._df_values = self._df_values.rename_axis('number').rename_axis(None)
		self._df_values['number'] = self._df_values['number'].apply(lambda x: sorted(x))
		return self._df_values['number']

	def __clean_indexes(self, df: DataFrame, numbers: Series):
		self.df_skips = df.copy()
		idx_numbers = numbers.index.unique().tolist()
		skips_0 = self.df_skips[self.df_skips['7'] == 0].index.unique().tolist()

		for idx in skips_0:
			try:
				self._df_values.loc[idx]
			except KeyError:
				self.df_skips.drop(index=idx,inplace=True)

		self.df_skips = self.df_skips[self.df_skips.index.isin(idx_numbers)]
		return self.df_skips

	def __init__(self, euromillions: Criteria):
		# Store an instance of Euro Millions (Criteria) for access to it´s attributes and methods
		self.euromillions = euromillions

		# Copy of DataFrames for future reference
		self.recommended_numbers = self.euromillions.recommended_numbers.copy()
		self.not_recommended_numbers = self.euromillions.not_recommended_numbers.copy()

		# List of numbers to be selected
		self.numbers = self.__df_numbers()

		# List of clean indexes
		self.__clean_indexes(self.euromillions.skips_7_12,self.numbers)

		# Counter for cold numbers
		self._cold = 0

		# Counter/control for double selection
		self._idx_dy = {}
		self.rng_count = 0
		self._even = 0
		self._odd = 0
		self._low = 0
		self._high = 0

		# Counter for recommended numbers
		self._recommended_numbers_selected = 0

		# Counter for not recommended numbers
		self._not_recommended_numbers_selected = 0

		# List to store the selected numbers
		self._selected_numbers = []

	def __probability(self, n_category: DataFrame) -> DataFrame:
		probability = 1 / len(n_category)
		n_category['criteria'] = n_category['criteria'] * (1 + probability)
		return n_category

	def __remove_number(self, number: np.int32 | int, n_category: DataFrame) -> DataFrame:
		category = self.recommended_numbers if n_category.equals(self.recommended_numbers) else self.not_recommended_numbers
		category = category.drop(category[category['numbers'] == number].index).reset_index(drop=True)
		category = self.__probability(category)

		if n_category.equals(self.recommended_numbers):
			self.recommended_numbers = category
		else:
			self.not_recommended_numbers = category

		return category

	def __sum_selected_number(self, selected_number: int):
			if self.euromillions.recommended_numbers['numbers'].isin([selected_number]).any():
				self._recommended_numbers_selected += 1
			else:
				self._not_recommended_numbers_selected += 1
	
	def __update_category(self, condition, counter_name, idx):
		if condition and getattr(self, counter_name) < 5:
			setattr(self, counter_name, getattr(self, counter_name) + 1)
		else:
			self.__check_idx(idx, removal=True)
			raise ValueError(f"The selected number category {counter_name} is complete. Select a new number")

	def __process_number(self, idx: Union[Index | int], number: int | np.int32, n_category: DataFrame):
		if isinstance(number, np.int32):
			number = int(number)

		if number in self._selected_numbers:
			raise ValueError("The number already exists in the list")
		
		residue = number % 2

		self.__update_category(residue == 0, "_even", idx)
		self.__update_category(residue != 0, "_odd", idx)

		self.__update_category(number in self.euromillions.low_numbers, "_low", idx)
		self.__update_category(number in self.euromillions.high_numbers, "_high", idx)

		self._selected_numbers.append(number)
		self.__sum_selected_number(number)
		self.__remove_number(number, n_category)

	def __write_number(self, idx: Union[Index | int], number: Union[np.int32,list,np.ndarray,int], n_category: DataFrame):
		if isinstance(number, np.ndarray) or isinstance(number, list):
			for num in number:
				self.__process_number(num,n_category)
		else:
			self.__process_number(idx, number,n_category)

	def __check_number(self, number: int):
		idx_list = self.euromillions.last_draw.loc[self.euromillions.last_draw['number'] == number, 'skips'].tolist()
		idx = next(iter(idx_list), None)

		if idx is None:
			raise ValueError("The list is empty. Please, select a number again")
		
		assert isinstance(idx,int), "The answer is not a number. Please, select a new number/index"

		self.__check_idx(idx)

		if isinstance(idx, int):
			return 
		else:
			raise ValueError("There is no validation for the selected number. Please, select a new number")

	def __check_idx(self, idx: int, max_occurrences: int = 2, removal: bool = None):
		if removal:
			if idx in self._idx_dy:
				self._idx_dy[idx] -= 1
				if self._idx_dy[idx] <= 0:
					del self._idx_dy[idx]
			return 
		
		if idx in self._idx_dy:
			self._idx_dy[idx] += 1
		else:
			self._idx_dy[idx] = 1

		if self._increase == 1:
			self._idx_dy[idx] += 1
		elif self._increase == 2:
			self._idx_dy[idx] += 2

		count_reached = sum(1 for v in self._idx_dy.values() if v >= max_occurrences)

		if count_reached >= 2 and self._idx_dy[idx] > max_occurrences:
			self.__check_idx(idx, removal=True)
			raise ValueError(f"The index {idx} has been selected {max_occurrences} times. A new index must be selected")


	def __validation(self, type: Literal["idx","int"], number: Union[pd.Index, int]) -> None:
		assert type in ("idx","int"), "No valid string validation registered"

		self._increase = 1
		if type == "idx":
			try:
				self.__check_idx(number)
			except ValueError:
				raise ValueError(f"No index validation success")

		if type == "int":
			try:
				self.__check_number(number)
			except ValueError:
				raise ValueError(f"No number validation success")

	def __operation(self, available_numbers: list, n_category: DataFrame):
		match self.idx:
			case 0:
				rng = 1
			case _:
				rng = random.randint(1, 2)
		
		self._increase = rng
		if rng == 2 and len(available_numbers) < 2:
			selected_number = np.random.choice(available_numbers, size=1, replace=False)[0]
			self.__validation("idx", self.idx)
			self.__write_number(self.idx, selected_number, n_category)

		elif rng == 2 and len(available_numbers) >= 2 and self.rng_count == 0:
			if self._recommended_numbers_selected < 5:
				selected_numbers = np.random.choice(available_numbers, size=rng, replace=False)
				self.__validation("idx", self.idx)
				self.__write_number(selected_numbers, n_category)
				self.rng_count += 1
			else:
				selected_number = np.random.choice(available_numbers,size=1,replace=False)[0]
				self.__validation("idx",self.idx)
				self.__write_number(selected_number,n_category)

		elif rng == 2 and self.rng_count > 0:
			selected_number = np.random.choice(available_numbers,size=1,replace=False)[0]
			self.__validation("idx", self.idx)
			self.__write_number(selected_number,n_category)

		elif rng == 1 and len(available_numbers) > 0:
			selected_number = np.random.choice(available_numbers,size=rng,replace=False)[0]
			self.__validation("idx", self.idx)
			self.__write_number(selected_number,n_category)

	def __list_of_numbers(self, n_category: DataFrame) -> tuple[list,DataFrame]:
		assert self.numbers is not None and not self.numbers.empty, "There are no available numbers to select. Please check the DataFrame from the last_draw."

		# We locate the row. The list will be created with the numbers in this location
		row = self.numbers.loc[0] if self.idx == 0 else self.numbers.loc[self.idx]

		# Search if the number is in the DataFrame that determines the category
		if len(row) == 1:
			available_numbers = n_category[n_category['numbers'].isin(row)]['numbers'].tolist()
		elif len(row) > 1:
			exploded_row = pd.Series(row).explode().reset_index(drop=True)
			available_numbers = n_category[n_category['numbers'].isin(exploded_row)]['numbers'].tolist()

		if available_numbers:
			return available_numbers
		elif not available_numbers and self.idx > 18:
			self._cold -= 1
			raise ValueError("The list of numbers to be selected is empty.")
		else:
			raise ValueError("The list of numbers to be selected is empty.")

	def __select_number(self, n_category: DataFrame) -> int:
		numbers = self.__list_of_numbers(n_category)
		self.__operation(numbers, self.idx, n_category)

	def __select_skip(self, zero_selection: bool = None) -> int:
		# Filters the indexes following the condition of zero_selection
		skips = self.df_skips[self.df_skips['7'] == 0].index if zero_selection else self.df_skips[self.df_skips['7'] != 0].index

		# Enstablish the max number of the indexes to be selected
		allowed_max_idx = 18 if len(self.euromillions.last_draw[self.euromillions.last_draw['skips'] <= 12]) >= 39 else 100

		idx = np.random.choice(skips, size=1, replace=False)[0]

		if (idx > allowed_max_idx or (idx < allowed_max_idx and idx > 18)) and self._cold == 0:
			self._cold += 1

		if allowed_max_idx == 100 and idx > 18 and self._cold > 0:
			skips = [i for i in skips if i < 19]
			idx = np.random.choice(skips, size=1, replace=False)[0]
		
		return idx

	def distribution_evaluation(self):
		self.distribution = self.euromillions.evaluation.apply(lambda column: float(Decimal(column.mean()).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)), axis=0).to_frame().T

	def first_number(self):
		try:
			self.__select_number(0,self.recommended_numbers)
		except ValueError:
			self.__select_number(0,self.not_recommended_numbers)

	def suggested_numbers(self):
		zero_selected = 0

		while self._recommended_numbers_selected < 6:
			if zero_selected < 3:
				self.idx = self.__select_skip(True)
			else:
				idx = self.__select_skip()

			count = 0
			while True:
				if count < 5:
					try:
						self.__select_number(self.idx, self.recommended_numbers)
						break
					except ValueError:
						count += 1
						self.idx = self.__select_skip(zero_selection=(zero_selected < 3))
				else:
					try:
						available_numbers = self.recommended_numbers["numbers"].unique().tolist()
						selected_number = np.random.choice(available_numbers,size=1,replace=False)[0]
						if self.__validation("int",selected_number):
							break
					except ValueError:
						continue

					self.__write_number(selected_number,self.recommended_numbers)
					break

			zero_selected += 1 if (zero_selected < 3) else 0

		while self._not_recommended_numbers_selected < 4:
			idx = self.__select_skip()
			while True:
				try:
					self.__select_number(idx, self.not_recommended_numbers)
					break
				except ValueError:
					idx = self.__select_skip()
