# Standard Libraries of Python
import random
from typing import Union, Literal

# Dependencies
import pandas as pd
import numpy as np
from pandas import DataFrame
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
        # Store an instance of Euro Millions (Criteria) for access to its attributes and methods
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
        self._idx_list = []
        self.rng_count = 0

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
    
    def __process_number(self, number: int | np.int32, n_category: DataFrame):
        if isinstance(number, np.int32):
            number = int(number)
        
        if number in self._selected_numbers:
            raise ValueError("The number already exists in the list")
        
        self._selected_numbers.append(number)
        self.__sum_selected_number(number)
        self.__remove_number(number, n_category)

    def __write_number(self, number: Union[np.int32,list,np.ndarray,int], n_category: DataFrame):
        if isinstance(number, np.ndarray) or isinstance(number, list):
            for num in number:
                self.__process_number(num,n_category)
        else:
            self.__process_number(number,n_category)
    
    def __check_number(self, number: int) -> int:
        idx = self.euromillions.last_draw.loc[self.euromillions.last_draw['number'] == number, 'skips'].tolist()

        if idx:
            result = idx[0]
            assert isinstance(result,int), "There is no number as an answer. Please, select a new number"
        else:
            raise ValueError("The list is empty. Please, select a number again")

        i = self.__check_idx(result,1)

        if isinstance(i,int):
            return number
        else:
            raise ValueError("There is no validation for the selected number. Please, select a new number")

    def __check_idx(self, idx: int, increase: int, max_occurrences: int = 2) -> int:
        match increase:
            case 1:
                if idx in self._idx_dy:
                    self._idx_dy[idx] += 1
                else:
                    self._idx_dy[idx] = 1
            case 2:
                if idx in self._idx_dy:
                    self._idx_dy[idx] += 2
                else:
                    self._idx_dy[idx] = 2

        v_count = 2
        counter = sum(1 for value in self._idx_dy.values() if value == v_count)

        if counter > 2:
            keys = [k for k,v in self._idx_dy.items() if v == counter]
            ind = keys.index(idx)
            raise ValueError(f"The index {ind} has been selected {max_occurrences} times. A new index must be selected")
        else:
            self._idx_list.append(idx)
            return idx

    def __validation(self, type: Literal["idx","int"], number: Union[pd.Index,int], increase: int = 1) -> pd.Index|int:
        assert type in ("idx","int"), "No valid string validation registered"

        if type == "idx":
            try:
                self.__check_idx(number, increase)
                return number
            except ValueError:
                raise ValueError(f"No index validation success")
        
        if type == "int":
            try:
                result = self.__check_number(number)
                return result
            except ValueError:
                raise ValueError(f"No number validation success")

    def __operation(self, available_numbers: list, idx: pd.Index | int, n_category: DataFrame) -> int:
        match idx:
            case 0:
                rng = 1
            case _:
                rng = random.randint(1, 2)

        if rng == 2 and len(available_numbers) < 2:
            selected_number = np.random.choice(available_numbers,size=1,replace=False)[0]
            self.__validation("idx",idx,1)
            self.__write_number(selected_number,n_category)

        elif rng == 2 and len(available_numbers) >= 2 and self.rng_count == 0:
            if self._recommended_numbers_selected < 5:
                selected_numbers = np.random.choice(available_numbers,size=rng,replace=False)
                self.__validation("idx",idx,rng)
                self.__write_number(selected_numbers,n_category)
                self.rng_count += 1
            else:
                selected_number = np.random.choice(available_numbers,size=1,replace=False)[0]
                self.__validation("idx",idx,1)
                self.__write_number(selected_number,n_category)

        elif rng == 2 and self.rng_count > 0:
            selected_number = np.random.choice(available_numbers,size=1,replace=False)[0]
            self.__validation("idx",idx,1)
            self.__write_number(selected_number,n_category)

        elif rng == 1 and len(available_numbers) > 0:
            selected_number = np.random.choice(available_numbers,size=rng,replace=False)[0]
            self.__validation("idx",idx,1)
            self.__write_number(selected_number,n_category)

    def __list_of_numbers(self, idx: pd.Index | int, n_category: DataFrame) -> tuple[list,DataFrame]:
        assert self.numbers is not None and not self.numbers.empty, "There are no available numbers to select. Please check the DataFrame from the last_draw."

        # We locate the row. The list will be created with the numbers in this location
        row = self.numbers.loc[0] if idx == 0 else self.numbers.loc[idx]

        # Search if the number is in the DataFrame that determines the category
        if len(row) == 1:
            available_numbers = n_category[n_category['numbers'].isin(row)]['numbers'].tolist()
        elif len(row) > 1:
            exploded_row = pd.Series(row).explode().reset_index(drop=True)
            available_numbers = n_category[n_category['numbers'].isin(exploded_row)]['numbers'].tolist()

        if available_numbers:
            return available_numbers, n_category
        elif not available_numbers and idx > 18:
            self._cold -= 1
            raise ValueError("The list of numbers to be selected is empty.")
        else:
            raise ValueError("The list of numbers to be selected is empty.")

    def __select_number(self, idx: pd.Index | int, n_category: DataFrame) -> int:
        numbers, n_category = self.__list_of_numbers(idx,n_category)
        self.__operation(numbers,idx,n_category)

    def __select_skip(self, zero_selection: bool = None) -> int:
        # Filters the indexes following the condition of zero_selection
        skips = self.df_skips[self.df_skips['7'] == 0].index if zero_selection else self.df_skips[self.df_skips['7'] != 0].index

        # Enstablish the max number of the indexes to be selected
        allowed_max_idx = 18 if len(self.euromillions.last_draw[self.euromillions.last_draw['skips'] <= 12]) >= 39 else 100

        idx = np.random.choice(skips, size=1, replace=False)[0]

        if (idx > allowed_max_idx or (idx < allowed_max_idx and idx > 18)) and self._cold == 0:
            self._cold += 1
            return idx

        if allowed_max_idx == 100 and idx > 18 and self._cold > 0:
            skips = [i for i in skips if i < 19]
            idx = np.random.choice(skips, size=1, replace=False)[0]
            return idx
        else:
            return idx

    def first_number(self):
        try:
            self.__select_number(0,self.recommended_numbers)
        except ValueError:
            self.__select_number(0,self.not_recommended_numbers)
    
    def suggested_numbers(self):
        zero_selected = 0

        while self._recommended_numbers_selected < 6:
            if zero_selected < 3:
                idx = self.__select_skip(True)
            else:
                idx = self.__select_skip()

            count = 0
            while True:
                if count < 5:
                    try:
                        self.__select_number(idx, self.recommended_numbers)
                        break
                    except ValueError:
                        count += 1
                        idx = self.__select_skip(zero_selection=(zero_selected < 3))
                else:
                    try:
                        available_numbers = self.recommended_numbers["numbers"].unique().tolist()
                        selected_number = np.random.choice(available_numbers,size=1,replace=False)[0]
                        if self.__validation("int",selected_number) == int:
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