"""MAIN FUNCTIONS WITH OOP PARADIGM FOR DATA ANALYSIS. IN THIS CASE, EURO MILLIONS LOTTERY GAME ANALYSIS"""

# Standard Libraries of Python
import random
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP, getcontext
getcontext().prec = 5

# Dependencies
import pandas as pd
import numpy as np
from pandas import DataFrame
from pandas import Series
np.set_printoptions(precision=5)

# Libraries made for this Proyect
from data.cleaning import database

def draw_generator(size:int) -> int:
    for draw in range(12,size):
        yield draw

def games_7(column:Series) -> pd.DataFrame.columns:
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

def games_12(column:Series) -> pd.DataFrame.columns:
    games_dict = {
        8: 0.65,
        9: 0.55,
        10: 0.45,
        11: 0.35,
        12: 0.25
    }
    return games_dict.get(column,0)

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

# Main Object - Super Class
class Analysis:
    def __init__(self, is_star: bool = False) -> object | DataFrame | range:
        # Little temporary line to scrap or not scrap the website, in order to read the database directly from .parquet file
        answer = int(input("Do you wish to scrap the database directly from the website? (Enter 1 for yes, 0 for no): "))

        if answer == 1:
            # Perform further testing with scraping
            self.scrap = database()
        else:
            # Proceed without scraping and testing with .parquet file
            self.scrap = pd.read_parquet('data/files/db.parquet')

        self.df = self.scrap.copy()
        self.df_stars = self.df.copy()

        # Modification of original DataFrame to contain the record of stars at the beginning of a certain draw, becuase of a change in the rules of the game. Check cleaning file.
        self.df_stars = self.df.drop(
            columns=['nro1','nro2','nro3','nro4','nro5'],
            index=range(0,863)
        )

        dates_construct = self.df.copy()

        # Change in date column for future apply function
        dates_construct['dates'] = dates_construct['dates'].dt.year

        # Change in the whole analisys class. This is established at the beginning of it, with the __init__ bool argument
        if not is_star:
            self.dates_construct = dates_construct.drop(
                columns=['draw','star_1','star_2']
            )
        else:
            self.dates_construct = dates_construct.drop(
                columns=['draw','nro1','nro2','nro3','nro4','nro5'],
                index=range(0,863)
            )
        
        # Ranges to syntetize the rest of the code, for more readability
        self._numbers = range(1,51)
        self._stars = range(1,13)
        self._draws = range(1,len(self.df)+1)
        self._draws_stars = range(864,len(self.df)+1)
        self._col_df = range(1,6) # Selection of the first (5) numbers. First columns of the main DataFrame self.scrap
        self._col_stars_df = range(1,3) # Selection of the two (2) stars. Last two (2) columns of the main DataFrame self.scrap

    @property
    def db(self) -> DataFrame:
        return self.df
    
    @db.setter
    def db(self, modified_data: DataFrame, is_star: bool = False) -> DataFrame:

        assert modified_data is not None and isinstance(modified_data,DataFrame), "modified_data must be a DataFrame"

        # DataFrame with part of the information, in order to execute the loop to determine de % of success of method
        self.df = modified_data

        # Modifing the range from __init__
        self.df.index = (range(1,len(modified_data)+1))
        self._draws = range(1,len(self.df)+1)

        dates_construct = self.df.copy()
        dates_construct['dates'] = dates_construct['dates'].dt.year

        if not is_star:
            self.dates_construct = dates_construct.drop(
                columns=['draw','star_1','star_2']
            )
        else:
            self.dates_construct = dates_construct.drop(
                columns=['draw','nro1','nro2','nro3','nro4','nro5'],
                index=range(0,863)
            )

    @Memoize
    def groups_info(self) -> DataFrame:
        winning_numbers = self.df.drop(
            columns=['dates','star_1','star_2']
        )

        groups = [list(range(i,i+10)) for i in range(1,51,10)]
        group_names = [tuple(range(i,i+10)) for i in range(1,51,10)]
        results = {i: {
                group_name: sum([1 for num in row if num in group])
                for group_name,group in zip(group_names,groups)
            }
            for i,row in winning_numbers.iterrows()
        }
        self.groups = pd.DataFrame.from_dict(
            results,
            orient='index'
        )
        self.groups.columns = [f'{i}_to_{i+9}' for i in range(1,51,10)]
        
        present_sg_10 = self.groups.iloc[-10:]
        future_sg_10 = self.groups.iloc[-9:]

        self.present_groups = pd.DataFrame({
            '10_games': (present_sg_10 > 0).sum()}
        ).T
        self.future_groups = pd.DataFrame({
            '10_games': (future_sg_10 > 0).sum()}
        ).T

    def __control_condition(self, is_star: bool = False) -> DataFrame | range:
        # Function to improve readability of the code, by simplify the DataFrame and ranges in each particular case
        if not is_star:
            self._df = self.df # DataFrame
            self._col = self._numbers # Range
            self._id = self._draws # Range
            self._col_data = self._col_df # Range
        else:
            self._df = self.df_stars # DataFrame
            self._col = self._stars # Range
            self._id = self._draws_stars # Range
            self._col_data = self._col_stars_df # Range
    
    def __transformation_into_columns(self, row: pd.Index) -> DataFrame:
        for draw in range(1,6):
            if not np.isnan(self.year_history.loc[row.dates,row[f'nro{draw}']]):
                self.year_history.loc[row.dates,row[f'nro{draw}']] += 1
            else:
                self.year_history.loc[row.dates,row[f'nro{draw}']] = 1

    def __stars_transformation_into_columns(self, row: pd.Index) -> DataFrame:
        for draw in range(1,3):
            if not np.isnan(self.year_history.loc[row.dates,row[f'star_{draw}']]):
                self.year_history.loc[row.dates,row[f'star_{draw}']] += 1
            else:
                self.year_history.loc[row.dates,row[f'star_{draw}']] = 1

    @Memoize
    def apply_transformation(self, is_star: bool = False) -> DataFrame:
        self.__control_condition(is_star)
        self.year_history = pd.DataFrame(columns=self._col,
            index=np.arange(
                self.dates_construct['dates'].iloc[0],
                self.dates_construct['dates'].iloc[-1] + 1
            )
        )

        if not is_star:
            self.dates_construct.apply(
                self.__transformation_into_columns,axis=1
            )
        else:
            self.dates_construct.apply(
                self.__stars_transformation_into_columns,axis=1
            )

        self.year_history.fillna(0,inplace=True)

        self.hits = self.year_history.sum().to_frame().rename(
            columns={0: 'hits'}
        ).T.astype('int32')

        self.mean = self.year_history.mean().to_frame().rename(
            columns={0: 'average'}
        ).T.astype('float32')

        self.median = self.year_history.median().to_frame().rename(
            columns={0: 'median'}
        ).T.astype('float32')

    @Memoize
    def __numbers_boolean(self, is_star: bool = False) -> DataFrame:
        self.__control_condition(is_star)
        self.booleans_df = pd.DataFrame(False,columns=self._col,
            index=self._id
        )
        for i in self._col_data:
            if not is_star:
                col_name = f'nro{i}'
            else:
                col_name = f'star_{i}'
            self.booleans_df = self.booleans_df | (
                self._df[col_name].to_numpy()
                [:, None] == self._col
            )
    
    @Memoize
    def count_skips(self, is_star: bool = False) -> DataFrame:
        self.__numbers_boolean(is_star)
        mask = self.booleans_df == 0
        reset_mask = self.booleans_df == 1
        cumulative_sum = np.cumsum(mask)
        cumulative_sum[reset_mask] = 0
        result = np.where(self.booleans_df == 0, 1, cumulative_sum)
        result = pd.DataFrame(
            result,
            index=self._id,
            columns=self._col
        )
        df_t = result != 0
        self.counts = df_t.cumsum()-df_t.cumsum().where(~df_t).ffill().fillna(0).astype(int)

    def skips_for_last_12_draws(self) -> DataFrame:
        self.skips = range(0,19)
        if len(self.counts) - 11 == 1:
            last_12_draws = range(2,len(self.counts))
        else:
            last_12_draws = range(len(self.counts) - 11,len(self.counts) + 1)

        aus_12 = [
            self.counts.loc[i-1,int(column)] 
            for i in last_12_draws[0:12]
            for column in self.counts if self.counts.loc[i,int(column)] == 0
        ]
        
        counter_l_7 = [Counter(aus_12[25:60]).get(i,0) for i in self.skips]
        counter_l_12 = [Counter(aus_12).get(i,0) for i in self.skips]
        self.skips_7_12 = pd.DataFrame(
                {
                '7': counter_l_7,
                '12': counter_l_12
            }
        )

    def __total_average_hits(self, is_star: bool = False, aprox: bool = False):
        self.__control_condition(is_star)
        divide = 2 if is_star else 5
        self.average = self.hits.apply(lambda hits: hits / len(self._df) / divide).iloc[0]
        if aprox:
            return Decimal(self.average.sum()) / Decimal(len(self._col)) + Decimal(0.001)
        else:
            return Decimal(self.average.sum()) / Decimal(len(self._col))

    def m_hits(self,is_star: bool = False, aprox: bool = False):
        min_hits = self.__total_average_hits(is_star,aprox)
        return min_hits * Decimal(int(self.hits.iloc[0,0])) / Decimal(float(self.average.iat[0]))

    def __natural_rotation(self,is_star: bool = False,aprox: bool = False):
        self.__control_condition(is_star)
        self.__total_average_hits(is_star,aprox)
        rotation = pd.DataFrame(
            {'hits': self.hits.iloc[0],
            'average_of_numbers': self.average,
            'total_average': self.__total_average_hits(is_star,aprox),
            'minimal_hits_needed': self.m_hits(is_star,aprox)},
            index=self._col)
        
        rotation['difference'] = rotation['hits'] - rotation['minimal_hits_needed']
        return rotation

    def get_natural_rotations(self, is_star: bool = False):
        self.exact_rotation = self.__natural_rotation(is_star,aprox=False)
        self.aprox_rotation = self.__natural_rotation(is_star,aprox=True)

    def numbers_clasification(self, is_star: bool = False):
        self.__control_condition(is_star)
        self.best_numbers = self.aprox_rotation.loc[
            self.aprox_rotation['hits'] > self.aprox_rotation['minimal_hits_needed']
            ].index.to_numpy()
        self.normal_numbers = self.exact_rotation.loc[
            (self.exact_rotation['hits'] > self.exact_rotation['minimal_hits_needed'])
            & ~(self.exact_rotation.index.isin(self.best_numbers))
            ].index.to_numpy()
        self.category = {number: 0 for number in self._col}
        for number in self.best_numbers:
            self.category[number] = 2
        for number in self.normal_numbers:
            self.category[number] = 1

# Sub class for numbers analysis
class Criteria(Analysis):
    def __skips_last_draws(self):
        last_draw = self.counts.iloc[-1].sort_values().to_frame()
        self.last_draw = last_draw.reset_index().rename(
            columns={'index':'number',
                    len(self.df):'skips'
                }
            )

    def year_criterion(self):
        current_year = self.dates_construct['dates'].iloc[-1]
        year_criteria = {key: [] for key in self._numbers}

        for number in self._numbers:
            x = self.year_history.at[current_year,number]
            number_median = self.median.at['median',number]
            number_mean = self.mean.at['average',number]
            if number_mean != 0 and not np.isnan(number_mean):
                if x == 0 or x <= number_median / 2:
                    y = round((1 - (((number_median / 2) * 100) / number_mean) / 100),2)
                else:
                    x_percentage = round((x * 100 / number_mean) / 100,2)
                    y = round((1 - x_percentage if x_percentage > 1 else 1 - x_percentage),2)
            else:
                y = 0.50

            year_criteria[number] = float(y)

        self.years_criteria = pd.DataFrame.from_dict(
            year_criteria,
            orient='index',
            columns=['year_criteria']
        )
    
    def rotation_criterion(self):
        current_hits_needed = super().m_hits(aprox=True)

        self.rotation = next(
            (draw for draw in range(len(self.df['draw'])+1,len(self.df['draw'])+10) 
            if draw * current_hits_needed/len(self.df)>
            int(current_hits_needed)+1),None
        )
        rotation_criteria = {}
        for number in self._numbers:
            category = self.category[number]
            diff_exact = self.exact_rotation.at[number,'difference']
            diff_aprox = self.aprox_rotation.at[number,'difference']
            if category == 1 and diff_exact > 0:
                rotation_criteria[number] = 0.50
            elif category == 1 and -1 < diff_exact <= 0.60:
                cr = 0.50
                x = (abs(diff_exact) + (Decimal(str(cr)) * Decimal('100')) / Decimal(str(cr))) / Decimal('100') + Decimal(str(cr))
                rotation_criteria[number] = float(x.quantize(Decimal('0.01'),rounding=ROUND_HALF_UP))
            elif category == 2 and diff_aprox > 0:
                rotation_criteria[number] = 1
                cr = 1.00
            elif category == 2 and diff_aprox > -1 and diff_aprox < 0.60 and len(self.best_numbers) < 17:
                x = (abs(diff_exact) * Decimal('100')) / Decimal(str(cr)) / Decimal('100') + Decimal(str(cr))
                rotation_criteria[number] = float(x.quantize(Decimal('0.01'),rounding=ROUND_HALF_UP))
            else:
                rotation_criteria[number] = 0.25
        
        self.rotation_criteria = pd.DataFrame.from_dict(
            rotation_criteria,
            orient='index',
            columns=['rotation_criteria']
        )

    def position_criterion(self):
        self.__skips_last_draws()
        self.skips_7_12[['values_7','values_12']] = self.skips_7_12[['7','12']].applymap(
            lambda x: games_7(x) if x <= 7 
            else games_12(x)
        )

        position_criteria = {number: [] for number in self._numbers}

        last_game = 0.50
        if self.skips_7_12.iloc[0,0] <= 5:
            self.skips_7_12.iloc[0,2] = last_game
        
        for i, key in zip(self.last_draw.iloc[:,1],self.last_draw.iloc[:,0]):
            value_added = None
            for e in range(len(self.skips_7_12)):
                if self.skips_7_12.iloc[e,0] <= 7 and e == i:
                    position_criteria[int(key)] = self.skips_7_12.at[e,'values_7']
                    value_added = True
                elif self.skips_7_12.iloc[e,0] >= 8 and e == i:
                    position_criteria[int(key)] = self.skips_7_12.at[e,'values_12']
                    value_added = True
            if not value_added:
                if len(self.last_draw.loc[self.last_draw['skips'] <= 12]) <= 38:
                    position_criteria[int(key)] = 1
                else:
                    position_criteria[int(key)] = 0.25
        
        self.position_criteria = pd.DataFrame.from_dict(
            position_criteria,
            orient='index',
            columns=['position_criteria']
        )
        
        self.skips_7_12.drop(self.skips_7_12.iloc[:,[2,3]],axis=1,inplace=True)
    
    def group_criterion(self):
        groups = [list(range(i,i+10)) for i in range(1,51,10)]
        self.future_groups.columns = [1,2,3,4,5]
        gp10 = {}
        gp_values = {
            0: 1,
            1: 1,
            2: 1,
            3: 1,
            4: 1,
            5: 1,
            6: 0.85,
            7: 0.75,
            8: 0.650,
            9: 0.6,
            10: 0.5
        }
        for column,gp in self.future_groups.loc['10_games'].items():
            gp_range = tuple(range(1+10*(column-1),11+10*(column-1)))
            gp10[gp_range] = gp_values.get(gp,0)
        
        gp10_df = pd.DataFrame.from_dict(
            gp10,
            orient='index'
        ).T
        
        gp10_df = gp10_df.rename(index={0: 'criteria'})
        gp10_df.columns = [1,2,3,4,5]
        group_criteria = {number:[] for number in self._numbers}

        for index,group in enumerate(groups):
            gn = group
            for number in self._numbers:
                if number in gn:
                    group_criteria[number] = gp10_df.iloc[0,index]
        
        self.group_criteria = pd.DataFrame.from_dict(
            group_criteria,
            orient='index',
            columns=['group_criteria']
        )
    
    def numbers_of_tomorrow(self):
        df_numbers = pd.concat(
            [
                self.years_criteria,
                self.rotation_criteria,
                self.position_criteria,
                self.group_criteria
            ],
            axis=1
        )
        df_numbers['criteria'] = df_numbers.apply(
            lambda row: row['year_criteria'] +
            row['rotation_criteria'] + 
            row['group_criteria'] + 
            row['position_criteria'],
            axis=1
        )

        tomorrow_numbers = df_numbers[['criteria']].sort_values(by='criteria',ascending=False)
        criterion = tomorrow_numbers['criteria'].median()

        self.recommended_numbers = tomorrow_numbers.loc[tomorrow_numbers['criteria'] >= criterion].reset_index().rename(
            columns={
                'index':'numbers'
            }
        )
        self.not_recommended_numbers = tomorrow_numbers.loc[tomorrow_numbers['criteria'] < criterion].reset_index().rename(
            columns={
                'index':'numbers'
            }
        )

# Creation of the recommended tickets
class Tickets():
    def __init__(self, euromillions: Criteria) -> object | DataFrame | int:
        # Inherits all the instances of Euro Millions
        self.euromillions = euromillions

        # List of numbers to be selected
        self.numbers = self.__df_numbers()

        # Counter of control, to prevent the selection of two number in the same row of self.numbers, more than two times
        self._s_n = 0

        # Counter for recommended numbers
        self._recommended_numbers_selected = 0

        # Counter for not recommended numbers
        self._not_recommended_numbers_selected = 0

        # List to store the selected numbers
        self._selected_numbers = []

    def __df_numbers(self) -> DataFrame:
        self._df_values = self.euromillions.last_draw.groupby('skips')['number'].apply(lambda x: list(x)).reset_index().set_index('skips')
        self._df_values = self._df_values.rename_axis('number').rename_axis(None)
        self._df_values['number'] = self._df_values['number'].apply(lambda x: sorted(x))
        return self._df_values['number']

    def draw_skips(self) -> DataFrame:
        # DataFrame to be populated
        self.d_skips = pd.DataFrame(columns=['nro1','nro2','nro3','nro4','nro5'])

        # Index to be updated. Row to be analize
        for index, row in self.euromillions.counts.iterrows():
            # Row to be populated
            new_row = []

            for column, value in row.items():
                # We search the value in each column of self.euromillions.counts, within the row of reference
                if value == 0:
                    try:
                        aus = self.euromillions.counts.loc[index - 1, column]
                    except (KeyError,IndexError):
                        aus = 0
                    if pd.isna(aus):
                        aus = 0
                    new_row.append(aus)
        
            while len(new_row) < 5:
                new_row.append(0)
        
            self.d_skips.loc[index] = new_row
        
        return self.d_skips
        
    def skips_evaluation(self) -> DataFrame:
        self.evaluation = pd.DataFrame(columns=['0','5','7','10','13'])
        counts = pd.DataFrame(0, index=self.d_skips.index, columns=self.evaluation.columns)

        counts['0'] = self.d_skips.apply(lambda row: row.eq(0).sum(),axis=1)
        counts['5'] = self.d_skips.apply(lambda row: row.between(0,5).sum(),axis=1)
        counts['7'] = self.d_skips.apply(lambda row: row.between(0,7).sum(),axis=1)
        counts['10'] = self.d_skips.apply(lambda row: row.between(0,10).sum(),axis=1)
        counts['13'] = self.d_skips.apply(lambda row: row.between(0,13).sum(),axis=1)

        self.evaluation = pd.concat([self.evaluation,counts],ignore_index=True)
        self.evaluation = self.evaluation.set_index(pd.RangeIndex(1, len(self.evaluation) + 1))

        return self.evaluation

    def __list_of_numbers(self, idx: pd.Index, n_category: DataFrame) -> list | DataFrame:
        assert self.numbers is not None and not self.numbers.empty, "There are no available numbers to select. Please check the DataFrame from the last draw."

        row = self.numbers.loc[0] if idx == 0 else self.numbers.loc[idx]
        available_numbers = n_category[n_category['numbers'].isin(row)]['numbers'].tolist()

        if idx == 0 and self._s_n > 0:
            raise ValueError("Idx 0 was already selected. Please, put another index.")
        elif available_numbers:
            return available_numbers, n_category
        elif not available_numbers:
            not_n_category = self.euromillions.not_recommended_numbers
            available_numbers = not_n_category[not_n_category['numbers'].isin(row)]['numbers'].tolist()
            return available_numbers, not_n_category
        else:
            raise ValueError("The list of numbers to be selected is empty.")
    
    def __select_number(self, idx: pd.Index, n_category: DataFrame) -> int:
        numbers, n_category = self.__list_of_numbers(idx,n_category)
        self.__operation(numbers,idx,n_category)

    def __operation(self, available_numbers: list, idx: pd.Index, n_category: DataFrame) -> int:
        match idx:
            case 0:
                rng = 1
            case _:
                if self._s_n > 0 and self._s_n <= 2:
                    rng = random.randint(1,2)
                else:
                    rng = 0

        if rng == 2 and len(available_numbers) > 2:
            selected_numbers = np.random.choice(available_numbers,size=rng,replace=False)
            self._selected_numbers.extend(selected_numbers)
            self._s_n += rng
            self.__remove_number(selected_numbers,n_category)
            self.__sum_selected_number(selected_numbers)
        elif rng == 1 or (rng == 0 and len(available_numbers) > 0):
            selected_number = np.random.choice(available_numbers,replace=False)
            self._selected_numbers.append(selected_number)
            self._s_n += 1
            self.__remove_number(selected_number,n_category)
            self.__sum_selected_number(selected_number)
    
    def __remove_number(self, value: np.int32 | list, n_category: DataFrame) -> DataFrame:
        if n_category.equals(self.euromillions.recommended_numbers):
            self.euromillions.recommended_numbers = self.euromillions.recommended_numbers.drop(
                self.euromillions.recommended_numbers.loc[self.euromillions.recommended_numbers['numbers'] == value].index).reset_index(drop=True)
            self.euromillions.recommended_numbers = self.__probability(self.euromillions.recommended_numbers)
            return self.euromillions.recommended_numbers
        else:
            self.euromillions.not_recommended_numbers = self.euromillions.not_recommended_numbers.drop(
                self.euromillions.not_recommended_numbers.loc[self.euromillions.not_recommended_numbers['numbers'] == value].index).reset_index(drop=True)
            self.euromillions.not_recommended_numbers = self.__probability(self.euromillions.not_recommended_numbers)
            return self.euromillions.not_recommended_numbers
    
    def __probability(self, n_category: DataFrame) -> DataFrame:
        probability = 1 / len(n_category)
        n_category['criteria'] = n_category['criteria'] * (1 + probability)
        return n_category

    def __sum_selected_number(self, selected_number: int | list) -> int:
        if isinstance(selected_number,int):
            if self.euromillions.recommended_numbers[
                self.euromillions.recommended_numbers['numbers'] == selected_number]['numbers'].tolist():
                self._recommended_numbers_selected += 1
            else:
                self._not_recommended_numbers_selected += 1
        if isinstance(selected_number,list):
            if self.euromillions.recommended_numbers[
                self.euromillions.recommended_numbers['numbers'].isin(selected_number)]['numbers'].tolist():
                self._recommended_numbers_selected += 1
            else:
                self._not_recommended_numbers_selected += 1
    
    def first_number(self) -> int:
        self.__select_number(0,self.euromillions.recommended_numbers)
    
    def suggested_numbers(self) -> list:
        zero_selected = 0
        skips = self.euromillions.skips_7_12[self.euromillions.skips_7_12['7'] == 0].index.unique().tolist()

        try:
            for idx in skips:
                self._df_values.loc[idx]
        except KeyError:
            for idx in skips:
                print(f"the index {idx} will be deleted from {self.euromillions.skips_7_12}")
                self.euromillions.skips_7_12.drop(index=idx,inplace=True)

        while self._recommended_numbers_selected < 6:
            if zero_selected == 0:
                idx = np.random.choice(skips,size=1,replase=False)
                self.__select_number(idx,self.euromillions.recommended_numbers)
            try:
                self.__select_number(idx, self.euromillions.recommended_numbers)
            except ValueError:
                while idx == 0:
                    idx = np.random.choice(skips,size=1,replace=False)
                self.__select_number(idx,self.euromillions.recommended_numbers)

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