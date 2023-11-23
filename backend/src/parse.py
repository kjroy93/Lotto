"""MAIN FUNCTIONS WITH OOP PARADIGM FOR DATA ANALYSIS. IN THIS CASE, EURO MILLIONS LOTTERY GAME ANALYSIS"""

# Standard Libraries of Python
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP, getcontext
getcontext().prec = 5

# Dependencies
import pandas as pd
import numpy as np
from pandas import DataFrame,Index,Series
np.set_printoptions(precision=5)

# Libraries made for this Proyect
from .functions import games_7, games_12, Memoize
from database.structure import database

# Main Object - Super Class
class Analysis:
    def __init__(self, is_star: bool = False):
        # Little temporary line to scrap or not scrap the website, in order to read the database directly from .parquet file
        answer = int(input("Do you wish to scrap the database directly from the website? (Enter 1 for yes, 0 for no): "))

        if answer == 1:
            # Perform further testing with scraping
            self.scrap = database()
        
        else:
            # Proceed without scraping and testing with .parquet file
            self.scrap = pd.read_parquet('database/files/db.parquet')

        self.df = self.scrap.copy()
        self.df_stars = self.df.copy()

        # Modification of original DataFrame to contain the record of stars at the beginning of a certain draw.
        # Change in the rules of the game. Check cleaning file.
        self.df_stars = self.df.drop(
            columns=['nro1','nro2','nro3','nro4','nro5'],
            index=range(0,863)
        )

        dates_construct = self.df.copy()
        dates_construct['dates'] = dates_construct['dates'].dt.year

        # This is established at the beginning of it, with the __init__ bool argument
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

        # Selection of the first five (5) numbers. First columns of the main DataFrame self.scrap
        self._col_df = range(1,6)

        # Selection of the two (2) stars. Last two (2) columns of the main DataFrame self.scrap
        self._col_stars_df = range(1,3)

        # Intersection of numbers population
        self.low_numbers = sorted(list(set(range(1,26)).intersection(self._numbers)))
        self.high_numbers = sorted(list(set(range(26,51)).intersection(self._numbers)))

    def __count(self, values: list, columns: Index) -> Series:
        count = self.df[columns].apply(lambda row: row.isin(values).sum(),axis=1)
        return count
    
    @Memoize
    def new_count_columns(self):
        # Columns to be considered from the main DataFrame
        selected_columns = self.df.columns[2:7]
        
        self.df['L'] = self.__count(self.low_numbers,selected_columns)
        self.df['H'] = self.__count(self.high_numbers,selected_columns)

    @property
    def db(self) -> DataFrame:
        return self.df
    
    @db.setter
    def db(self, modified_data: DataFrame, is_star: bool = False) -> DataFrame:

        assert modified_data is not None and isinstance(modified_data,DataFrame), "modified_data must be a DataFrame"

        # DataFrame with part of the information, in order to execute the loop to determine the % success of method
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

    def __control_condition(self, is_star: bool = False):
        # Function to improve readability of the code, by simplify the DataFrame and ranges in each particular case
        # Check __init__
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
    
    def __transformation_into_columns(self, row: pd.Index):
        for draw in range(1,6):
            if not np.isnan(self.year_history.loc[row.dates,row[f'nro{draw}']]):
                self.year_history.loc[row.dates,row[f'nro{draw}']] += 1

            else:
                self.year_history.loc[row.dates,row[f'nro{draw}']] = 1

    def __stars_transformation_into_columns(self, row: pd.Index):
        for draw in range(1,3):
            if not np.isnan(self.year_history.loc[row.dates,row[f'star_{draw}']]):
                self.year_history.loc[row.dates,row[f'star_{draw}']] += 1

            else:
                self.year_history.loc[row.dates,row[f'star_{draw}']] = 1
    
    @Memoize
    def apply_transformation(self, is_star: bool = False):
        # We determine the DataFrames and rages to be used
        self.__control_condition(is_star)

        # DataFrame to be populated
        self.year_history = pd.DataFrame(columns=self._col,
            index=np.arange(
                self.dates_construct['dates'].iloc[0],
                self.dates_construct['dates'].iloc[-1] + 1
            )
        )

        # We apply the corresponding transformation, in order to obtain the amount of times a number was a hit in every year
        if not is_star:
            self.dates_construct.apply(
                self.__transformation_into_columns,axis=1
            )

        else:
            self.dates_construct.apply(
                self.__stars_transformation_into_columns,axis=1
            )
        
        # Filling the NaN with 0
        self.year_history.fillna(0,inplace=True)

        self.hits = self.year_history.sum().to_frame().rename(columns={0: 'hits'}).T.astype('int32')
        self.mean = self.year_history.mean().to_frame().rename(columns={0: 'average'}).T.astype('float32')
        self.median = self.year_history.median().to_frame().rename(columns={0: 'median'}).T.astype('float32')

    def __total_average_hits(self, is_star: bool = False, aprox: bool = False) -> int:
        # We determine the DataFrames and rages to be used
        self.__control_condition(is_star)

        divide = 2 if is_star else 5
        self.average = self.hits.apply(lambda hits: hits / len(self._df) / divide).iloc[0]

        if aprox:
            return Decimal(self.average.sum()) / Decimal(len(self._col)) + Decimal(0.001)
        
        else:
            return Decimal(self.average.sum()) / Decimal(len(self._col))

    def __natural_rotation(self,is_star: bool = False,aprox: bool = False) -> DataFrame:
        # We determine the DataFrames and rages to be used
        self.__control_condition(is_star)

        self.__total_average_hits(is_star,aprox)

        rotation = pd.DataFrame(
            {'hits': self.hits.iloc[0],
            'average_of_numbers': self.average,
            'total_average': self.__total_average_hits(is_star,aprox),
            'minimal_hits_needed': self.get_min_hits(is_star,aprox)},
            index=self._col)
        
        rotation['difference'] = rotation['hits'] - rotation['minimal_hits_needed']
        return rotation

    def get_min_hits(self, is_star: bool = False, aprox: bool = False) -> int:
        min_hits = self.__total_average_hits(is_star,aprox)
        return min_hits * Decimal(int(self.hits.iloc[0,0])) / Decimal(float(self.average.iat[0]))
    
    def get_natural_rotations(self, is_star: bool = False) -> DataFrame:
        self.exact_rotation = self.__natural_rotation(is_star,aprox=False)
        self.aprox_rotation = self.__natural_rotation(is_star,aprox=True)

    def get_numbers_clasification(self, is_star: bool = False) -> np.ndarray | dict:
        # We determine the DataFrames and rages to be used
        self.__control_condition(is_star)

        self.best_numbers = self.aprox_rotation.loc[
            self.aprox_rotation['hits'] > self.aprox_rotation['minimal_hits_needed']].index.to_numpy()
        
        self.normal_numbers = self.exact_rotation.loc[
            (self.exact_rotation['hits'] > self.exact_rotation['minimal_hits_needed']) & ~(self.exact_rotation.index.isin(self.best_numbers))].index.to_numpy()
        
        self.category = {number: 0 for number in self._col}

        for number in self.best_numbers:
            self.category[number] = 2

        for number in self.normal_numbers:
            self.category[number] = 1
    
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
                
            self.booleans_df = self.booleans_df | (self._df[col_name].to_numpy()[:, None] == self._col)
        
        return self.booleans_df
    
    @Memoize
    def count_skips(self, is_star: bool = False) -> DataFrame:
        self.__numbers_boolean(is_star)

        # We create a mask in the positions where the value is 0 and 1: It will be True where the values are 0, and False where the values are equal to 1
        mask = self.booleans_df == 0

        # Another mask, where the value equals 1 is True (where the count restart itself) and False in the rest of positions
        reset_mask = self.booleans_df == 1

        # Mask variable contain booleans where True means skip and False means no skip
        cumulative_sum = np.cumsum(mask)

        # Puts a 0 in the True spots of reset_mask.
        cumulative_sum[reset_mask] = 0

        # This condition evaluates: if the elements in self.booleans_df are equal to 0, assigning them a (1) in the corresponding position. If is not 0, it assing the value of the cumulative sum
        result = np.where(self.booleans_df == 0, 1, cumulative_sum)

        result = pd.DataFrame(
            result,
            index=self._id,
            columns=self._col
        )

        # Creates a new DataFrame with booleans. The bool will be True where the values is not 0. The other positions will be False
        df_t = result != 0

        # This line calculates the cumsum of the array. With the method 'where', it puts a NaN where is a True, and then move the results to the next row vertically. Then, fills the NaN with 0 and converts all the frame to int
        self.counts = df_t.cumsum()-df_t.cumsum().where(~df_t).ffill().fillna(0).astype(int)
    
    @Memoize
    def groups_info(self) -> DataFrame:
        # Because of the amount of columns and data in the primary array, we just take this counter for the first five (5) numbers
        winning_numbers = self.df.drop(
            columns=['dates','star_1','star_2']
        )

        groups = [list(range(i,i+10)) for i in range(1,51,10)]
        group_names = [tuple(range(i,i+10)) for i in range(1,51,10)]

        # We iterate over the list and tuple to obtain the amount of numbers that won the last draw, in the same group
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

        # Set the five (5) arrays of numbers
        self.groups.columns = [f'{i}_to_{i+9}' for i in range(1,51,10)]

        # Actual situation of array population
        present_sg_10 = self.groups.iloc[-10:]

        # DataFrame for future reference, in order to exclude groups of numbers
        future_sg_10 = self.groups.iloc[-9:]

        self.present_groups = pd.DataFrame({
            '10_games': (present_sg_10 > 0).sum()}
        ).T

        self.future_groups = pd.DataFrame({
            '10_games': (future_sg_10 > 0).sum()}
        ).T

    @Memoize
    def draw_skips(self):
        # DataFrame to be populated
        self.skips_history = pd.DataFrame(columns=['nro1','nro2','nro3','nro4','nro5'])

        # Index to be updated. Row to be analize
        for index, row in self.counts.iterrows():
            # Row to be populated
            new_row = []

            for column, value in row.items():

                # We search the value in each column of euromillions.counts, within the row of reference
                if value == 0:
                    try:
                        aus = self.counts.loc[index - 1, column]
                    except (KeyError,IndexError):
                        aus = 0

                    if pd.isna(aus):
                        aus = 0
                    new_row.append(aus)
        
            while len(new_row) < 5:
                new_row.append(0)
        
            self.skips_history.loc[index] = new_row

    @Memoize 
    def skips_evaluation(self):
        self.evaluation = pd.DataFrame(columns=['0','5','7','10','13'])
        counts = pd.DataFrame(0, index=self.skips_history.index, columns=self.evaluation.columns)

        counts['0'] = self.skips_history.apply(lambda row: row.eq(0).sum(),axis=1)
        counts['5'] = self.skips_history.apply(lambda row: row.between(0,5).sum(),axis=1)
        counts['7'] = self.skips_history.apply(lambda row: row.between(0,7).sum(),axis=1)
        counts['10'] = self.skips_history.apply(lambda row: row.between(0,10).sum(),axis=1)
        counts['13'] = self.skips_history.apply(lambda row: row.between(0,13).sum(),axis=1)

        self.evaluation = pd.concat([self.evaluation,counts],ignore_index=True)
        self.evaluation = self.evaluation.set_index(pd.RangeIndex(1, len(self.evaluation) + 1))

    def skips_for_last_12_draws(self) -> DataFrame:
        # This prevents the NaN and the inexistance of a previous game from the first one. When it is equal to 1, the range begins at the second game
        if len(self.counts) - 11 == 1:
            last_12_draws = range(2,len(self.counts))

        else:
            last_12_draws = range(len(self.counts) - 11,len(self.counts) + 1)

        # [i-1] is becuase it evaluates the previous game, not the current one. For that, the condition above exist.
        aus_12 = [
            self.counts.loc[i-1,int(column)] 
            for i in last_12_draws[0:12]
            for column in self.counts if self.counts.loc[i,int(column)] == 0
        ]

        value = max(aus_12)
        if value < 13:
            self.skips = range(0,19)

        else:
            self.skips = range(0,value+1)
        
        # Counts the skips from the last seventh an twelveth games
        counter_l_7 = [Counter(aus_12[25:60]).get(i,0) for i in self.skips]
        counter_l_12 = [Counter(aus_12).get(i,0) for i in self.skips]

        self.skips_7_12 = pd.DataFrame(
                {
                '7': counter_l_7,
                '12': counter_l_12
            }
        )

        return self.skips_7_12

# Sub class for numbers analysis
class Criteria(Analysis):
    def __size_control(self, df_recommended: DataFrame, df_not_recommended: DataFrame) -> DataFrame:
        df = df_recommended.copy()
        surplus_numbers = df.drop(index=range(0,25)).reset_index(drop=True)
        self.not_recommended_numbers = pd.concat([df_not_recommended,surplus_numbers]).sort_values(by='criteria',ascending=False).reset_index(drop=True)
        self.recommended_numbers = df_recommended.head(25)

        return self.recommended_numbers, self.not_recommended_numbers
    
    def __skips_last_draws(self) -> DataFrame:
        last_draw = self.counts.iloc[-1].sort_values().to_frame()
        self.last_draw = last_draw.reset_index().rename(
            columns={'index':'number',
                    len(self.df):'skips'
                }
            )
        
        return self.last_draw

    def year_criterion(self) -> DataFrame:
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

    def rotation_criterion(self) -> DataFrame:
        current_hits_needed = super().get_min_hits(aprox=True)

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

    def position_criterion(self) -> DataFrame:
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

    def group_criterion(self) -> DataFrame:
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
    
    def numbers_of_tomorrow(self) -> DataFrame:
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

        self.recommended_numbers = tomorrow_numbers.loc[
            tomorrow_numbers['criteria'] >= criterion
        ].reset_index().rename(
            columns={
                'index':'numbers'
            }
        )
        
        self.not_recommended_numbers = tomorrow_numbers.loc[
            tomorrow_numbers['criteria'] < criterion
        ].reset_index().rename(
            columns={
                'index':'numbers'
            }
        )

        if len(self.recommended_numbers) > 25:
            self.__size_control(self.recommended_numbers,self.not_recommended_numbers)