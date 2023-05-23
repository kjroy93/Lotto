"""New functions with the refactorization and vectorization process, with aplication of OOP"""

# Standard Libraries of Python
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP, getcontext
getcontext().prec = 5

# Dependencies
import pandas as pd
import numpy as np
np.set_printoptions(precision=5)

# Main Object - Super Class
class Analysis:
    def __init__(self,is_star=False):
        self.df = pd.read_parquet('database/db.parquet')
        self.df_stars = self.df.drop(
            columns=['nro1','nro2','nro3','nro4','nro5'],
            index=range(0,863)
        )
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
    
    def groups_info(self):
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

    def __ranges(self):
        self._draws = range(1,len(self.df)+1)
        self._numbers = range(1,51)
        self._stars = range(1,13)
        self._draws_stars = range(864,len(self.df)+1)

    def __control_condition(self,is_star=False):
        self.__ranges()
        if not is_star:
            self._df = self.df
            self._col = self._numbers
            self._id = self._draws
            self._col_df = range(1,6)
        else:
            self._df = self.df_stars
            self._col = self._stars
            self._id = self._draws_stars
            self._col_df = range(1,3)
    
    def __transformation_into_columns(self,row):
        for draw in range(1,6):
            if not np.isnan(self.year_history.loc[row.dates,row[f'nro{draw}']]):
                self.year_history.loc[row.dates,row[f'nro{draw}']] += 1
            else:
                self.year_history.loc[row.dates,row[f'nro{draw}']] = 1

    def __stars_transformation_into_columns(self,row):
        for draw in range(1,3):
            if not np.isnan(self.year_history.loc[row.dates,row[f'star_{draw}']]):
                self.year_history.loc[row.dates,row[f'star_{draw}']] += 1
            else:
                self.year_history.loc[row.dates,row[f'star_{draw}']] = 1

    def apply_transformation(self,is_star=False):
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

    def __numbers_boolean(self,is_star=False):
        self.__control_condition(is_star)
        self.booleans_df = pd.DataFrame(False,columns=self._col,
            index=self._id)
        for e in self._col_df:
            if not is_star:
                col_name = f'nro{e}'
            else:
                col_name = f'star_{e}'
            self.booleans_df = self.booleans_df | (
                self._df[col_name].to_numpy()
                [:, None] == self._col
            )

    def count_skips(self,is_star=False):
        self.__numbers_boolean(is_star)
        mask = self.booleans_df == 0
        reset_mask = self.booleans_df == 1
        cumulative_sum = np.cumsum(mask)
        cumulative_sum[reset_mask] = 0
        result = np.where(self.booleans_df == 0, 1, cumulative_sum)
        result = pd.DataFrame(
            result,
            index=self._id,
            columns=self._col)
        df_t = result != 0
        self.counts = df_t.cumsum()-df_t.cumsum().where(~df_t).ffill().fillna(0).astype(int)

    def skips_for_last_12_draws(self):
        self.skips = range(0,19)
        if len(self.counts) - 12 == 0:
            last_12_draws = range(1,len(self.counts))
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
            {'7': counter_l_7,
            '12': counter_l_12
            }
        )

    def __total_average_hits(self,is_star=False,aprox=False):
        self.__control_condition(is_star)
        divide = 2 if is_star else 5
        self.average = self.hits.apply(lambda hits: hits / len(self._df) / divide).iloc[0]
        if aprox:
            return Decimal(self.average.sum()) / Decimal(len(self._numbers)) + Decimal(0.001)
        else:
            return Decimal(self.average.sum()) / Decimal(len(self._numbers))

    def m_hits(self,is_star=False,aprox=False):
        min_hits = self.__total_average_hits(is_star,aprox)
        return min_hits * Decimal(int(self.hits.iloc[0,0])) / Decimal(float(self.average.iat[0]))

    def __natural_rotation(self,is_star=False,aprox=False):
        self.__control_condition(is_star)
        self.__total_average_hits(is_star,aprox)
        rotation = pd.DataFrame(
            {'hits': self.hits.iloc[0],
            'average_of_numbers': self.average,
            'total_average': self.__total_average_hits(is_star,aprox),
            'minimal_hits_needed': self.m_hits(is_star,aprox)},
            index=self._numbers)
        
        rotation['difference'] = rotation['hits'] - rotation['minimal_hits_needed']
        return rotation

    def get_natural_rotations(self,is_star=False):
        self.exact_rotation = self.__natural_rotation(is_star,aprox=False)
        self.aprox_rotation = self.__natural_rotation(is_star,aprox=True)

    def numbers_clasification(self,is_star):
        self.__control_condition(is_star)
        self.best_numbers = self.aprox_rotation.loc[
            self.aprox_rotation['hits'] > self.aprox_rotation['minimal_hits_needed']
            ].index.to_numpy()
        self.normal_numbers = self.exact_rotation.loc[
            (self.exact_rotation['hits'] > self.exact_rotation['minimal_hits_needed'])
            & ~(self.exact_rotation.index.isin(self.best_numbers))
            ].index.to_numpy()
        self.category = {number: 0 for number in self._numbers}
        for number in self.best_numbers:
            self.category[number] = 2
        for number in self.normal_numbers:
            self.category[number] = 1

# Sub class
class Criteria(Analysis):
    def __init__(self):
        super().__init__()
    
    def __skips_last_draws(self):
        super().count_skips()
        last_draw = self.counts.iloc[-1].sort_values().to_frame()
        self.last_draw = last_draw.reset_index().rename(
            columns={'index':'number',
                    len(self.df):'skips'
                }
            )

    def year_criterion(self):
        super().apply_transformation()
        current_year = self.dates_construct['dates'].iloc[-1]
        year_criteria = {key: [] for key in self.numbers}

        for number in self.numbers:
            x = self.year_history.at[current_year,number]
            number_median = self.median.at['median',number]
            number_mean = self.mean.at['average',number]
            if number_mean != 0 and not np.isnan(number_mean):
                if x == 0 or x <= number_median / 2:
                    y = round(
                        (1 - (
                        ((number_median / 2) * 100) / 
                        number_mean
                    ) / 100),2
                )
                else:
                    x_percentage = round(
                        (x * 100 / number_mean) / 100,2
                    )
                    y = round(
                        (1 - x_percentage if x_percentage > 1 
                        else 1 - x_percentage),2
                    )
            else:
                y = 0.50

            year_criteria[number] = float(y)

        self.years_criteria = pd.DataFrame.from_dict(
            year_criteria,
            orient='index',
            columns=['year_criteria']
        )
    
    def rotation_criterion(self):
        super().get_natural_rotations()
        super().numbers_clasification()
        current_hits_needed = super().m_hits(aprox=True)

        self.rotation = next(
            (draw for draw in range(len(self.df['draw'])+1,len(self.df['draw'])+10) 
            if draw * current_hits_needed/len(self.df)>
            int(current_hits_needed)+1),None
        )
        rotation_criteria = {}
        for number in self.numbers:
            category = self.category[number]
            diff_exact = self.exact_rotation.at[number,'difference']
            diff_aprox = self.aprox_rotation.at[number,'difference']
            if category == 1 and diff_exact > 0:
                rotation_criteria[number] = 0.50
            elif category == 1 and -1 < diff_exact <= 0.60:
                cr = 0.50
                x = (
                    abs(diff_exact) + 
                    (Decimal(str(cr)) * Decimal('100')) / 
                    Decimal(str(cr))
                ) / Decimal('100') + Decimal(str(cr))
                rotation_criteria[number] = float(x.quantize(Decimal('0.01'),
                    rounding=ROUND_HALF_UP)
                )
            elif category == 2 and diff_aprox > 0:
                rotation_criteria[number] = 1
                cr = 1.00
            elif category == 2 and diff_aprox > -1 and diff_aprox < 1 and self.rotation is not None and len(self.best_numbers) < 17:
                x = (
                    abs(diff_exact) * Decimal('100')
                ) / Decimal(str(cr)) / Decimal('100') + Decimal(str(cr))
                rotation_criteria[number] = float(x.quantize(Decimal('0.01'),
                    rounding=ROUND_HALF_UP)
                )
            else:
                rotation_criteria[number] = 0.25
        
        self.rotation_criteria = pd.DataFrame.from_dict(
            rotation_criteria,
            orient='index',
            columns=['rotation_criteria']
        )

    def position_criterion(self):
        self.__skips_last_draws()
        super().skips_for_last_12_draws()
        self.skips_7_12[['values_7','values_12']] = self.skips_7_12[['7','12']].applymap(
            lambda x: games_7(x) if x <= 7 else
            games_12(x)
        )

        position_criteria = {number: [] for number in self.numbers}

        last_game = 0.5
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
    
    def group_criterion(self):
        super().groups_info()
        groups = [list(range(i,i+10)) for i in range(1,51,10)]
        self.future_groups.columns = [1,2,3,4,5]
        gp10 = {}
        gp_values = {
            0: 1,
            1: 1,
            2: 1,
            3: 1,
            4: 0.80,
            5: 0.75,
            6: 0.675,
            7: 0.650,
            8: 0.625,
            9: 0.6,
            10: 0.5
        }
        for column, gp in self.future_groups.loc['10_games'].items():
            gp_range = tuple(range(1+10*(column-1),11+10*(column-1)))
            gp10[gp_range] = gp_values.get(gp,0)
        
        gp10_df = pd.DataFrame.from_dict(
            gp10,
            orient='index',
        ).T
        
        gp10_df = gp10_df.rename(index={0: 'criteria'})
        gp10_df.columns = [1,2,3,4,5]
        group_criteria = {number:[] for number in self.numbers}

        for index, group in enumerate(groups):
            gn = group
            for number in self.numbers:
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

def clean_df(df,columns_id,name):
    df.columns = columns_id
    df.columns.name = name
    df.index.name = 'Draws'
    return df

def draw_generator(lenght):
    for draw in range(12,lenght):
        yield draw

def games_7(column):
    games_dict = {0: 1, 1: 1, 2: 0.75, 3: 0.65, 4: 0.55, 5: 0.45, 6: 0.35, 7: 0.25}
    return games_dict.get(column,0)

def games_12(column):
    games_dict = {8: 0.65, 9: 0.55, 10: 0.45, 11: 0.35, 12: 0.25}
    return games_dict.get(column,0)

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