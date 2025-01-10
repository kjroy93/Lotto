from .parse import Criteria

class Selection():
    def __init__(self):
        self.euromillions = Criteria(is_star=True)

    def __find_group(value: int) -> str | None:
        stars_group = {
        'A': [1,3,5],
        'B': [2,4,6],
        'C': [7,9,11],
        'D': [8,10,12]
    }
        
        for k, v in stars_group.items():
            if value in v:
                return k
        
        return None
    
    def define_groups(self):
        self.euromillions.df_stars['group_0'] = self.euromillions.df_stars['star_1'].apply(self.__find_group) 
        self.euromillions.df_stars['group_1'] = self.euromillions.df_stars['star_2'].apply(self.__find_group)

        self.euromillions.df_stars['combined_groups'] = self.euromillions.df_stars[['group_0','group_1']].apply(lambda row: [value for value in row if isinstance(value, str)], axis = 1)

    def suggested_stars(self):
        pass