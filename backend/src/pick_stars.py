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
	
	def group_skip(self):
		self.euromillions_df_stars_skips = self.euromillions.df_stars.iloc[:,[1,2,3,4,5,6]]
		# Definir las combinaciones posibles
		combinations = [('B','D'), ('B','C'), ('A','C'), ('A','D'), ('A','B'), ('C','D'), ('B','A'), ('A','A'), ('B','B'), ('D','C'), ('C','C'), ('D','D')]

		# Inicializamos las nuevas columnas para almacenar los días desde la última aparición
		for comb in combinations:
			col_name = f'days_since_{comb[0]}_{comb[1]}'
			self.euromillions_df_stars_skips[col_name] = None

		# Inicializamos el contador de días
		days_since = {comb: 0 for comb in combinations}

		# Iteramos sobre las filas para calcular los días desde la última aparición de cada combinación
		for i in range(863,len(self.euromillions_df_stars_skips)):
			for comb in combinations:
				col_name = f'days_since_{comb[0]}_{comb[1]}'
				
				if self.euromillions_df_stars_skips.loc[i, 'combined_groups'] == list(comb):
					# Si la combinación actual coincide, reiniciamos el contador a 0
					days_since[comb] = 0
				else:
					# Si no coincide, incrementamos el contador de días
					days_since[comb] += 1
				
				# Asignamos el valor del contador de días en la nueva columna
				self.euromillions_df_stars_skips.loc[i, col_name] = days_since[comb]

	def suggested_stars(self):
		mean = self.euromillions_df_stars_skips.iloc[:,6:].median().to_frame().T
		
		pass