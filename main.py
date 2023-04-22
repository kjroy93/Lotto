import pandas as pd

df = pd.DataFrame(columns=['name', 'age', 'year'],data=[['John', 23, 1995], ['Mary', 21, 1994], ['Peter', 25, 1996]])
print(df.year.iloc[-1])