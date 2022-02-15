import pandas as pd
import numpy as np
from scipy import stats
from statistics import mode
# array1 = np.array([1,1,3,1,0,6,7,1,9]).reshape(3,3)
# array2 = np.array([np.NaN,1,3,1,np.NaN,6,7,1,9]).reshape(3,3)
# df_0 = pd.DataFrame(array1, columns=list('ABC')) # no ID
# df_1 = pd.DataFrame(array2, columns=list('ABC')) # ID

# print(df_1['A'].dropna(axis=0))
# print(df_1.isna())

# list = [[1, 2], [3, 4], [5, 1]]
# print(list)
# #print(set(list))
# result = [a for list in list for a in list]
# print(result)
# print(set(result))
# fruit = {
#     "banana": 1.00,
#     "apple": 1.53,
#     "kiwi": 2.00,
#     "avocado": 3.23,
#     "mango": 2.33,
#     "pineapple": 1.44,
#     "strawberries": 1.95,
#     "melon": 2.34,
#     "grapes": 0.98
# }
# analysis = pd.DataFrame(columns = ('frequency', 'mean_ID', 'std_ID', 'mean_control', 'std_control'))
# analysis.index.name = 'Feature'
# for key in fruit:
#     print(key)
#     print('nieuw')
#     analysis.loc[key] = [fruit[key], np.mean(fruit[key]), 3, 2, 3]

# print(analysis)
# analysis.to_excel("test.xlsx")
# list = ['RF', 'SVM']
# print(list[0])

