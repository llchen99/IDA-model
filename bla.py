import pandas as pd
import numpy as np
array1 = np.array([1,1,3,1,0,6,7,1,9]).reshape(3,3)
array2 = np.array([1,1,3,1,1,6,7,1,9]).reshape(3,3)
df_0 = pd.DataFrame(array1, columns=list('ABC')) # no ID
df_1 = pd.DataFrame(array2, columns=list('ABC')) # ID
print(df_0.iloc[0])
print(df_1)