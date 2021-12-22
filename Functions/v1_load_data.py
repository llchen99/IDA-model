import pandas as pd

def load_data(path, columns):
    '''With this function, Excel files can be loaded into Python. The path to the file and the desired columns from the
    Excel file must be specified. A dataframe is returned.'''
    df = pd.read_excel(path, usecols=columns)
    return df