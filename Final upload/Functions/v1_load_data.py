import pandas as pd


def load_data(path, columns):
    '''With this function, Excel files can be loaded into Python. 
    Inputs:
        path: path leading to the Excel file.
        columns: the desired columns from the Excel file.
    Returns: 
        df: dataframe containing the loaded data.'''
    df = pd.read_excel(path, usecols=columns)
    return df
