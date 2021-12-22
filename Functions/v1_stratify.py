import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

def stratify(df_no_ID, df_ID):
    ''' With this fuction, the control data is stratified so the control group is the same size as the ID group. 
    Each patient in the ID group is analyzed on gender and age. First, a list of all patients with the same gender is found. 
    Then, the age for each of these patients is taken to compare with the age of the ID patient. Patients are selected if the age difference is <3 years. 
    A dataframe with the stratified patient selection is returned.'''
    control_list = []
    df_no_ID2 = df_no_ID
    for i in df_ID.index:
        age_1 = df_ID['Leeftijd'][i]
        gen_1 = int(df_ID['Geslacht'][i])
        age_0_list = [] 
        gen_0_list = df_no_ID2.index[df_no_ID2['Geslacht']==gen_1].tolist() #indexes of all controls with the same gender
        for ind in gen_0_list:
            pt_no_0 = df_no_ID['Pt_no'][ind] #vindt het patiëntnummer bij de index bij de originele dataframe
            pt_row_0 = df_no_ID2.index[df_no_ID2['Pt_no']==pt_no_0][0]   # vindt de index bij het patiëntnummer, zodat het consistent blijft als de indices veranderen
            age_0_list.append(int(df_no_ID2['Leeftijd'][pt_row_0])) # genders of all the controls with the same age

            # wat we hier kunnen doen is testen of de leeftijd hetzelfde is
            # zo ja, dan break
            if 0 < int(df_no_ID2['Leeftijd'][pt_row_0]) - age_1 < 3:
                # append bij de control_list
                control_list.append(pt_no_0)
                # verwijder uit df_2
                df_no_ID2 = df_no_ID2.drop(pt_row_0)
                break
            # zo nee, ga dan door met de loop
            elif 0 > int(df_no_ID2['Leeftijd'][pt_row_0]) - age_1 > -3 :
                # append bij de control_list
                control_list.append(pt_no_0)
                df_no_ID2 = df_no_ID2.drop(pt_row_0)
                break
            else:
                pass  
    # En dan moeten we nu nog al die patiënt nummers weer in een nieuwe dataframe gooien 
    pt_ind = []
    for control in control_list:
        pt_ind.append(int(df_no_ID.index[df_no_ID['Pt_no']==control][0]))
    df_control = df_no_ID.iloc[pt_ind,:]

    return df_control