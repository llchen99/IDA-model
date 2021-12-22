import pandas as pd
import numpy as np
array1 = np.array([1,1,3,1,0,6,7,1,9]).reshape(3,3)
array2 = np.array([1,1,3,1,1,6,7,1,9]).reshape(3,3)
df_0 = pd.DataFrame(array1, columns=list('ABC')) # no ID
df_1 = pd.DataFrame(array2, columns=list('ABC')) # ID
print(df_0)
print(df_1)

control_list = []
df_2 = df_0
print(f'Shape of df_0_all_baseline_2 {df_2.shape}')
for i in df_1.index:
    print(f'i {i}')
    age_1 = df_1['A'][i]
    # print(f'age {age_1}')
    gen_1 = int(df_1['B'][i])
    age_0_list = df_2.index[df_2['A']==age_1].tolist() #indexes of all controls with the same age
    # print(f'age_0_list {age_0_list}')
    gen_0_list = []

    for ind in age_0_list: 
        pt_no_0 = df_0['C'][ind] #vindt het patiëntnummer bij de index bij de originele dataframe
        # print(f'pt no {pt_no_0}')
        pt_row_0 = df_2.index[df_2['C']==pt_no_0][0]   # vindt de index bij het patiëntnummer, zodat het consistent blijft als de indices veranderen
        # als hij hier geen index kan vinden, betekent het dat die patiënt al geselecteerd is en hij door moet gaan
        # print(f'pt_row_0 {pt_row_0}')
        gen_0_list.append(int(df_2['B'][pt_row_0])) # genders of all the controls with the same age
        yo = int(df_2['B'][pt_row_0]) # hier geeft ie dus zowel de index als de waarde [0 2] zo. als je int gebruikt niet meer
        # wat we hier kunnen doen is testen of het geslacht hetzelfde is
        # zo ja, dan break
        if int(df_2['B'][pt_row_0]) == gen_1:
            # append bij de control_list
            control_list.append(pt_no_0)
            # verwijder uit df_2
            df_2 = df_2.drop(pt_row_0)
            break
        # zo nee, ga dan door met de loop
        else:  # dus basically als het geslacht niet overeenkomt
            # als het niet het laatste element is uit age_0_list, ga dan door 
            if ind+1 in age_0_list:
                pass
            # als het wel het laatste element is uit age_0_list, append dan gewoon
            else:
                control_list.append(pt_no_0)
                df_2 = df_2.drop(pt_row_0)
    print(df_2)
    print('ha')

print(f'control group {control_list}')
# volgens mij doet hij het!
# nu overzetten naar v1.3