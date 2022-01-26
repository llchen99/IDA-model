import matplotlib.pyplot as plt
 
def syndromen(df_syndromen, df_control, df_vb):

    # Syndromen bekijken
    # nu nog beter plotten en in een functie gooien
    # Controle groep
    # eerst even alle missings veranderen naar 0
    df_syndromen = df_syndromen.fillna(0)

    syndroom_control = []
    for pt in df_control['Pt_no']:
        slice = df_syndromen.loc[df_syndromen['Pt_no'] == pt]    # Fill dataframe with p-values
        syndroom = slice['Syndroom'].item()
        syndroom_control.append(syndroom)
    plt.hist(syndroom_control, bins = 70)
    hist.suptitle('Distribution of syndromes in control group')

    plt.show()

    # initialize a dictionary to store the frequency of each element
    control_count = {}
    for element in syndroom_control:
        if element in control_count:
            control_count[element] += 1
        else:
            control_count[element] = 1

    print("Syndrome distribution in control group:")

    for key, value in control_count.items():
        print(f"{key}:{value}")

    # VB groep
    syndroom_ID = []
    for pt in df_vb['Pt_no']:
        slice = df_syndromen.loc[df_syndromen['Pt_no'] == pt]    # Fill dataframe with p-values
        syndroom = slice['Syndroom'].item()
        syndroom_ID.append(syndroom)
    #print(syndroom_ID)
    # Verwijder de waarde 999 want daar hebben we toch niets aan
    syndroom_ID.remove(999.0)
    plt.hist(syndroom_ID, bins = 70)
    hist.suptitle('Distribution of syndromes in ID group')

    plt.show()

    # initialize a dictionary to store the frequency of each element
    ID_count = {}
    for element in syndroom_ID:
        if element in ID_count:
            ID_count[element] += 1
        else:
            ID_count[element] = 1

    print("Syndrome distribution in ID group:")

    for key, value in ID_count.items():
        print(f"{key}:{value}")
    
    return control_count, ID_count, syndroom_control, syndroom_ID