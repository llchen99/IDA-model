import matplotlib.pyplot as plt


def syndromen(df_syndromen, df_0, df_1, df_syndroomcodes):
    ''' With this fuction, the syndromes of the subjects are analyzed.
    Inputs:
        df_syndromen: dataframe containing the patient IDs and syndrome codes.
        df_0: patient numbers of patients without ID.
        df_1: patient numbers of patients with ID.
        df_syndroomcodes: dataframe containing the numerical code for each syndrome and the syndrome name.
    Returns:
        Histograms containing the number of subjects with a certain syndrome.
        Pie charts containing the distribution of syndromes within the two groups.
    '''
    df_syndromen = df_syndromen.fillna(0)

    syndroom_control = []
    for pt in df_0['Pt_no']:
        slice = df_syndromen.loc[df_syndromen['Pt_no'] == pt]    # Fill dataframe with p-values
        syndroom = slice['Syndroom'].item()
        syndroom_control.append(syndroom)

    synnaam_control = []
    for syndroom in syndroom_control:
        slice2 = df_syndroomcodes.loc[df_syndroomcodes['number'] == syndroom]
        synnaam = slice2['syndrome'].item()
        synnaam_control.append(synnaam)

    plt.hist(synnaam_control, bins=70)
    plt.suptitle('Distribution of syndromes in control group')
    plt.xticks(rotation=90)
    plt.show()

    # initialize a dictionary to store the frequency of each element
    control_count = {}
    for element in synnaam_control:
        if element in control_count:
            control_count[element] += 1
        else:
            control_count[element] = 1
    # Huisstijl kleurtjes KT
    colors = ['#456D75', '#FFFFFF', '#5EB9ED', '#FFFFFF', '#5C6599', '#FFFFFF']

    plt.pie(control_count.values(), labels=control_count.keys(), colors=colors, wedgeprops={'edgecolor': 'black'})
    plt.suptitle('Pie chart of syndromes in control group')
    plt.show()

    print("Syndrome distribution in control group:")

    for key, value in control_count.items():
        print(f"{key}:{value}")
    # VB groep
    syndroom_ID = []
    for pt in df_1['Pt_no']:
        slice = df_syndromen.loc[df_syndromen['Pt_no'] == pt]
        syndroom = slice['Syndroom'].item()
        syndroom_ID.append(syndroom)

    synnaam_ID = []
    for syndroom in syndroom_ID:
        slice2 = df_syndroomcodes.loc[df_syndroomcodes['number'] == syndroom]
        synnaam = slice2['syndrome'].item()
        synnaam_ID.append(synnaam)

    plt.hist(synnaam_ID, bins=70)
    plt.suptitle('Distribution of syndromes in ID group')
    plt.xticks(rotation=90)
    plt.show()

    # initialize a dictionary to store the frequency of each element
    ID_count = {}
    for element in synnaam_ID:
        if element in ID_count:
            ID_count[element] += 1
        else:
            ID_count[element] = 1

    print("Syndrome distribution in ID group:")

    for key, value in ID_count.items():
        print(f"{key}:{value}")

    plt.pie(ID_count.values(), labels=ID_count.keys(), colors=colors, wedgeprops={'edgecolor': 'black'})
    plt.suptitle('Pie chart of syndromes in ID group')
    plt.show()
    return
