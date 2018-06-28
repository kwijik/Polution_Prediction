import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



def display_play(df_temp):
    y = df_temp['PM10']


    cols = df_temp.columns.tolist()
    cols.remove('PM10')

    for col in cols:
        x = df_temp[col]
        fig, ax = plt.subplots()
        ax.plot(x, y, 'o')
        ax.set_title('plot of ' + col)
        plt.show()

    h_df = df_temp[df_temp['Season'] == 'H']
    p_df = df_temp[df_temp['Season'] == 'P']
    e_df = df_temp[df_temp['Season'] == 'E']
    a_df = df_temp[df_temp['Season'] == 'A']

    winter = h_df['PM10'].mean()
    spring = p_df['PM10'].mean()
    summer = e_df['PM10'].mean()
    autaumn = a_df['PM10'].mean()

    labels = 'winter', 'spring', 'summer', 'automn'
    sizes = [winter, spring, summer, autaumn]
    explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.show()