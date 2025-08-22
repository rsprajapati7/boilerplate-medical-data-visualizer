import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv')

# 2
BMI = df['weight']/ ((df['height']/100)**2)
df['overweight'] = (BMI>25).astype(int) # using astype(int) will strictly put the value either 0 or 1
# 3
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)
# 4
def draw_cat_plot():
    # 5
    df_cat = pd.melt(
        df,
        id_vars = ['cardio'],
        value_vars = ['cholesterol', 'gluc', 'smoke', 'alco', 'active','overweight']
    )

    # 6
    df_cat = df_cat.groupby(['cardio', 'variable', 'value'])['value'].size().reset_index(name='total') # variable is for the feature we are looking for and the value could be either 0 or 1

    # 7

    plot = sns.catplot(
        x='variable',
        y = 'total',
        hue='value',
        col='cardio',
        data=df_cat,
        kind='bar'
    )


    # 8
    fig = plot.fig


    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    df_heat = df.loc[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ] # Cleaning the data according to the given requirements

    # 12
    corr = df_heat.corr().round(1)

    # 13
    mask = np.triu(np.ones_like(corr,dtype=bool)) # creates a matrix with True values and will only keep the upper triangle



    # 14
    fig, ax = plt.subplots(figsize=(12,8)) # using matplot 

    # 15

    sns.heatmap(corr, mask=mask, annot=True, cmap="coolwarm", ax=ax,fmt='.1f',square=True,cbar_kws={"shrink": 0.5}) # mask will hide the upper triangle and annot = True will show the values

    # 16
    fig.savefig('heatmap.png')
    return fig
