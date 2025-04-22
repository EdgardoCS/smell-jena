"""
author: EdgardoCS @FSU Jena
date: 16.04.2025
"""

import pandas as pd
import seaborn as sns
import pingouin as pg
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

pd.options.mode.chained_assignment = None

input_data = "source/output/data_sorted_all.xlsx"
data = pd.read_excel(input_data)

data_front = data[data["location"] == "front"]
data_back = data[data["location"] == "back"]

df = data_front

# f, axes = plt.subplots(1, 2)
# sns.boxplot(data_front, x="gender", y="points", hue="type", fill=False, ax=axes[0])
# sns.boxplot(data_back, x="gender", y="points", hue="type", fill=False, ax=axes[1])

# df['group'] = df['gender'] + '_' + df['segment']

formula = 'points ~ C(gender) + C(type) + C(gender):C(type)'
model = ols(formula, data_front).fit()
aov_table = anova_lm(model, typ=2)
print(aov_table)

aov = pg.anova(df, dv="points", between=["gender", "type"], detailed=True)
print("--------------------------------- ANOVA --------------------------------------")
print(aov)

mixed_anova = pg.mixed_anova(dv='points',
                             within='type',
                             between='gender',
                             subject='id',
                             data=df)
print("--------------------------- MIXEL-MODEL ANOVA --------------------------------")
print(mixed_anova)

front_df = data[
    (data['location'] == 'front') &
    (data['gender'] == 'male') &
    (data['type'].isin(['self', 'other']))
    ].copy()

front_df['group'] = front_df['type'] + '+' + front_df['segment']

tukey_results = pg.pairwise_tukey(
    dv='points',
    between='group',
    data=front_df
)

for index, row in tukey_results.iterrows():
    if row['A'].split('+')[1] == row['B'].split('+')[1]:
        print(row)
        print(row['A'].split('+'), row['B'].split('+'), row['p-tukey'])
