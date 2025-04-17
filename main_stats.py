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

input_data = "source/output/data_sorted.xlsx"
data = pd.read_excel(input_data)

data_front = data[data["location"] == "front"]
data_back = data[data["location"] == "back"]

data_front_self = data_front[data_front["type"] == "self"]

f, axes = plt.subplots(1, 2)
sns.boxplot(data_front, x="gender", y="points", hue="type", fill=False, ax=axes[0])
sns.boxplot(data_back, x="gender", y="points", hue="type", fill=False, ax=axes[1])

# df['group'] = df['gender'] + '_' + df['segment']
"""
formula = 'points ~ C(gender) + C(type) + C(gender):C(type)'
model = ols(formula, data_front).fit()
aov_table = anova_lm(model, typ=2)
print(aov_table)

print("-------------------------------------------------------------------------------")
"""
aov = pg.anova(data_front, dv="points", between=["gender", "type"], detailed=True)
print("--------------------------------- ANOVA --------------------------------------")
print(aov)

# Mixed-model ANOVA with two within-subjects factors:
# anova = pg.rm_anova(dv='Preference',
#                     within=['Condition', 'BodyPart'],
#                     subject='Subject',
#                     data=df,
#                     detailed=True)
#
# print(anova)

mixed_anova = pg.mixed_anova(dv='points',
                             within='type',
                             between='gender',
                             subject='id',
                             data=data_front)
print("--------------------------- MIXEL-MODEL ANOVA --------------------------------")
print(mixed_anova)

df = data_front

df['group'] = df['segment'] + '+' + df['type']
tukey_test = pg.pairwise_tukey(dv="points", between="group", data=df)

tukey_pd = pd.DataFrame(tukey_test)

# segments_names = ["hair", "mouth", "neck", "chest", "r_armpit", "r_hand", "l_armpit",
#                   "l_hand", "pelvis", "r_knee", "r_foot", "l_knee", "l_foot"]

for index, row in tukey_pd.iterrows():
    if row['A'].split('+')[0] == row['B'].split('+')[0]:
        print(row['A'].split('+'), row['B'].split('+'), row['p-tukey'])

# First, compute the mean and confidence intervals for plotting
summary = df.groupby(['gender', 'segment'])['points'].agg(['mean', 'std']).reset_index()

# plt.figure(figsize=(12, 6))
# sns.pointplot(data=df,
#               x='segment',
#               y='points',
#               hue='type',
#               errorbar='sd',
#               markers=['o', '*'],
#               linewidth=2,
#               )
# plt.grid(True)
#
# plt.figure(figsize=(12, 6))
# sns.pointplot(data=df,
#               x='segment',
#               y='points',
#               hue='gender',
#               errorbar='sd',
#               markers=['o', '*'],
#               linewidth=2,
#               )
# plt.grid(True)

plt.show()
