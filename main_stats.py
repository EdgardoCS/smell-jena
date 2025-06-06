"""
author: EdgardoCS @FSU Jena
date: 16.04.2025
"""

import pandas as pd
import seaborn as sns
import pingouin as pg
import matplotlib.pyplot as plt

import scikit_posthocs as sp
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols, mixedlm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

pd.options.mode.chained_assignment = None

input_data = "source/output/data_sorted_all(1).xlsx"
data = pd.read_excel(input_data)

data_front = data[(data['location'] == 'front')]

target1 = data[
    (data['location'] == 'front') &
    (data['type'] == 'self') &
    (data['gender'].isin(['female', 'male']))
    ].copy()

target2 = data[
    (data['location'] == 'front') &
    (data['type'] == 'other') &
    (data['gender'].isin(['female', 'male']))
    ].copy()

target3 = data[
    (data['location'] == 'back') &
    (data['type'] == 'self') &
    (data['gender'].isin(['female', 'male']))
    ].copy()

target4 = data[
    (data['location'] == 'back') &
    (data['type'] == 'other') &
    (data['gender'].isin(['female', 'male']))
    ].copy()

df1 = [target1, target2]
df2 = [target3, target4]

for d in df1:
    model = mixedlm("points ~ segment * gender", d, groups=d["id"]).fit()
    print(model.summary())
    print("****************************************************************************")

for e in df2:
    model = mixedlm("points ~ segment * gender", e, groups=e["id"]).fit()
    print(model.summary())
    print("****************************************************************************")

three_model = ols("""points ~ C(segment) + C(gender) + C(type) +
               C(segment):C(gender) + C(segment):C(type) + C(gender):C(type) +
               C(segment):C(gender):C(type)""", data=data_front).fit()
anova_lm(three_model, typ=2)
print(three_model.summary())