"""
author:
editor: EdgardoCS @FSU Jena
date: 24/03/2025
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import seaborn as sns
import numpy as np


def countInside(area, areaRange):
    """

    :param area: surface area to extend the search
    :param areaRange: set of points of smell map
    :return: points (in number) and image to be saved
    """
    match = []
    targetData = area
    x = areaRange[0]
    y = areaRange[1]
    w = areaRange[2]
    h = areaRange[3]

    for i in range(0, len(targetData)):
        if x <= targetData.iloc[i, 3] < x + w and y <= targetData.iloc[i, 2] < y + h:
            match.append([int(targetData.iloc[i, 3]), int(targetData.iloc[i, 2])])
    match = np.array(match)

    count = len(match)
    for point in match:
        plt.scatter(point[0], point[1], c="red")
    plt.show()
    return count


questionnaire = pd.read_excel("data/smell_behavior_sociodemographics.xlsx")

# Extract valid entries (id) from cleaned questionnaire dataframe
included = questionnaire["id"].unique()

# Load self and other body odor dataframes
self = pd.read_excel("data/body_silhouettes_self.xlsx")
other = pd.read_excel("data/body_silhouettes_other.xlsx")

# Filter for valid entries
self = self[self['id'].isin(included)]
other = other[other['id'].isin(included)]

# Group by id and count the number of rows for each id
self_count = self.groupby("id").size().reset_index(name='count')
self_agg = pd.merge(self[["country", "id"]], self_count, on="id").drop_duplicates()
# self_agg

# Group by id and count the number of rows for each id
other_count = other.groupby("id").size().reset_index(name='count')
other_agg = pd.merge(other[["country", "id"]], other_count, on="id").drop_duplicates()
# other_agg

# df = other
df = other.loc[other["country"] == "HKG"]  # to choose a particular country

# Load image
map_img = mpimg.imread('data/humanbody_clear.png')  # change path to image path

# Plot image
hmax = sns.scatterplot(x="other_x", y="other_y", data=df, alpha=0.08, zorder=2)

hmax.imshow(map_img)
hmax.invert_yaxis()
hmax.invert_xaxis()

hmax.set_xlim(hmax.get_xlim()[::-1])
hmax.set_ylim(hmax.get_ylim()[::-1])

# Find segments
# Load body segments
segments = pd.read_excel("data/body_segments.xlsx")
for i in range(0, len(segments)):
    included = [1,2]
    if segments.iloc[i]["id_segment"] in included:
        rectangleRange = [segments.iloc[i]['x'], segments.iloc[i]['y'], segments.iloc[i]['w'], segments.iloc[i]['h']]
        matchedPoints = countInside(df, rectangleRange)
        print("segment has", matchedPoints, "points")

plt.axis("on")
# plt.savefig("bodysilhouette_other.png", dpi=300, format="png")
print("Body odor perceived from body of others")
plt.show()
# print("           front ----------------------------- back")
