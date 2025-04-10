"""
author: Antonie Bierling @FSU Jena
editor: EdgardoCS @FSU Jena
date: 24/03/2025
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def count_inside(area, area_range, body_segment):
    """
    Takes a rectangle from area_range and count how many points are inside that surface,
    points are taken from area, which has the data
    :param area: surface area to extend the search
    :param area_range: set of points of smell map
    :param body_segment: name of each body segment
    :return: points to be saved
    """
    match = []
    unmatch = []

    target_data = area
    x = area_range[0]
    y = area_range[1]
    w = area_range[2]
    h = area_range[3]

    for i in range(0, len(target_data)):
        if x <= target_data.iloc[i, 3] < x + w and y <= target_data.iloc[i, 2] < y + h:
            match.append([int(target_data.iloc[i, 3]), int(target_data.iloc[i, 2])])
        else:
            unmatch.append([int(target_data.iloc[i, 3]), int(target_data.iloc[i, 2])])

    match = np.array(match)
    unmatch = np.array(unmatch)

    # count_match = len(match)
    # count_unmatch = len(unmatch)
    # print("body segment", body_segment, "has", count_match, "points from a total of", count_unmatch + count_match,
    #       "points")

    return match


if __name__ == "__main__":

    # TODO: Normalize for each country
    # TODO: count how many points are in total to calculate percentages
    # TODO: use bubbles or percentage in image to plot

    # Load data from questionnaire
    questionnaire = pd.read_excel("source/data/smell_behavior_sociodemographics.xlsx")
    # Load self and other body odor dataframes
    self = pd.read_excel("source/data/body_silhouettes_self.xlsx")
    other = pd.read_excel("source/data/body_silhouettes_other.xlsx")

    # Extract valid entries (id) from cleaned questionnaire dataframe and filter for valid entries
    included = questionnaire["id"].unique()
    self = self[self['id'].isin(included)]
    other = other[other['id'].isin(included)]

    # Group by id and count the number of rows for each id.
    # This will count how many points each subject marked, and store that value on self-agg
    # The same for other_agg
    self_count = self.groupby("id").size().reset_index(name='count')
    self_agg = pd.merge(self[["country", "id"]], self_count, on="id").drop_duplicates()

    other_count = other.groupby("id").size().reset_index(name='count')
    other_agg = pd.merge(other[["country", "id"]], other_count, on="id").drop_duplicates()

    countries = self_agg["country"].unique()

    # Load body segments and image
    segments = pd.read_excel("source/data/body_segments.xlsx")
    map_img = mpimg.imread('source/img/humanbody_clear.png')  # change path to image path

    df = other

    # plt.savefig("source/output/" + "body_others.png", dpi=300, format="png")

    colors = ["#FF6B6B", "#FFA94D", "#FFD43B", "#69DB7C", "#38D9A9", "#4DABF7", "#5C7CFA",
              "#9775FA", "#DA77F2", "#F783AC", "#ADB5BD", "#343A40", "#FFC9E3", "#B2F2BB"]

    data_x1, data_x2 = [], []
    data_y1, data_y2 = [], []
    data_diameter1, data_diameter2 = [], []
    data_countries = []

    for i, country in enumerate(countries):

        df = other.loc[other["country"] == country]

        scatterPoints_front = []
        scatterPoints_back = []
        wholePoints_front = []
        wholePoints_back = []

        print("Sorting data...", country)

        for j in range(0, len(segments)):
            included_front = [0, 1, 2, 3, 4, 6, 7, 9, 10, 12, 14, 16, 18]
            included_back = list(map(lambda j: j + 19, included_front))
            whole_front = [38]
            whole_back = [39]

            if segments.iloc[j]["id_segment"] in included_front:
                rectangleRange = [segments.iloc[j]['x'], segments.iloc[j]['y'],
                                  segments.iloc[j]['w'], segments.iloc[j]['h']]
                matchedPoints = count_inside(df, rectangleRange, segments.iloc[j]["segment"])
                scatterPoints_front.append(matchedPoints)

            elif segments.iloc[j]["id_segment"] in included_back:
                rectangleRange = [segments.iloc[j]['x'], segments.iloc[j]['y'],
                                  segments.iloc[j]['w'], segments.iloc[j]['h']]
                matchedPoints = count_inside(df, rectangleRange, segments.iloc[j]["segment"])
                scatterPoints_back.append(matchedPoints)

            elif segments.iloc[j]["id_segment"] in whole_front:
                rectangleRange = [segments.iloc[j]['x'], segments.iloc[j]['y'],
                                  segments.iloc[j]['w'], segments.iloc[j]['h']]
                matchedPoints = count_inside(df, rectangleRange, segments.iloc[j]["segment"])
                wholePoints_front.append(matchedPoints)

            elif segments.iloc[j]["id_segment"] in whole_back:
                rectangleRange = [segments.iloc[j]['x'], segments.iloc[j]['y'],
                                  segments.iloc[j]['w'], segments.iloc[j]['h']]
                matchedPoints = count_inside(df, rectangleRange, segments.iloc[j]["segment"])
                wholePoints_back.append(matchedPoints)

        x1, x2 = [], []
        y1, y2 = [], []
        diameter1, diameter2 = [], []

        for k in range(0, len(scatterPoints_front)):
            included_front = [0, 1, 2, 3, 4, 6, 7, 9, 10, 12, 14, 16, 18]
            included_back = [19, 20, 21, 22, 23, 25, 26, 28, 29, 31, 33, 35, 37]

            x1.append(float(segments.iloc[included_back[k]]['xc']))
            y1.append(float(segments.iloc[included_back[k]]['yc']))
            diameter1.append((len(scatterPoints_front[k]) * 100) / len(wholePoints_front[0]))

            x2.append(float(segments.iloc[included_back[k]]['xc']))
            y2.append(float(segments.iloc[included_back[k]]['yc']))
            diameter2.append((len(scatterPoints_back[k]) * 100) / len(wholePoints_back[0]))

        data_x1.append(x1)
        data_y1.append(y1)
        data_diameter1.append(diameter1)

        data_x2.append(x2)
        data_y2.append(y2)
        data_diameter2.append(diameter2)

        data_countries.append(country)

    print(data_x1)
    print(type(data_x1))
    print(data_x1[0])
    print(type(data_x1[0]))
    print(data_x1[0][0])
    print(type(data_x1[0][0]))


"""
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    for a in range(1, 17):
        plt.subplot(2, 8, a)
        hmax = ax1.sns.scatterplot(x=data_x1[i], y=data_y1[i], size=data_diameter1[i], sizes=(100, 400), legend=False,
                               alpha=0.5, color = "blue")
        hmax = ax1.sns.scatterplot(x=data_x2[i], y=data_y2[i], size=data_diameter2[i], sizes=(100, 400), legend=False,
                               alpha=0.5, color = "orange")
        hmax.imshow(map_img)
        hmax.invert_yaxis()
        hmax.invert_xaxis()

        hmax.set_xlim(hmax.get_xlim()[::-1])
        hmax.set_ylim(hmax.get_ylim()[::-1])

    plt.show()
"""

# for i in range(0, len(scatterPoints_front)):
#     x1 = scatterPoints_front[i][:, 0]
#     y1 = scatterPoints_front[i][:, 1]
#
#     x2 = scatterPoints_back[i][:, 0]
#     y2 = scatterPoints_back[i][:, 1]
#
#     sns.scatterplot(x=x1, y=y1, alpha=0.5, color=colors[i])
#     sns.scatterplot(x=x2, y=y2, alpha=0.5, color=colors[i])
#
# plt.show()
#
# plt.axis("on")
# plt.savefig("source/output/" + "body_others_painted.png", dpi=300, format="png")
# print("Body odor perceived from body of others")

"""
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
 
# Create dataframe
df = pd.DataFrame({
'x': [1, 1.5, 3, 4, 5],
'y': [5, 15, 5, 10, 2],
'group': ['A','other group','B','C','D']
})
 
sns.scatterplot(data=df, x="x", y="y", s=200)

# add annotations one by one with a loop
for line in range(0,df.shape[0]):
     plt.text(
          df["x"][line]+0.2,
          df["y"][line],
          df["group"][line],
          ha='left',
          weight='bold'
     )

plt.show()
"""
