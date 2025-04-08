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


    #TODO: count how many points are in total to calculate percentages
    #TODO: draw armpits on image and set segment
    #TODO: use bubbles or percentage in image to plot

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

    df = other
    # df = other.loc[other["country"] == "HKG"]  # to choose a particular country

    # Load image
    map_img = mpimg.imread('source/img/humanbody_clear.png')  # change path to image path

    # Plot image
    hmax = sns.scatterplot(x="other_x", y="other_y", data=df, alpha=0.00, zorder=2)
    hmax.imshow(map_img)
    hmax.invert_yaxis()
    hmax.invert_xaxis()

    hmax.set_xlim(hmax.get_xlim()[::-1])
    hmax.set_ylim(hmax.get_ylim()[::-1])
    # plt.savefig("source/output/" + "body_others.png", dpi=300, format="png")

    # Load body segments
    segments = pd.read_excel("source/data/body_segments.xlsx")

    scatterPoints_front = []
    scatterPoints_back = []

    front_points = 0
    back_points = 0

    # Z

    print("Sorting data...")
    for i in range(0, len(segments)):
        included = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                    14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
        if segments.iloc[i]["id_segment"] in included:
            rectangleRange = [segments.iloc[i]['x'], segments.iloc[i]['y'],
                              segments.iloc[i]['w'], segments.iloc[i]['h']]
            matchedPoints = count_inside(df, rectangleRange, segments.iloc[i]["segment"])
            if i < 13:
                scatterPoints_front.append(matchedPoints)
            elif i > 12:
                scatterPoints_back.append(matchedPoints)

    print("Ready to plot")

    colors = ["#FF6B6B", "#FFA94D", "#FFD43B", "#69DB7C", "#38D9A9", "#4DABF7", "#5C7CFA",
              "#9775FA", "#DA77F2", "#F783AC", "#ADB5BD", "#343A40", "#FFC9E3"]

    for i, segment in enumerate(scatterPoints_front):
        x1 = scatterPoints_front[i][:, 0]
        y1 = scatterPoints_front[i][:, 1]

        x2 = scatterPoints_back[i][:, 0]
        y2 = scatterPoints_back[i][:, 1]

        sns.scatterplot(x=x1, y=y1, alpha=0.5, color=colors[i])
        sns.scatterplot(x=x2, y=y2, alpha=0.5, color=colors[i])

    plt.show()
#
#   plt.axis("on")
#   plt.savefig("source/output/" + "body_others_painted.png", dpi=300, format="png")
#   print("Body odor perceived from body of others")
