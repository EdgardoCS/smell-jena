"""
author: Antonie Bierling @FSU Jena
editor: EdgardoCS @FSU Jena
date: 24/03/2025
"""

import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def truncate(number, digits) -> float:
    # Improve accuracy with floating point operations, to avoid truncate(16.4, 2) = 16.39 or truncate(-1.13, 2) = -1.12
    nbDecimals = len(str(number).split('.')[1])
    if nbDecimals <= digits:
        return number
    stepper = 10.0 ** digits
    return math.trunc(stepper * number) / stepper


def count_whole(data, rectangle_range):
    """

    :param data:
    :param rectangle_range:
    :return:
    """
    f = []
    b = []

    target_data = data

    x = rectangle_range['x']
    y = rectangle_range['y']
    w = rectangle_range['w']
    h = rectangle_range['h']

    for i in range(0, len(target_data)):
        if x <= target_data.iloc[i, 3] < x + w and y <= target_data.iloc[i, 2] < y + h:
            f.append([int(target_data.iloc[i, 3]), int(target_data.iloc[i, 2])])

    for i in range(0, len(target_data)):
        b.append([int(target_data.iloc[i, 3]), int(target_data.iloc[i, 2])])

    f = len(f)
    b = len(b)

    b = abs(f - b)
    return f, b


def count_inside(data, rectangle_range):
    """

    :param data:
    :param rectangle_range:
    :return:
    """
    f = []
    b = []

    target_data = data

    x = rectangle_range.loc[rectangle_range['location'] == "front", 'x'].to_numpy()
    y = rectangle_range.loc[rectangle_range['location'] == "front", 'y'].to_numpy()
    w = rectangle_range.loc[rectangle_range['location'] == "front", 'w'].to_numpy()
    h = rectangle_range.loc[rectangle_range['location'] == "front", 'h'].to_numpy()

    for i in range(0, len(target_data)):
        if x <= target_data.iloc[i, 3] < x + w and y <= target_data.iloc[i, 2] < y + h:
            f.append([int(target_data.iloc[i, 3]), int(target_data.iloc[i, 2])])

    x = rectangle_range.loc[rectangle_range['location'] == "back", 'x'].to_numpy()
    y = rectangle_range.loc[rectangle_range['location'] == "back", 'y'].to_numpy()
    w = rectangle_range.loc[rectangle_range['location'] == "back", 'w'].to_numpy()
    h = rectangle_range.loc[rectangle_range['location'] == "back", 'h'].to_numpy()

    for i in range(0, len(target_data)):
        if x <= target_data.iloc[i, 3] < x + w and y <= target_data.iloc[i, 2] < y + h:
            b.append([int(target_data.iloc[i, 3]), int(target_data.iloc[i, 2])])

    return f, b


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

    segments_names = ["hair", "mouth", "neck", "chest", "r_armpit", "r_hand", "l_armpit",
                      "l_hand", "pelvis", "r_knee", "r_foot", "l_knee", "l_foot"]
    whole_body = ["front_side", "back_side"]
    segments_location = ["front", "back"]

    fig, axes = plt.subplots(nrows=2, ncols=9)
    # fig, axes = plt.subplots(1, 3)
    axes = axes.flatten()

    for i, country in enumerate(countries):
        df = other.loc[other["country"] == country]

        whole_front = 0
        whole_back = 0
        front_points = 0
        back_points = 0

        print("Sorting data...", country)
        rectangle = segments.set_index('segment').loc["front_side", ['location', 'x', 'y', 'w', 'h']]
        whole_front, whole_back = count_whole(df, rectangle)

        for segment in segments_names:
            rectangle = segments.set_index('segment').loc[segment, ['location', 'x', 'y', 'w', 'h']]
            front_points, back_points = count_inside(df, rectangle)

            front_points = len(front_points)
            back_points = len(back_points)

            f_percentage = truncate(front_points * 100 / whole_front, 1)
            b_percentage = truncate(back_points * 100 / whole_back, 1)

            new_data = {
                "frontal_percentage": [f_percentage],
                "back_percentage": [b_percentage]
            }
            new_df = pd.DataFrame(new_data)

            # segments = segments.reset_index()

            segments.loc[
                (segments['segment'] == segment) & (segments['location'] == "front"),
                ['percentage']] = [f_percentage]

            segments.loc[
                (segments['segment'] == segment) & (segments['location'] == "back"),
                ['percentage']] = [b_percentage]
        new_segments = segments[segments["percentage"].notnull()]

        hmax = sns.scatterplot(data=new_segments, x="xc", y="yc", size="percentage", sizes=(250, 1000), legend=False,
                               ax=axes[i], alpha=0.5)
        for line in range(0, 38):
            try:
                hmax.text(
                    new_segments["xc"][line] + 0.1,
                    new_segments["yc"][line] + 0.2,
                    new_segments["percentage"][line],
                    ha="left",
                    weight="bold"
                )
            except KeyError:
                pass
        hmax.imshow(map_img)
        hmax.invert_yaxis()
        hmax.invert_xaxis()

        hmax.set_xlim(hmax.get_xlim()[::-1])
        hmax.set_ylim(hmax.get_ylim()[::-1])
        axes[i].axis("off")
        axes[i].set_title(country)

    fig.delaxes(axes[17])
    plt.show()
