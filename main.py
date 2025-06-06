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
from matplotlib.pyplot import tight_layout


def sort_to_export(df1, df2, df3, names):
    """

    :param df1:
    :param df2:
    :param df3:
    :param names:
    :return:
    """
    df_self1 = df1[df1["gender"] != "male"]
    df_self2 = df1[df1["gender"] != "female"]
    df_other1 = df2[df2["gender"] != "male"]
    df_other2 = df2[df2["gender"] != "female"]

    dfs = [df_self1, df_self2, df_other1, df_other2]
    titles = ["Females (Self)", "Males (Self)", "Females (Others)", "Males (Others)"]

    data_sorted = pd.DataFrame()

    for i, sex in enumerate(dfs):

        print("Working on", titles[i])

        for segment in names:
            print("Working on", segment)
            rectangle = df3.set_index('segment').loc[segment, ['location', 'x', 'y', 'w', 'h']]

            x1 = rectangle.loc[rectangle['location'] == "front", 'x'].to_numpy()
            y1 = rectangle.loc[rectangle['location'] == "front", 'y'].to_numpy()
            w1 = rectangle.loc[rectangle['location'] == "front", 'w'].to_numpy()
            h1 = rectangle.loc[rectangle['location'] == "front", 'h'].to_numpy()

            x2 = rectangle.loc[rectangle['location'] == "back", 'x'].to_numpy()
            y2 = rectangle.loc[rectangle['location'] == "back", 'y'].to_numpy()
            w2 = rectangle.loc[rectangle['location'] == "back", 'w'].to_numpy()
            h2 = rectangle.loc[rectangle['location'] == "back", 'h'].to_numpy()
            if segment == "r_armpit" or segment == "l_armpit":
                segment = "armpit"
            if segment == "r_hand" or segment == "l_hand":
                segment = "hand"
            if segment == "r_knee" or segment == "l_knee":
                segment = "knee"
            if segment == "r_foot" or segment == "l_foot":
                segment = "feet"

            for j in range(0, len(sex)):
                if "(Self)" in titles[i]:
                    if x1 <= sex.iloc[j, 3] < x1 + w1 and y1 <= sex.iloc[j, 2] < y1 + h1:
                        sex.loc[(sex['self_x'] == sex.iloc[j, 3]) & (sex['self_y'] == sex.iloc[j, 2]), ['segment']] = [
                            segment]
                        sex.loc[(sex['self_x'] == sex.iloc[j, 3]) & (sex['self_y'] == sex.iloc[j, 2]), ['location']] = [
                            "front"]
                        sex.loc[(sex['self_x'] == sex.iloc[j, 3]) & (sex['self_y'] == sex.iloc[j, 2]), ['type']] = [
                            "self"]

                    if x2 <= sex.iloc[j, 3] < x2 + w2 and y2 <= sex.iloc[j, 2] < y2 + h2:
                        sex.loc[(sex['self_x'] == sex.iloc[j, 3]) & (sex['self_y'] == sex.iloc[j, 2]), ['segment']] = [
                            segment]
                        sex.loc[(sex['self_x'] == sex.iloc[j, 3]) & (sex['self_y'] == sex.iloc[j, 2]), ['type']] = [
                            "self"]
                        sex.loc[(sex['self_x'] == sex.iloc[j, 3]) & (sex['self_y'] == sex.iloc[j, 2]), ['location']] = [
                            "back"]

                elif "(Others)" in titles[i]:
                    if x1 <= sex.iloc[j, 3] < x1 + w1 and y1 <= sex.iloc[j, 2] < y1 + h1:
                        sex.loc[
                            (sex['other_x'] == sex.iloc[j, 3]) & (sex['other_y'] == sex.iloc[j, 2]), ['segment']] = [
                            segment]
                        sex.loc[(sex['other_x'] == sex.iloc[j, 3]) & (sex['other_y'] == sex.iloc[j, 2]), ['type']] = [
                            "other"]
                        sex.loc[
                            (sex['other_x'] == sex.iloc[j, 3]) & (sex['other_y'] == sex.iloc[j, 2]), ['location']] = [
                            "front"]

                    if x2 <= sex.iloc[j, 3] < x2 + w2 and y2 <= sex.iloc[j, 2] < y2 + h2:
                        sex.loc[
                            (sex['other_x'] == sex.iloc[j, 3]) & (sex['other_y'] == sex.iloc[j, 2]), ['segment']] = [
                            segment]
                        sex.loc[(sex['other_x'] == sex.iloc[j, 3]) & (sex['other_y'] == sex.iloc[j, 2]), ['type']] = [
                            "other"]
                        sex.loc[
                            (sex['other_x'] == sex.iloc[j, 3]) & (sex['other_y'] == sex.iloc[j, 2]), ['location']] = [
                            "back"]

        duplicate_counts = (
            sex.groupby(['id', 'segment', 'location'])
            .size()
            .reset_index(name='points')
        )

        info_df = sex[['id', 'gender', 'segment', 'location', 'type']].drop_duplicates()

        result_df = pd.merge(duplicate_counts, info_df, on=['id', 'segment', 'location'], how='left')
        result_df = result_df[['id', 'points', 'gender', 'segment', 'location', 'type']]

        data_sorted = pd.concat([data_sorted, result_df], ignore_index=True)

    data_sorted.to_excel("source/output/data_sorted_all(1).xlsx")


def plot_sex(df1, df2, df3, names, image):
    """
    Take both self and others data and plot the results of how many points are within a Region of Interest (ROI).
    The results are plotted in terms of percentages and are separated by gender (female/male) and type (self/others)
    It also takes the segments DataFrame and add new columns, such as "points" (number of points counted on each
    segment), "gender" and "type" to further analysis.
    :param df1: "self" data
    :param df2: "others" data
    :param df3: segments as DataFrame
    :param names: name of segments
    :param image: body silhouette to plot
    :return: Nothing
    """

    fig, axes = plt.subplots(nrows=2, ncols=2)
    axes = axes.flatten()

    df_self1 = df1[df1["gender"] != "male"]
    df_self2 = df1[df1["gender"] != "female"]
    df_other1 = df2[df2["gender"] != "male"]
    df_other2 = df2[df2["gender"] != "female"]

    dfs = [df_self1, df_self2, df_other1, df_other2]
    titles = ["Females (Self)", "Males (Self)", "Females (Others)", "Males (Others)"]

    data_sorted = pd.DataFrame()

    for i, sex in enumerate(dfs):

        print("Working on", titles[i])

        rectangle = df3.set_index('segment').loc["front_side", ['location', 'x', 'y', 'w', 'h']]
        whole_front, whole_back = count_whole(sex, rectangle)

        for segment in names:
            rectangle = df3.set_index('segment').loc[segment, ['location', 'x', 'y', 'w', 'h']]

            front_points, back_points = count_inside(sex, rectangle)

            front_points = len(front_points)
            back_points = len(back_points)

            f_percentage = truncate(front_points * 100 / whole_front, 1)
            b_percentage = truncate(back_points * 100 / whole_back, 1)

            df3.loc[
                (df3['segment'] == segment) & (df3['location'] == "front"),
                ['points']] = [front_points]

            df3.loc[
                (df3['segment'] == segment) & (df3['location'] == "back"),
                ['points']] = [back_points]

            df3.loc[
                (df3['segment'] == segment) & (df3['location'] == "front"),
                ['percentage']] = [f_percentage]

            df3.loc[
                (df3['segment'] == segment) & (df3['location'] == "back"),
                ['percentage']] = [b_percentage]

            if i == 0:
                df3.loc[
                    (df3['segment'] == segment),
                    ['gender']] = ["female"]
                df3.loc[
                    (df3['segment'] == segment),
                    ['type']] = ["self"]
            elif i == 1:
                df3.loc[
                    (df3['segment'] == segment),
                    ['gender']] = ["male"]
                df3.loc[
                    (df3['segment'] == segment),
                    ['type']] = ["self"]

            elif i == 2:
                df3.loc[
                    (df3['segment'] == segment),
                    ['gender']] = ["female"]
                df3.loc[
                    (df3['segment'] == segment),
                    ['type']] = ["other"]
            elif i == 3:
                df3.loc[
                    (df3['segment'] == segment),
                    ['gender']] = ["male"]
                df3.loc[
                    (df3['segment'] == segment),
                    ['type']] = ["other"]

        new_segments = df3[df3["points"].notnull()]
        new_segments = new_segments.astype({"points": int})
        data_sorted = pd.concat([data_sorted, new_segments], ignore_index=True)

        new_segments.loc[:, 'colors'] = ["#00DBFF"] * new_segments.shape[0]

        new_segments.loc[
            (new_segments['points'] == new_segments.iloc[:len(names)]["points"].max()),
            ['colors']] = ["#FF2400"]
        new_segments.loc[
            (new_segments['points'] == new_segments.iloc[:len(names)]["points"].min()),
            ['colors']] = ["#FFD700"]
        new_segments.loc[
            (new_segments['points'] == new_segments.iloc[len(names):]["points"].max()),
            ['colors']] = ["#FF2400"]
        new_segments.loc[
            (new_segments['points'] == new_segments.iloc[len(names):]["points"].min()),
            ['colors']] = ["#FFD700"]

        hmax = sns.scatterplot(data=new_segments,
                               x="xc",
                               y="yc",
                               size="points",
                               sizes=(250, 1000),
                               legend=False,
                               ax=axes[i],
                               hue="colors",
                               alpha=0.5)

        for line in range(0, 38):
            try:
                hmax.text(
                    new_segments["xc"][line],
                    new_segments["yc"][line],
                    new_segments["points"][line],
                    ha="left",
                    weight="bold"
                )
            except KeyError:
                pass
        hmax.imshow(image)
        hmax.invert_yaxis()
        hmax.invert_xaxis()

        hmax.set_xlim(hmax.get_xlim()[::-1])
        hmax.set_ylim(hmax.get_ylim()[::-1])

        axes[i].axis("off")
        axes[i].set_title(titles[i])

    plt.show()
    # data_sorted.to_excel("source/output/data_sorted.xlsx")


def plot_countries(df1, df2, df3, names, image):
    """
    Take either self or others data and plot the results of each country regarding the zones they like to sniff or
    be sniffed. Plotted as percentages (ROI vs Whole)

    :param df1: "self" data
    :param df2: "others" data
    :param df3: segments as DataFrame
    :param names: name of segments
    :param image: body silhouette to plot
    :return: Nothing
    """

    # Get countries names
    countries = df1["country"].unique()
    dfs = [df1, df2]

    for f, df in enumerate(dfs):
        fig, axes = plt.subplots(nrows=2, ncols=9)
        axes = axes.flatten()

        for i, country in enumerate(countries):
            df_target = df.loc[df["country"] == country]

            print("Sorting data...", country)

            rectangle = df3.set_index('segment').loc["front_side", ['location', 'x', 'y', 'w', 'h']]
            whole_front, whole_back = count_whole(df_target, rectangle)

            for segment in names:
                rectangle = df3.set_index('segment').loc[segment, ['location', 'x', 'y', 'w', 'h']]
                front_points, back_points = count_inside(df_target, rectangle)

                front_points = int(len(front_points))
                back_points = int(len(back_points))

                f_percentage = truncate(front_points * 100 / whole_front, 1)
                b_percentage = truncate(back_points * 100 / whole_back, 1)

                df3.loc[
                    (df3['segment'] == segment) & (df3['location'] == "front"),
                    ['percentage']] = [f_percentage]

                df3.loc[
                    (df3['segment'] == segment) & (df3['location'] == "back"),
                    ['percentage']] = [b_percentage]

            new_segments = df3[df3["percentage"].notnull()]
            new_segments = new_segments.astype({"percentage": int})

            hmax = sns.scatterplot(data=new_segments, x="xc", y="yc", size="percentage", sizes=(250, 1000),
                                   legend=False,
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
            hmax.imshow(image)
            hmax.invert_yaxis()
            hmax.invert_xaxis()

            hmax.set_xlim(hmax.get_xlim()[::-1])
            hmax.set_ylim(hmax.get_ylim()[::-1])
            axes[i].axis("off")
            axes[i].set_title(country)

        fig.delaxes(axes[17])
        if f == 0:
            fig.suptitle('SELF', fontsize=16)
        elif f == 1:
            fig.suptitle('OTHERS', fontsize=16)
        plt.show()
        print("--------------------------------------------------")


def truncate(number, digits) -> float:
    """
    Function by https://stackoverflow.com/users/541420/erwin-mayer
    :param number: input number
    :param digits: target decimals to be left
    :return: float number with target decimals
    """
    # Improve accuracy with floating point operations, to avoid truncate(16.4, 2) = 16.39 or truncate(-1.13, 2) = -1.12
    nbDecimals = len(str(number).split('.')[1])
    if nbDecimals <= digits:
        return number
    stepper = 10.0 ** digits
    return math.trunc(stepper * number) / stepper


def count_whole(data, rectangle_range):
    """
    Takes an area and search for every data that fits within that area, data has x,y coordinates.
    Calculates the number of points inside the area for front and back draws.
    This function will take the whole body picture, to have the total amount of points for the front and
    for the back
    :param data: target data
    :param rectangle_range: target rectangle range
    :return: f = points on the front (whole picture) ; b = points on the back (whole picture)
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
    Takes an area and search for every data that fits within that area, data has x,y coordinates.
    Calculates the number of points inside the area for front and back draws
    :param data: target data
    :param rectangle_range: target rectangle range
    :return: f = points on the front ; b = points on the back
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
    # Load data from questionnaire
    questionnaire = pd.read_excel("source/data/smell_behavior_sociodemographics.xlsx")

    # Load self and other body odor dataframes
    self = pd.read_excel("source/data/body_silhouettes_self.xlsx")
    other = pd.read_excel("source/data/body_silhouettes_other.xlsx")

    # Extract valid entries (id) from cleaned questionnaire dataframe and filter for valid entries
    included = questionnaire["id"].unique()
    self = self[self['id'].isin(included)]
    other = other[other['id'].isin(included)]

    # Let's create DataFrames for gender-oriented analysis
    self_sex = self
    self_sex = self_sex.merge(questionnaire[['id', 'gender']], on='id', how='left')

    other_sex = other
    other_sex = other_sex.merge(questionnaire[['id', 'gender']], on='id', how='left')

    # Load body segments and image
    segments = pd.read_excel("source/data/body_segments.xlsx")
    map_img = mpimg.imread('source/img/humanbody_clear.png')  # change path to image path

    # Just in case, a set of colors for plotting
    colors = ["#FF6B6B", "#FFA94D", "#FFD43B", "#69DB7C", "#38D9A9", "#4DABF7", "#5C7CFA",
              "#9775FA", "#DA77F2", "#F783AC", "#ADB5BD", "#343A40", "#FFC9E3", "#B2F2BB"]

    # Names of all the segments of interest
    segments_names = ["hair", "mouth", "neck", "chest", "r_armpit", "r_hand", "l_armpit",
                      "l_hand", "pelvis", "r_knee", "r_foot", "l_knee", "l_foot"]
    # plot_countries(self, other, segments, segments_names, map_img)
    plot_sex(self_sex, other_sex, segments, segments_names, map_img)
    # sort_to_export(self_sex, other_sex, segments, segments_names)

# https://pingouin-stats.org/build/html/generated/pingouin.rm_anova.html
# https://pingouin-stats.org/build/html/generated/pingouin.pairwise_tests.html
