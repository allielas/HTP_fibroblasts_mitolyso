"""
Helper functions to build superplots
Allie Spangaro, Toronto Metropolitan University
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from plate_preprocessing import *
from plate_information import getpairs

import scipy
import seaborn as sns

"""
See this awesome course note here: https://biapol.github.io/PoL-BioImage-Analysis-TS-Early-Career-Track/day3a_plotting/03_Statistic_Annotations_in_Seaborn_Bonus.html

NOTE for custom annotations in statannotations:
    Use results from a previous stat test e.g.
    stat_results_GMF = [mannwhitneyu(Gentoo_values_female['bill_length_mm'], Gentoo_values_male['bill_length_mm'], alternative="two-sided"),]
    pvalues = [result.pvalue for result in stat_results_GMF]
    formatted_pvalues = [f"p={p:.2e}" for p in pvalues]
    annotator.set_custom_annotations(formatted_pvalues)
"""


def create_superplot_fortwo(data, x_value, y_value, replicates, save_name):
    sns.set_style("whitegrid")

    # Calculate the mean for each group
    group_averages = data.groupby([x_value, replicates], as_index=False).agg(
        {y_value: "mean"}
    )
    group_ave_pivot = group_averages.pivot_table(
        columns=x_value, values=y_value, index=replicates
    )

    # Perform paired t-test
    statistic, pvalue = scipy.stats.ttest_rel(
        group_ave_pivot.iloc[:, 0], group_ave_pivot.iloc[:, 1]
    )
    pvalue_str = str(float(round(pvalue, 3)))
    if pvalue < 0.05:
        pvalue_str = pvalue_str + "*"

    # Create the superplot
    sns.swarmplot(x=x_value, y=y_value, hue=replicates, data=data)  # plot the data
    ax = sns.swarmplot(
        x=x_value,
        y=y_value,
        hue=replicates,
        size=15,
        edgecolor="k",
        linewidth=2,
        data=group_averages,
    )  # plot the averages from each replicate
    ax.legend_.remove()  # remove the legend

    x1, x2 = 0, 1
    y, h, col = data[y_value].max() + 2, 2, "k"

    # Add the p-value annotation
    plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
    plt.text(
        (x1 + x2) * 0.5,
        y + h * 2,
        "p = " + pvalue_str,
        ha="center",
        va="bottom",
        color=col,
    )

    # Save the plot
    plt.savefig(save_name)
    plt.show()


def create_superplot(df, ax):
    """Derived from Marco Dalla Vecchia, December 6, 2024
    Actually consider the replicates
    Consider mean value of each replicate per treatment
    directly taken from S5 of original publication

    Args:
        df (DataFrame): _description_
        ax (MatPlotLib Axis): _description_

    Returns:
        MatPlotLib Axis): _description_
    """
    from scipy.stats import ttest_rel, ttest_ind
    import matplotlib.ticker

    ReplicateAverages = df.groupby(["Treatment", "Replicate"], as_index=False).agg(
        {"Speed": "mean"}
    )
    ReplicateAvePivot = ReplicateAverages.pivot_table(
        columns="Treatment", values="Speed", index="Replicate"
    )
    # Calculate 'appropriate' p-value considering n=3
    good_pvalue = ttest_rel(
        ReplicateAvePivot["Control"], ReplicateAvePivot["Drug"]
    ).pvalue

    # Just to copy the colors in the paper (https://doi.org/10.1083/jcb.202001064) I extract the RGB values from the original figure
    paper_palette = [
        (0.792156862745098, 0.5529411764705883, 0.1411764705882353),
        (0.36470588235294116, 0.6274509803921569, 0.7490196078431373),
        (0.5803921568627451, 0.592156862745098, 0.592156862745098),
    ]

    # Instead of making the background swarmplot white I kept the corresponding categorical color
    # but lower the opacity down by 50%
    # this is the largest difference with the original figure
    alpha_palette = [(r, g, b, 0.5) for (r, g, b) in paper_palette]

    sns.swarmplot(  # plot all data points by replicate as swarmplot
        data=df,
        x="Treatment",
        y="Speed",
        hue="Replicate",
        size=4,
        zorder=0,
        palette=alpha_palette,
        linewidth=1,
        legend=False,
        ax=ax,
    )

    sns.pointplot(  # one dot representing mean of each replicate separated by color and marker type
        data=df,
        x="Treatment",
        y="Speed",
        hue="Replicate",
        palette=paper_palette,
        linestyle="none",
        errorbar=None,
        markers=["o", "s", "^"],
        dodge=True,
        markeredgecolor="k",
        markeredgewidth=1.1,
        legend=False,
        ax=ax,
    )

    sns.pointplot(  # standard error bars of mean values of each replicate
        data=ReplicateAverages,
        x="Treatment",
        y="Speed",
        linestyle="none",
        capsize=0.3,
        errorbar="se",
        err_kws={"linewidth": 1.4},
        marker="_",
        markersize=50,
        markeredgewidth=1.4,
        color="black",
        legend=False,
        ax=ax,
    )

    # Plot adjustments
    sns.despine()
    ax.set_ylabel(r"Speed ($\mu$m/min)")
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0, 60)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(20))

    x1, x2 = 0, 1
    y, h = df["Speed"].max() + 2, 2

    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c="black")
    ax.text(
        (x1 + x2) * 0.5,
        y + h * 2,
        "P = {:.3f}".format(good_pvalue),
        ha="center",
        va="bottom",
    )
    ax.set_facecolor("#F0F0F7")
    return ax


def remove_outliers_iqr(df, col=None):
    """Remove outliers based on interquartile range

    Args:
        df (DataFrame): _description_
        col (string, optional): the name of the column to remove from. Defaults to None, removes all by default.

    Returns:
        _type_: _description_
    """
    cols = df.select_dtypes(
        "number"
    ).columns  # limits to a (float), b (int) and e (timedelta)
    df_sub = df.loc[:, cols]

    iqr = df_sub.quantile(0.75, numeric_only=False) - df_sub.quantile(
        0.25, numeric_only=False
    )

    # calculate  extreme outlisers by dividing median by iqr
    lim = np.abs((df_sub - df_sub.median()) / iqr) < 2.22

    # replace outliers with nan
    df.loc[:, cols] = df_sub.where(lim, np.nan)
    df.dropna(subset=cols, inplace=True)  # drop rows with NaN in numerical columns
    return df


def shapiro_pvalue(
    group_avg_df,
    replicate,
    feature_meas,
    replicate_col_name="Replicate_Number",
    debug=False,
):
    """Function to apply the shapiro wilk test to a dataframe aggregated by replicate for a single feature

    Args:
        group_avg_df (DataFrame): the aggregated dataframe
        replicate (string, int): string or int representation of replicate number
        feature_meas (string): _description_
        replicate_col_name (str, optional): the name of the replicate column. Defaults to "Replicate_Number".
        debug (bool, optional): print out p values. Defaults to False.

    Returns:
        float: p_value from test on that replicate
    """
    from scipy.stats import shapiro

    rep_df = group_avg_df[group_avg_df[replicate_col_name] == replicate]
    # Assume 'df' is your DataFrame and 'feature_meas' is the column to test
    stat, p_value = shapiro(rep_df[feature_meas].dropna())

    if debug:
        print(f"Shapiro-Wilk statistic: {stat}, p-value: {p_value}")
        if p_value < 0.05:
            print("Data is not normally distributed (reject H0)")
        else:
            print("Data is normally distributed (fail to reject H0)")
    return p_value


def apply_shapiro_wilk_test_to_df(
    group_avg_df, feature_meas, replicate_col_name="Replicate_Number", alpha=0.05
):
    """Function to applies the shapiro wilk test row-by-row onto an aggregated dataframe by replicate

    Args:
        group_avg_df (DataFrame): the aggregated dataframe
        feature_meas (string): _description_
        replicate_col_name (str, optional): the name of the replicate column. Defaults to "Replicate_Number".
        alpha (float): the p value threshold. Defaults to p=0.05

    Returns:
        DataFrame: The aggregated dataframe with a "Shaprio_pvalue" column and a boolean "Shapiro_normality" column
    """
    # apply the shapiro-wilk test to a dataframe
    group_avg_df = group_avg_df.dropna()
    group_avg_df["Shapiro_pvalue"] = group_avg_df.apply(
        lambda row: shapiro_pvalue(
            group_avg_df, replicate=row[replicate_col_name], feature_meas=feature_meas
        ),
        axis=1,
    )
    # reject null hypothesis if p < 0.05 - i.e. significant chance that the data is not normally distributed
    group_avg_df["Shapiro_normality"] = group_avg_df.apply(
        lambda row: row["Shapiro_pvalue"] > alpha, axis=1
    )
    return group_avg_df


def average_groups_pivot(group_avg_df, x_value, y_value, replicate_col_name):
    """Make a pivot table from the averaged dataframe

    Args:
        group_avg_df (DataFrame): your dataframe output from average_groups_by_plate()
        x_value (string): the grouping variable (x value)
        y_value (string): the quantitavie feature to measure (y value)
        replicate_col_name (string): the variable representing experimental replicates for grouping

    Returns:
        DataFrame: a pivot table
    """
    group_avg_df_pivot = group_avg_df.pivot_table(
        columns=x_value, values=y_value, index=replicate_col_name
    )
    return group_avg_df_pivot


def pvalues_anova_and_tukeyhsd_posthoc(
    data_df,
    pivot_df,
    x_value,
    y_value,
    replicate_number_col="Replicate_Number",
    desired_pairs=None,
    order=None,
):
    """Perform Tukey's HSD post-hoc test on the data.
    See https://github.com/4dcu-be/CodeNuggets/blob/main/Post%20hoc%20tests%20with%20statannotations.ipynb
    Also https://www.biorxiv.org/content/10.1101/2025.02.02.636071v1.full.pdf
    Args:
        data_df (pd.DataFrame): DataFrame table containing the data.
        pivot_df (pd.DataFrame): DataFrame pivot table containing the means.
        x_value (str): Column name for the independent variable.
        y_value (str): Column name for the dependent variable.
        grouping_variable (str): Column name for the replicate number.
        order (list, optional): Order of groups for plotting. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame with Tukey's HSD results.
    """
    from scipy.stats import f_oneway
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    groups = []  # Convert pivot table to list of groups
    # display(pivot_df)
    for col in pivot_df:
        if col == x_value or col == replicate_number_col:
            # print("Skipping col:" + col)
            continue  # Skip the first column (usually the index or grouping variable)
        else:
            # print("Adding col:" + col)
            groups.append(pivot_df[col].dropna())
    # One-Way ANOVA
    # display(groups)
    f_value, p_value_anova = f_oneway(
        *list(groups)
    )  # Pass groups as args to run ANOVA on all groups
    print(f"ANOVA F statistic: {f_value}")
    print(f"ANOVA p value: {p_value_anova}")

    # Tukey's HSD (post hoc test)
    if p_value_anova < 0.05:
        tukey_result = pairwise_tukeyhsd(
            endog=data_df[y_value], groups=data_df[x_value], alpha=0.05
        )
        # Extract the data from the Statsmodels SimpleTable
        tukey_data = tukey_result._results_table.data[1:]  # Exclude the header line
        headers = tukey_result._results_table.data[0]  # Get the header line
        tukey_result_df = pd.DataFrame(tukey_data, columns=headers)
        tukey_result_pairs = tukey_result_df[["group1", "group2"]].itertuples(
            index=False, name=None
        )
        pairs = list(tukey_result_pairs)
        p_values = tukey_result_df["p-adj"].tolist()
        # display(tukey_result_df)
        return (pairs, p_values)
    else:
        print("ANOVA test is not significant, skipping Tukey's HSD post-hoc test.")
        return ([], [])


def anova_with_tukey_posthoc(
    data_df,
    x_value,
    y_value,
    replicate_number_col="Replicate_Number",
    desired_pairs=None,
    order=None,
    display_results=False,
):
    from scikit_posthocs import posthoc_tukey

    # Make groups [x,y] for tukey test
    groups = np.unique(data_df[x_value])
    data = []
    for group in groups:
        data.append(data_df[data_df[x_value] == group][y_value])

    anova_result = stats.f_oneway(*data)
    anova_result_pvalue = anova_result.pvalue
    print(f"One-way ANOVA F statistic: {anova_result.statistic}")
    print(f"ANOVA p value: {anova_result_pvalue}")

    if anova_result_pvalue < 0.05:
        # posthoc turkey test
        tukey_df = posthoc_tukey(data_df, val_col=y_value, group_col=x_value)

        # melt the dunn_df to long format
        remove = np.tril(np.ones(tukey_df.shape), k=0).astype("bool")
        tukey_df[remove] = np.nan
        molten_df = tukey_df.melt(ignore_index=False).reset_index().dropna()

        if display_results:
            # display(tukey_df)
            print(molten_df)

        dunn_pairs = molten_df[["index", "variable"]].itertuples(index=False, name=None)
        pairs = list(dunn_pairs)
        p_values = molten_df["value"].tolist()
        return (pairs, p_values)

    else:
        print("Oneway ANOVA is not significant, skipping Tukey's post-hoc test.")
        return ([], [])


def kruskal_with_dunn_posthoc(
    data_df,
    x_value,
    y_value,
    replicate_number_col="Replicate_Number",
    desired_pairs=None,
    order=None,
    p_correction="fdr_by",  # graphpad reccomneds two-step Benjamini/Yekutieli method
    display_results=False,
):
    from scikit_posthocs import posthoc_dunn

    # Make groups [x,y] for kruskal test
    groups = np.unique(data_df[x_value])
    data = []
    for group in groups:
        data.append(data_df[data_df[x_value] == group][y_value])

    kruskal_result = stats.kruskal(*data)
    kruskal_pvalue = kruskal_result.pvalue
    print(f"Kruskal-Wallis H statistic: {kruskal_result.statistic}")
    print(f"Kruskal-Wallis p value: {kruskal_pvalue}")

    if kruskal_pvalue < 0.05:
        # posthoc dunn test
        dunn_df = posthoc_dunn(
            data_df, val_col=y_value, group_col=x_value, p_adjust=p_correction
        )
        # melt the dunn_df to long format
        remove = np.tril(np.ones(dunn_df.shape), k=0).astype("bool")
        dunn_df[remove] = np.nan
        molten_df = dunn_df.melt(ignore_index=False).reset_index().dropna()

        if display_results:
            # display(dunn_df)
            print(molten_df)
        dunn_pairs = molten_df[["index", "variable"]].itertuples(index=False, name=None)
        pairs = list(dunn_pairs)
        p_values = molten_df["value"].tolist()
        return (pairs, p_values)

    else:
        print("Kruskal-Wallis test is not significant, skipping Dunn's post-hoc test.")
        return ([], [])


def annotate_with_anova_tukey(
    ax,
    pairs,
    data,
    x_value,
    y_value,
    replicate_col_name="Replicate_Name",
    order=None,
    plot="violinplot",
):
    """Add statistical annotations to the plot using one-way ANOVA test.
    see https://statannotations.readthedocs.io/en/latest/custom-test.html for more examples

    Args:
        ax (_type_): _description_
        pairs (_type_): _description_
        group_avg_df (_type_): _description_
        x_value (_type_): _description_
        y_value (_type_): _description_
        order (_type_, optional): _description_. Defaults to None.
        plot (str, optional): _description_. Defaults to "violinplot".

    Returns:
        _type_: _description_
    """
    from statannotations.Annotator import Annotator, StatTest
    from scipy.stats import tukey_hsd

    custom_long_name = "Pairwise Tukey HSD"
    custom_short_name = "tukey"
    custom_func = tukey_hsd
    tukey = StatTest(custom_func, custom_long_name, custom_short_name)
    # tukey = StatTest(tukey_hsd, custom_long_name, custom_short_name)

    # load the custom test
    annotator = Annotator(
        ax, pairs, data=data, order=order, plot=plot
    )  # x=x_value, y=y_value, hue=replicate_col_name,
    annotator.reset_configuration()
    annotator.configure(
        test=tukey,
        text_format="star",  #'simple','full'
        loc="inside",
        hide_non_significant=True,
        color="black",
        verbose=2,
    )
    annotator.apply_and_annotate()
    return ax


def annotate_pairs_with_calculated_pvalues(
    ax,
    data,
    pivot_data,
    x_value,
    y_value,
    replicate_col_name="Replicate_Name",
    test_name="tukey",
    pairs=None,
    order=None,
    plot="violinplot",
    show_test_name=False,
):
    """Add statistical annotations to the plot using Tukey's HSD test.
    see https://statannotations.readthedocs.io/en/latest/custom-test.html for more examples

    Args:
        ax (_type_): _description_
        pairs (_type_): _description_
        data (_type_): _description_
        x_value (_type_): _description_
        y_value (_type_): _description_
        replicate_col_name (str, optional): _description_. Defaults to "Replicate_Name".
        test_name (str, optional): _description_. Defaults to "tukey".
        pairs (_type_, optional): _description_. Defaults to None.
        order (_type_, optional): _description_. Defaults to None.
        plot (str, optional): _description_. Defaults to "violinplot".

    Returns:
        _type_: _description_
    """
    from statannotations.Annotator import Annotator

    if pairs is None:
        pairs = getpairs(data, x_value, order=order)

    if test_name in ["kruskal", "dunn", "kruskal-wallis"]:
        used_pairs, p_values = kruskal_with_dunn_posthoc(
            data,
            x_value=x_value,
            y_value=y_value,
            order=order,
            desired_pairs=pairs,
            p_correction="fdr_bh",
            display_results=True,
        )
    elif test_name in ["anova", "tukey", "tukeyhsd"]:
        # perform anova and tukey's post-hoc test
        used_pairs, p_values = pvalues_anova_and_tukeyhsd_posthoc(
            data, pivot_data, x_value, y_value, order=order, desired_pairs=pairs
        )
    elif test_name == "tukey_v2":
        used_pairs, p_values = anova_with_tukey_posthoc(
            data,
            x_value=x_value,
            y_value=y_value,
            replicate_number_col=replicate_col_name,
            order=order,
            desired_pairs=pairs,
        )
    else:
        raise ValueError(
            f"Test name '{test_name}' is invalid. Use 'tukey', 'anova', 'kruskal', or 'dunn'."
        )
    if used_pairs is None or len(used_pairs) == 0:
        print(f"No significant pairs found for the {test_name} test.")
        return ax
    else:
        annotator = Annotator(
            ax=ax,
            pairs=list(used_pairs),
            data=data,
            plot=plot,
            x=x_value,
            y=y_value,
            order=order,
        )
        annotator.reset_configuration()
        annotator.configure(
            text_format="full",
            test_short_name=test_name,
            pvalue_format_string="{:.3f}",
            # pvalue_format = [[1e-5, "1e-5"], [1e-4, "1e-4"], [1e-3, "0.001"], [1e-2, "0.01"], [5e-2, "0.05"]],
            loc="inside",
            hide_non_significant=True,
            color="black",
            verbose=2,
            show_test_name=show_test_name,
        )
        annotator.set_pvalues_and_annotate(p_values)
        return ax


def annotate_with_kruskal(
    ax,
    pairs,
    data,
    x_value,
    y_value,
    replicate_col_name="Replicate_Name",
    order=None,
    plot="violinplot",
):
    """Add statistical annotations to the plot using Kruskal-Wallis test.
    see https://statannotations.readthedocs.io/en/latest/custom-test.html for more examples

    Args:
        ax (_type_): _description_
        pairs (_type_): _description_
        data (_type_): _description_
        x_value (_type_): _description_
        y_value (_type_): _description_
        order (_type_, optional): _description_. Defaults to None.
        plot (str, optional): _description_. Defaults to "violinplot".

    Returns:
        _type_: _description_
    """
    from statannotations.Annotator import Annotator

    annotator = Annotator(
        ax,
        pairs=pairs,
        data=data,
        order=order,
        plot=plot,  # , x=x_value, y=y_value
    )
    annotator.reset_configuration()
    annotator.configure(
        test="Kruskal",
        text_format="simple",
        # pvalue_format = [[1e-5, "1e-5"], [1e-4, "1e-4"], [1e-3, "0.001"], [1e-2, "0.01"], [5e-2, "0.05"]],
        loc="inside",
        hide_non_significant=True,
        color="black",
        verbose=2,
    )
    annotator.apply_and_annotate()
    return ax


def annotate_legend_with_shapiro(
    ax,
    group_avg_df,
    group_col_name,
    shapiro_col_name="Shapiro_normality",
    palette="pastel",
    title="Replicate",
):
    """add an annotation to the legend of an axis if there is normality via shapiro test

    Args:
        ax (_type_): _description_
        group_avg_df (_type_): _description_
        replicate_col_name (_type_): _description_
    """
    import matplotlib.lines as mlines

    unique_replicates = group_avg_df[group_col_name].unique()
    unique_replicates = sorted(unique_replicates)
    L = plt.legend()
    custom_labels = []
    for rep in unique_replicates:
        label = str(rep)
        shapiro_val = group_avg_df[group_avg_df[group_col_name] == rep][
            shapiro_col_name
        ].iloc[0]
        if shapiro_val:
            label += " (normal)"
        custom_labels.append(label)

    # Create custom legend handles (using the same colors as swarmplot)
    palette = sns.color_palette(palette, n_colors=len(unique_replicates))
    handles = [
        mlines.Line2D(
            [],
            [],
            color=palette[i],
            marker="o",
            linestyle="None",
            markersize=12,
            markeredgecolor="black",
            label=custom_labels[i],
        )
        for i in range(len(unique_replicates))
    ]
    ax.legend_.set_title(title)
    ax.legend(handles=handles, title=title, loc="best")
    return ax


def annotate_legend_replicatesonly(
    ax,
    group_avg_df,
    group_col_name,
    palette="pastel",
    title="Replicate",
):
    """add an annotation to the legend of an axis to show color coded replicates

    Args:
        ax (_type_): _description_
        group_avg_df (_type_): _description_
        replicate_col_name (_type_): _description_
    """
    import matplotlib.lines as mlines

    unique_replicates = group_avg_df[group_col_name].unique()
    unique_replicates = sorted(unique_replicates)
    L = plt.legend()
    custom_labels = []
    for rep in unique_replicates:
        label = str(rep)
        custom_labels.append(label)

    # Create custom legend handles (using the same colors as swarmplot)
    palette = sns.color_palette(palette, n_colors=len(unique_replicates))
    handles = [
        mlines.Line2D(
            [],
            [],
            color=palette[i],
            marker="o",
            linestyle="None",
            markersize=12,
            markeredgecolor="black",
            label=custom_labels[i],
        )
        for i in range(len(unique_replicates))
    ]
    ax.legend_.set_title(title)
    ax.legend(handles=handles, title=title, loc="best")
    return ax


def super_splitviolinplot_helper(
    data_df,
    group_avg_df,
    ax,
    x_value,
    y_value,
    title,
    replicate_col_name,
    pairs=None,
    order=None,
    annotate=False,
    test=None,
    shapiro=True,
    show_test_on_plot=False,
):
    if pairs is None:
        pairs = getpairs(data_df, x_value, order=order)
    print(pairs)
    sns.violinplot(
        data=data_df,
        x=x_value,
        y=y_value,  # hue=x_value,
        # palette="Set2",
        split=True,  # using split violin plots - only one side, basically looks like a histogram
        inner="quart",
        color="gainsboro",
        width=0.9,
        linewidth=1.5,
        order=order,
        ax=ax,
    )
    sns.swarmplot(
        data=group_avg_df,
        x=x_value,
        y=y_value,
        hue=replicate_col_name,
        order=order,
        palette="pastel",
        size=12,
        edgecolor="k",
        linewidth=1,
        dodge=False,
        ax=ax,
    )
    # draw a boxplot to show the mean line
    sns.boxplot(
        data=group_avg_df,
        x=x_value,
        y=y_value,
        showmeans=True,
        meanline=True,
        meanprops={"color": "dimgray", "ls": "-", "lw": 2.5},
        medianprops={"visible": False},
        whiskerprops={"visible": False},
        zorder=2,
        showfliers=False,
        showbox=False,
        showcaps=False,
        ax=ax,
    )
    ax.set_title(title)

    # axes[0].text(
    #     x=row[x_value],
    #     y=row[y_value],
    #     s=str(row["Shapiro_normality"]),
    #     color="black",
    #     fontsize=10,
    #     ha="center"
    # )
    # use pivot table to get the average values for each group
    if annotate and test is not None:
        group_avg_pivot_table = average_groups_pivot(
            group_avg_df, x_value, y_value, replicate_col_name
        )
        try:
            ax = annotate_pairs_with_calculated_pvalues(
                ax,
                group_avg_df,
                group_avg_pivot_table,
                x_value,
                y_value,
                replicate_col_name=replicate_col_name,
                test_name=test,
                order=order,
                plot="violinplot",
                show_test_name=show_test_on_plot,
            )
        except Exception as e:
            print(f"Error annotating with statistical test: {e}")
            # ax = annotate_with_anova_tukey(ax, pairs, group_avg_df_pivot, x_value, y_value, replicate_col_name=replicate_col_name, order=order, plot="violinplot")
        # elif test == "kruskal":
        #     ax = annotate_with_kruskal(
        #         ax,
        #         pairs,
        #         group_avg_pivot_table,
        #         x_value,
        #         y_value,
        #         order=order,
        #         replicate_col_name=replicate_col_name,
        #         plot="violinplot",
        #     )
        if shapiro:
            ax = annotate_legend_with_shapiro(ax, group_avg_df, replicate_col_name)

    return ax


def superplot_for_area_threshold_comparisons(
    data_df_1,
    group_avg_df_1,
    data_df_2,
    group_avg_df_2,
    x_value="AllGroups",
    y_value="Cell_AreaShape_Area",
    replicate_col_name="Replicate_Number",
    out_dir="",
    xtitle=None,
    ytitle=None,
    order=None,
    legend=True,
    title1="Original Dataset",
    title2="Excluding Cells Touching Borders",
    annotate=False,
    test=None,
    export_pivot=False,
    show_hist=False,
):
    """Make two side-by-side superplots to compare area between different conditions
    Args:
        data_df_1 (_type_): _description_
        group_avg_df_1 (_type_): _description_
        data_df_2 (_type_): _description_
        group_avg_df_2 (_type_): _description_
        x_value (str, optional): _description_. Defaults to "AllGroups".
        y_value (str, optional): _description_. Defaults to "Cell_AreaShape_Area".
        replicate_col_name (str, optional): _description_. Defaults to "Replicate_Number".
        csv_dir (str, optional): _description_. Defaults to "".
        xtitle (_type_, optional): _description_. Defaults to None.
        ytitle (_type_, optional): _description_. Defaults to None.
    """
    import matplotlib.lines as mlines
    from statannotations.Annotator import Annotator
    from statannotations.stats.StatTest import StatTest
    from pathlib import Path

    if order == None:
        order = get_all_group_order()
    pairs = getpairs(data_df_1, x_value, order=order)
    print(pairs)

    if show_hist:
        hist = sns.kdeplot(
            data_df_1, x=y_value, hue=replicate_col_name, palette="pastel"
        )
        plt.show()
        plt.close(hist.figure)

        hist2 = px.histogram(data_df_1, x=y_value, color=x_value)
        hist2.show()

    fig, axes = plt.subplots(1, 2, figsize=(30, 10), sharey=True, sharex=False)
    # plt.style.use("ggplot")
    sns.set_context("talk", font_scale=1.2)
    sns.set_theme(style="whitegrid")

    # First subplot: Full Dataset
    # sns.stripplot(
    #     data=data_df_1,
    #     x=x_value, y=y_value, hue=x_value,
    #     palette="Set2",
    #     order=order,
    #     ax=axes[0],
    # )
    axes[0] = super_splitviolinplot_helper(
        data_df_1,
        group_avg_df_1,
        axes[0],
        x_value,
        y_value,
        title1,
        replicate_col_name,
        order=order,
        annotate=annotate,
        test=test,
    )
    axes[1] = super_splitviolinplot_helper(
        data_df_2,
        group_avg_df_2,
        axes[1],
        x_value,
        y_value,
        title2,
        replicate_col_name,
        order=order,
        annotate=annotate,
        test=test,
    )

    axes[0].set_title(title1)
    axes[1].set_title(title2)

    if legend:
        axes[0] = annotate_legend_with_shapiro(
            axes[0], group_avg_df_1, replicate_col_name
        )
        axes[1] = annotate_legend_with_shapiro(
            axes[1], group_avg_df_2, replicate_col_name
        )
    else:
        axes[0].legend_.remove()
    if ytitle is not None:
        axes[0].set_ylabel(ytitle)
    if xtitle is not None:
        axes[0].set_xlabel(xtitle)
        axes[1].set_xlabel(xtitle)

    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, f"combined_cellsize_boxplots_{title2}_{test}.png")
    )
    plt.show()
    if export_pivot:
        pivot_dir = os.path.join(out_dir, "pivot_tables")
        pivot_dir = Path(pivot_dir)
        Path.mkdir(pivot_dir, exist_ok=True)
        group_avg_df_1_pivot = average_groups_pivot(
            group_avg_df_1, x_value, y_value, replicate_col_name
        )
        group_avg_df_2_pivot = average_groups_pivot(
            group_avg_df_2, x_value, y_value, replicate_col_name
        )
        group_avg_df_1_pivot.to_csv(
            os.path.join(pivot_dir, f"area_pivot_{title1}.csv")
        )  # can plop this into graphpad and see what it tells me
        group_avg_df_2_pivot.to_csv(os.path.join(pivot_dir, f"area_pivot_{title2}.csv"))


def single_feature_super_splitviolinplot(
    data_df,
    group_avg_df,
    x_value="AllGroups",
    y_value="Cell_AreaShape_Area",
    replicate_col_name="Replicate_Number",
    out_dir="",
    xtitle=None,
    ytitle=None,
    order=None,
    legend=True,
    annotate=False,
    test=None,
    show_hist=False,
):
    """Make a superplot to do multiple comparisons for a feature between different conditions
    Args:
        data_df_1 (_type_): _description_
        group_avg_df_1 (_type_): _description_
        data_df_2 (_type_): _description_
        group_avg_df_2 (_type_): _description_
        x_value (str, optional): _description_. Defaults to "AllGroups".
        y_value (str, optional): _description_. Defaults to "Cell_AreaShape_Area".
        replicate_col_name (str, optional): _description_. Defaults to "Replicate_Number".
        csv_dir (str, optional): _description_. Defaults to "".
        xtitle (_type_, optional): _description_. Defaults to None.
        ytitle (_type_, optional): _description_. Defaults to None.
    """
    import matplotlib.lines as mlines
    from statannotations.Annotator import Annotator
    from statannotations.stats.StatTest import StatTest
    from pathlib import Path

    if order == None:
        order = get_all_group_order()
    pairs = getpairs(data_df, x_value, order=order)
    print(pairs)

    if show_hist:
        import plotly.express as px

        hist = sns.kdeplot(data_df, x=y_value, hue=replicate_col_name, palette="pastel")
        plt.show()
        plt.close(hist.figure)

        hist2 = px.histogram(data_df, x=y_value, color=x_value)
        hist2.show()

    fig, ax = plt.subplots(figsize=(15, 8))
    # plt.style.use("ggplot")
    sns.set_context("talk", font_scale=1.2)
    sns.set_theme(style="whitegrid")

    ax = super_splitviolinplot_helper(
        data_df,
        group_avg_df,
        ax,
        x_value,
        y_value,
        title=" ",
        replicate_col_name=replicate_col_name,
        pairs=pairs,
        order=order,
        annotate=annotate,
        test=test,
    )

    if legend:
        ax = annotate_legend_with_shapiro(ax, group_avg_df, replicate_col_name)
    else:
        ax.legend_.remove()
    if ytitle is not None:
        ax.set_ylabel(ytitle)
    if xtitle is not None:
        ax.set_xlabel(xtitle)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{y_value}_{test}.png"))
    plt.show()


def make_superswarmplot_with_annotation(
    data_df,
    x_value,
    y_value,
    replicate_col_name="Replicate_Number",
    pairs=None,
    order=None,
    annotate=False,
    test="tukey",
    ytitle=None,
    xtitle=None,
    ylim=None,
    pallete="pastel",
    figsize=(8, 10),
    dpi=300,
):
    # Set values to defaults if a parameter is not loaded for order or y axis parameters
    if ytitle is None:
        ytitle = y_value.replace("_", " ")
    if ylim is None:
        ylim = (0, 100)
    if order is None:
        order = data_df[x_value].dropna().unique().tolist()
    if xtitle is None:
        xtitle = x_value

    # feature_df = make_single_feature_df(
    #     data_df, group=x_value, feature=y_value, replicates=replicate_col_name
    # )
    pairs = getpairs(data_df, x_value, order)
    print(pairs)

    # Remove the n=1 replicate in the passage group code
    if x_value == "AllGroups":
        order = get_all_group_order()
        # feature_df = feature_df[feature_df[group] != "P22-24"]

    group_avg_df = average_groups_by_plate(
        data_df, x_value=x_value, y_value=y_value, replicates=replicate_col_name
    )

    group_avg_df_shapiro = apply_shapiro_wilk_test_to_df(
        group_avg_df, feature_meas=y_value
    )

    group_avg_df_pivot = average_groups_pivot(
        group_avg_df=group_avg_df_shapiro,
        x_value=x_value,
        y_value=y_value,
        replicate_col_name=replicate_col_name,
    )

    sns.set_theme(style="ticks")
    plt.figure(figsize=figsize)  # , dpi=dpi)
    sns.set_context("talk", font_scale=0.5)

    sns.swarmplot(
        data=data_df,
        x=x_value,
        y=y_value,
        hue=replicate_col_name,
        order=order,
        palette=pallete,
        dodge=True,
    )

    ax = sns.swarmplot(
        data=group_avg_df,
        x=x_value,
        y=y_value,
        hue=replicate_col_name,
        order=order,
        palette=pallete,
        size=10,
        edgecolor="k",
        linewidth=1,
        dodge=False,
    )

    sns.boxplot(
        data=group_avg_df,
        x=x_value,
        y=y_value,
        order=order,
        showmeans=True,
        meanline=True,
        meanprops={"color": "dimgray", "ls": "-", "lw": 2.5},
        medianprops={"visible": False},
        whiskerprops={"visible": False},
        zorder=2,
        showfliers=False,
        showbox=False,
        showcaps=False,
        ax=ax,
    )

    if ax.legend_ is not None:
        ax = annotate_legend_with_shapiro(ax, group_avg_df_shapiro, replicate_col_name)
        # ax.legend_.remove()

    sns.despine()
    plt.tight_layout()
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.ylim(ylim)
    if annotate and test is not None:
        try:
            print(test)
            ax = annotate_pairs_with_calculated_pvalues(
                ax,
                group_avg_df_shapiro,
                group_avg_df_pivot,
                x_value,
                y_value,
                replicate_col_name=replicate_col_name,
                test_name=test,
                order=order,
                plot="swarmplot",
            )
        except Exception as e:
            print(f"Error annotating with statistical test: {e}")
    # from statannotations.Annotator import Annotator
    # annotator = Annotator(ax, pairs, data=group_avg_df_pivot, order=order)
    # annotator.configure(
    #     test="Kruskal",
    #     text_format="star",
    #     loc="inside",
    #     hide_non_significant=True,
    #     color="black",
    #     verbose=2,
    # )
    # annotator.apply_and_annotate()

    plt.savefig(xtitle + "_" + y_value + "_superswarmplot.png", dpi=dpi)
    plt.show()
