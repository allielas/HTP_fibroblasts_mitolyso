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
from pathlib import Path
import scipy
import seaborn as sns
import scikit_posthocs as sp
from plate_information import *


"""
See this awesome course note here: https://biapol.github.io/PoL-BioImage-Analysis-TS-Early-Career-Track/day3a_plotting/03_Statistic_Annotations_in_Seaborn_Bonus.html

NOTE for custom annotations in statannotations:
    Use results from a previous stat test e.g.
    stat_results_GMF = [mannwhitneyu(Gentoo_values_female['bill_length_mm'], Gentoo_values_male['bill_length_mm'], alternative="two-sided"),]
    pvalues = [result.pvalue for result in stat_results_GMF]
    formatted_pvalues = [f"p={p:.2e}" for p in pvalues]
    annotator.set_custom_annotations(formatted_pvalues)
"""


def create_superplot_fortwo(data, x_value, y_value, plates, save_name):
    sns.set_style("whitegrid")

    # Calculate the mean for each group
    group_averages = data.groupby([x_value, plates], as_index=False).agg(
        {y_value: "mean"}
    )
    group_ave_pivot = group_averages.pivot_table(
        columns=x_value, values=y_value, index=plates
    )

    # Perform paired t-test
    statistic, pvalue = scipy.stats.ttest_rel(
        group_ave_pivot.iloc[:, 0], group_ave_pivot.iloc[:, 1]
    )
    pvalue_str = str(float(round(pvalue, 3)))
    if pvalue < 0.05:
        pvalue_str = pvalue_str + "*"

    # Create the superplot
    sns.swarmplot(x=x_value, y=y_value, hue=plates, data=data)  # plot the data
    ax = sns.swarmplot(
        x=x_value,
        y=y_value,
        hue=plates,
        size=15,
        edgecolor="k",
        linewidth=2,
        data=group_averages,
    )  # plot the averages from each plate
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
    Actually consider the plates
    Consider mean value of each plate per treatment
    directly taken from S5 of original publication

    Args:
        df (DataFrame): _description_
        ax (MatPlotLib Axis): _description_

    Returns:
        MatPlotLib Axis): _description_
    """
    from scipy.stats import ttest_rel, ttest_ind
    import matplotlib.ticker

    PlateAverages = df.groupby(["Treatment", "Plate"], as_index=False).agg(
        {"Speed": "mean"}
    )
    PlateAvePivot = PlateAverages.pivot_table(
        columns="Treatment", values="Speed", index="Plate"
    )
    # Calculate 'appropriate' p-value considering n=3
    good_pvalue = ttest_rel(PlateAvePivot["Control"], PlateAvePivot["Drug"]).pvalue

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

    sns.swarmplot(  # plot all data points by plate as swarmplot
        data=df,
        x="Treatment",
        y="Speed",
        hue="Plate",
        size=4,
        zorder=0,
        palette=alpha_palette,
        linewidth=1,
        legend=False,
        ax=ax,
    )

    sns.pointplot(  # one dot representing mean of each plate separated by color and marker type
        data=df,
        x="Treatment",
        y="Speed",
        hue="Plate",
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

    sns.pointplot(  # standard error bars of mean values of each plate
        data=PlateAverages,
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
    plate,
    feature_meas,
    plate_col_name="Plate_Number",
    debug=False,
):
    """Function to apply the shapiro wilk test to a dataframe aggregated by plate for a single feature

    Args:
        group_avg_df (DataFrame): the aggregated dataframe
        plate (string, int): string or int representation of plate number
        feature_meas (string): _description_
        plate_col_name (str, optional): the name of the plate column. Defaults to "Plate_Number".
        debug (bool, optional): print out p values. Defaults to False.

    Returns:
        float: p_value from test on that plate
    """
    from scipy.stats import shapiro

    rep_df = group_avg_df[group_avg_df[plate_col_name] == plate]
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
    group_avg_df, feature_meas, plate_col_name="Plate_Number", alpha=0.05
):
    """Function to applies the shapiro wilk test row-by-row onto an aggregated dataframe by plate

    Args:
        group_avg_df (DataFrame): the aggregated dataframe
        feature_meas (string): _description_
        plate_col_name (str, optional): the name of the plate column. Defaults to "Plate_Number".
        alpha (float): the p value threshold. Defaults to p=0.05

    Returns:
        DataFrame: The aggregated dataframe with a "Shaprio_pvalue" column and a boolean "Shapiro_normality" column
    """
    # apply the shapiro-wilk test to a dataframe
    group_avg_df = group_avg_df.dropna()
    group_avg_df["Shapiro_pvalue"] = group_avg_df.apply(
        lambda row: shapiro_pvalue(
            group_avg_df, plate=row[plate_col_name], feature_meas=feature_meas
        ),
        axis=1,
    )
    # reject null hypothesis if p < 0.05 - i.e. significant chance that the data is not normally distributed
    group_avg_df["Shapiro_normality"] = group_avg_df.apply(
        lambda row: row["Shapiro_pvalue"] > alpha, axis=1
    )
    return group_avg_df


def average_groups_pivot(group_avg_df, x_value, y_value, plate_col_name):
    """Make a pivot table from the averaged dataframe

    Args:
        group_avg_df (DataFrame): your dataframe output from average_groups_by_plate()
        x_value (string): the grouping variable (x value)
        y_value (string): the quantitavie feature to measure (y value)
        plate_col_name (string): the variable representing experimental plates for grouping

    Returns:
        DataFrame: a pivot table
    """
    group_avg_df_pivot = group_avg_df.pivot_table(
        columns=x_value, values=y_value, index=plate_col_name
    )
    return group_avg_df_pivot


def pvalues_anova_and_tukeyhsd_posthoc(
    data_df,
    pivot_df,
    x_value,
    y_value,
    plate_number_col="Plate_Number",
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
        grouping_variable (str): Column name for the plate number.
        order (list, optional): Order of groups for plotting. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame with Tukey's HSD results.
    """
    from scipy.stats import f_oneway
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    groups = []  # Convert pivot table to list of groups
    # display(pivot_df)
    for col in pivot_df:
        if col == x_value or col == plate_number_col:
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


def pvalues_anova_with_games_howell(
    data_df,
    pivot_df,
    x_value,
    y_value,
    plate_number_col="Plate_Number",
    desired_pairs=None,
    order=None,
):
    """Perform Games-Howell post-hoc test on the data. Same function as tukey HSD but with unequal variance / sample size
    Args:
        data_df (pd.DataFrame): DataFrame table containing the data.
        pivot_df (pd.DataFrame): DataFrame pivot table containing the means.
        x_value (str): Column name for the independent variable.
        y_value (str): Column name for the dependent variable.
        grouping_variable (str): Column name for the plate number.
        order (list, optional): Order of groups for plotting. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame with Tukey's HSD results.
    """
    from scipy.stats import f_oneway
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    groups = []  # Convert pivot table to list of groups
    # display(pivot_df)
    for col in pivot_df:
        if col == x_value or col == plate_number_col:
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
            endog=data_df[y_value],
            groups=data_df[x_value],
            alpha=0.05,
            use_var="unequal",
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
        print("ANOVA test is not significant, skipping Games-Howell post-hoc test.")
        return ([], [])


def pvalues_anova_with_games_howell_pingouin(
    data_df,
    pivot_df,
    x_value,
    y_value,
    plate_number_col="Plate_Number",
    desired_pairs=None,
    order=None,
    display=False,
):
    """Perform Games-Howell post-hoc test on the data. Same function as tukey HSD but with unequal variance / sample size with the Pingouin package.
    See https://pingouin-stats.org/build/html/guidelines.html for a flowchart
    Args:
        data_df (pd.DataFrame): DataFrame table containing the data.
        pivot_df (pd.DataFrame): DataFrame pivot table containing the means.
        x_value (str): Column name for the independent variable.
        y_value (str): Column name for the dependent variable.
        grouping_variable (str): Column name for the plate number.
        order (list, optional): Order of groups for plotting. Defaults to None.

    Returns:
     (pairs,pvalues) (tuple): pairs and corresponding p values
    """
    import pingouin as pg

    df = data_df.copy()

    # 1. This is a between subject design, so the first step is to test for equality of variances
    homo_df = pg.homoscedasticity(data=df.dropna(), dv=y_value, group=x_value)
    pg.print_table(homo_df)
    # 2. If the groups have equal variances, we can use a regular one-way ANOVA
    if homo_df["equal_var"][0]:
        anova_res = pg.anova(data=df, dv=y_value, between=x_value)
        # else use a welch's anova
    else:
        anova_res = pg.welch_anova(data=df, dv=y_value, between=x_value)
    pg.print_table(anova_res)

    p_value_anova = anova_res["p-unc"][0]
    # do games-howell
    if p_value_anova < 0.05:
        # print(df)
        games_result = pg.pairwise_gameshowell(
            data=df, dv=y_value, between=x_value
        )  # .round(3)
        # this is a dataframe; get the pairs and the pvalue cols
        games_result_pairs = games_result[["A", "B"]].itertuples(index=False, name=None)
        pairs = list(games_result_pairs)
        p_values = games_result["pval"].tolist()
        if display:
            pg.print_table(games_result)
        # display(tukey_result_df)
        return (pairs, p_values)
    else:
        print("ANOVA test is not significant, skipping Games-Howell post-hoc test.")
        return ([], [])


def pvalues_anova_with_tukey_pingouin(
    data_df,
    pivot_df,
    x_value,
    y_value,
    plate_number_col="Plate_Number",
    desired_pairs=None,
    order=None,
    display=False,
):
    """Perform Tukey's post-hoc test on the data. Uses Tukey-Kramer method with sample size with the Pingouin package.
    See https://pingouin-stats.org/build/html/guidelines.html for a flowchart
    Args:
        data_df (pd.DataFrame): DataFrame table containing the data.
        pivot_df (pd.DataFrame): DataFrame pivot table containing the means.
        x_value (str): Column name for the independent variable.
        y_value (str): Column name for the dependent variable.
        grouping_variable (str): Column name for the plate number.
        order (list, optional): Order of groups for plotting. Defaults to None.

    Returns:
     (pairs,pvalues) (tuple): pairs and corresponding p values
    """
    import pingouin as pg

    df = data_df.copy()

    # do the anova
    anova_res = pg.anova(data=df, dv=y_value, between=x_value)
    pg.print_table(anova_res)
    p_value_anova = anova_res["p-unc"][0]
    # do the tukey
    if p_value_anova < 0.05:
        # print(df)
        tukey_result = pg.pairwise_tukey(
            data=df, dv=y_value, between=x_value
        )  # .round(3)
        # this is a dataframe; get the pairs and the pvalue cols
        result_pairs = tukey_result[["A", "B"]].itertuples(index=False, name=None)
        pairs = list(result_pairs)
        p_values = tukey_result["p-tukey"].tolist()
        if display:
            pg.print_table(tukey_result)
        # display(tukey_result_df)
        return (pairs, p_values)
    else:
        print("ANOVA test is not significant, skipping Tukey's post-hoc test.")
        return ([], [])


def pvalues_anova_with_pairwise_tests_pingouin(
    data_df,
    x_value,
    y_value,
    plate_number_col="Plate_Number",
    pval_correction="bonf",
    parametric=True,
    desired_pairs=None,
    order=None,
    display=False,
):
    """Perform Tukey's post-hoc test on the data. Uses Tukey-Kramer method with sample size with the Pingouin package.
    See https://pingouin-stats.org/build/html/guidelines.html for a flowchart
    Args:
        data_df (pd.DataFrame): DataFrame table containing the data.
        x_value (str): Column name for the independent variable.
        y_value (str): Column name for the dependent variable.
        grouping_variable (str): Column name for the plate number.
        pval_correction (str, optional): Method for p-value correction. Defaults to "bonf".
        parametric (bool, optional): Use ttest if True, Mann-Whitney U or Wilcoxon Signed-Rank (paired) for nonparametric. Defaults to True.
        order (list, optional): Order of groups for plotting. Defaults to None.

    Returns:
     (pairs,pvalues) (tuple): pairs and corresponding p values
    """
    import pingouin as pg

    df = data_df.copy()

    # do the anova
    if parametric:
        homo_df = pg.homoscedasticity(data=df.dropna(), dv=y_value, group=x_value)
        pg.print_table(homo_df)
        # If the groups have equal variances, we can use a regular one-way ANOVA
        if homo_df["equal_var"][0]:
            anova_res = pg.anova(data=df, dv=y_value, between=x_value)
            # else use a welch's anova
        else:
            anova_res = pg.welch_anova(data=df, dv=y_value, between=x_value)
    else:
        anova_res = pg.kruskal(data=df, dv=y_value, between=x_value)
    pg.print_table(anova_res)
    # print(anova_res["pval"])
    anova_pvalue = anova_res["p-unc"][0]

    # do the tukey
    if anova_pvalue < 0.05:
        # print(df)
        test_result = pg.pairwise_tests(
            data=df,
            dv=y_value,
            between=x_value,
            parametric=parametric,
            padjust=pval_correction,
        )  # .round(3)
        # this is a dataframe; get the pairs and the pvalue cols
        result_pairs = test_result[["A", "B"]].itertuples(index=False, name=None)
        pairs = list(result_pairs)
        p_values = test_result["p-corr"].tolist()
        if display:
            pg.print_table(test_result)
        # display(tukey_result_df)
        return (pairs, p_values)
    else:
        print("ANOVA test is not significant, skipping post-hoc test.")
        return ([], [])


def anova_with_tukey_posthoc(
    data_df,
    x_value,
    y_value,
    plate_number_col="Plate_Number",
    desired_pairs=None,
    order=None,
    display_results=False,
):
    """Tukey HSD test using Scikit_posthocs

    Args:
        data_df (_type_): _description_
        x_value (_type_): _description_
        y_value (_type_): _description_
        plate_number_col (str, optional): _description_. Defaults to "Plate_Number".
        desired_pairs (_type_, optional): _description_. Defaults to None.
        order (_type_, optional): _description_. Defaults to None.
        display_results (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
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


def anova_with_corr_ttest_posthoc(
    data_df,
    x_value,
    y_value,
    plate_number_col="Plate_Number",
    desired_pairs=None,
    order=None,
    display_results=False,
    p_corr="bonferroni",
):
    """Welch's ttst with fdr corrections using Scikit_posthocs

    Args:
        data_df (_type_): _description_
        x_value (_type_): _description_
        y_value (_type_): _description_
        plate_number_col (str, optional): _description_. Defaults to "Plate_Number".
        desired_pairs (_type_, optional): _description_. Defaults to None.
        order (_type_, optional): _description_. Defaults to None.
        display_results (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    from scikit_posthocs import posthoc_ttest

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
        tt_df = posthoc_ttest(
            data_df, val_col=y_value, group_col=x_value, p_adjust=p_corr
        )

        # melt the dunn_df to long format
        remove = np.tril(np.ones(tt_df.shape), k=0).astype("bool")
        tt_df[remove] = np.nan
        molten_df = tt_df.melt(ignore_index=False).reset_index().dropna()

        if display_results:
            # display(tukey_df)
            print(molten_df)

        tt_pairs = molten_df[["index", "variable"]].itertuples(index=False, name=None)
        pairs = list(tt_pairs)
        p_values = molten_df["value"].tolist()
        return (pairs, p_values)

    else:
        print(
            f"Oneway ANOVA is not significant, skipping Welchs's ttest with {p_corr} correction."
        )
        return ([], [])


def anova_with_tahmane_posthoc(
    data_df,
    x_value,
    y_value,
    plate_number_col="Plate_Number",
    desired_pairs=None,
    order=None,
    display_results=False,
):
    """Perform Tamhane's T2 post-hoc test on the data. Alnother similar function like tukey HSD for normal data but supports unequal variance / sample size with the Pingouin package.
    See https://scikit-posthocs.readthedocs.io/en/latest/generated/scikit_posthocs.posthoc_tamhane.html#scikit_posthocs.posthoc_tamhane for a flowchart
    Args:
        data_df (pd.DataFrame): DataFrame table containing the data.
        x_value (str): Column name for the independent variable.
        y_value (str): Column name for the dependent variable.
        plate_number_col (str): Column name for the plate number.
        order (list, optional): Order of groups for plotting. Defaults to None.
        display_results (bool, optional): Defaults to false

    Returns:
     (pairs,pvalues) (tuple): pairs and corresponding p values
    """
    from scikit_posthocs import posthoc_tamhane
    import pingouin as pg

    df = data_df.copy()
    # anova_res = pg.welch_anova(data=df, dv=y_value, between=x_value)
    anova_res = pg.anova(data=df, dv=y_value, between=x_value)
    pg.print_table(anova_res)
    # print(anova_res["pval"])
    anova_pvalue = anova_res["p-unc"][0]
    if anova_pvalue < 0.05:
        # posthoc dunn test
        tam_df = posthoc_tamhane(
            data_df, val_col=y_value, group_col=x_value, welch=True
        )
        # melt the dunn_df to long format
        remove = np.tril(np.ones(tam_df.shape), k=0).astype("bool")
        tam_df[remove] = np.nan
        molten_df = tam_df.melt(ignore_index=False).reset_index().dropna()

        if display_results:
            # display(dunn_df)
            print(molten_df)
        tam_pairs = molten_df[["index", "variable"]].itertuples(index=False, name=None)
        pairs = list(tam_pairs)
        p_values = molten_df["value"].tolist()
        return (pairs, p_values)

    else:
        print("ANOVA test is not significant, skipping Tahmane's post-hoc test.")
        return ([], [])


def kruskal_with_dunn_posthoc(
    data_df,
    x_value,
    y_value,
    plate_number_col="Plate_Number",
    desired_pairs=None,
    order=None,
    p_correction="fdr_by",
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


def kruskal_with_drubin_posthoc(
    data_df,
    x_value,
    y_value,
    plate_number_col="Plate_Number",
    desired_pairs=None,
    order=None,
    display_results=False,
    p_correction=None,
):
    """Drubin's nonparametric test for unbalenced block design. Uses a grouped_df, not a pivot table as it needs to group the blocks

    Args:
        data_df (_type_): _description_
        x_value (_type_): _description_
        y_value (_type_): _description_
        plate_number_col (str, optional): _description_. Defaults to "Plate_Number".
        desired_pairs (_type_, optional): _description_. Defaults to None.
        order (_type_, optional): _description_. Defaults to None.
        display_results (bool, optional): _description_. Defaults to False.

    Returns:
        (pairs,pvalues) (tuple): _description_
    """
    from scikit_posthocs import posthoc_durbin

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
        drubin_df = posthoc_durbin(
            data_df,
            y_col=y_value,
            group_col=x_value,
            block_col=plate_number_col,
            block_id_col=plate_number_col,
            melted=True,
        )
        # melt the dunn_df to long format
        remove = np.tril(np.ones(drubin_df.shape), k=0).astype("bool")
        drubin_df[remove] = np.nan
        molten_df = drubin_df.melt(ignore_index=False).reset_index().dropna()

        if display_results:
            # display(dunn_df)
            print(molten_df)
        drubin_pairs = molten_df[["index", "variable"]].itertuples(
            index=False, name=None
        )
        pairs = list(drubin_pairs)
        p_values = molten_df["value"].tolist()
        return (pairs, p_values)

    else:
        print(
            "Kruskal-Wallis test is not significant, skipping Drubin's post-hoc test."
        )
        return ([], [])


def kruskal_with_conover_posthoc(
    data_df,
    x_value,
    y_value,
    plate_number_col="Plate_Number",
    desired_pairs=None,
    order=None,
    display_results=False,
    p_correction=None,
):
    """Conover's nonparametric post-hoc test for multiple comparisons. Alternative posthoc to Dunn's test with more statistial power

    Args:
        data_df (_type_): _description_
        x_value (_type_): _description_
        y_value (_type_): _description_
        plate_number_col (str, optional): _description_. Defaults to "Plate_Number".
        desired_pairs (_type_, optional): _description_. Defaults to None.
        order (_type_, optional): _description_. Defaults to None.
        display_results (bool, optional): _description_. Defaults to False.

    Returns:
        (pairs,pvalues) (tuple): _description_
    """
    from scikit_posthocs import posthoc_conover

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
        con_df = posthoc_conover(
            data_df, val_col=y_value, group_col=x_value, p_adjust=p_correction
        )

        # melt the dunn_df to long format
        remove = np.tril(np.ones(con_df.shape), k=0).astype("bool")
        con_df[remove] = np.nan
        molten_df = con_df.melt(ignore_index=False).reset_index().dropna()
        molten_df_sorted = molten_df.sort_values(
            by=["index", "variable"], key=lambda x: x.map(allgroups_sort_key)
        ).reset_index(drop=True)
        if display_results:
            print(molten_df_sorted)
        con_pairs = molten_df_sorted[["index", "variable"]].itertuples(
            index=False, name=None
        )
        pairs = list(con_pairs)
        p_values = molten_df_sorted["value"].tolist()
        return (pairs, p_values)

    else:
        print(
            "Kruskal-Wallis test is not significant, skipping Conover's post-hoc test."
        )
        return ([], [])


def kruskal_with_nemenyi_posthoc(
    data_df,
    x_value,
    y_value,
    plate_number_col="Plate_Number",
    desired_pairs=None,
    order=None,
    display_results=False,
    p_correction=None,
):
    """Nemenyi's nonparametric post-hoc test for multiple comparisons after Kruskal or Friedman's test. Alternative posthoc to Dunn's test that can also be used for repeated measurees

    Args:
        data_df (_type_): _description_
        x_value (_type_): _description_
        y_value (_type_): _description_
        plate_number_col (str, optional): _description_. Defaults to "Plate_Number".
        desired_pairs (_type_, optional): _description_. Defaults to None.
        order (_type_, optional): _description_. Defaults to None.
        display_results (bool, optional): _description_. Defaults to False.
    Returns:
        (pairs,pvalues) (tuple): _description_
    """
    from scikit_posthocs import posthoc_nemenyi

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
        nem_df = posthoc_nemenyi(data_df, val_col=y_value, group_col=x_value)

        # melt the dunn_df to long format
        remove = np.tril(np.ones(nem_df.shape), k=0).astype("bool")
        nem_df[remove] = np.nan
        molten_df = nem_df.melt(ignore_index=False).reset_index().dropna()

        if display_results:
            # display(dunn_df)
            print(molten_df)
        nem_pairs = molten_df[["index", "variable"]].itertuples(index=False, name=None)
        pairs = list(nem_pairs)
        p_values = molten_df["value"].tolist()
        return (pairs, p_values)

    else:
        print(
            "Kruskal-Wallis test is not significant, skipping Nemenyi's post-hoc test."
        )
        return ([], [])


def annotate_with_anova_tukey(
    ax,
    pairs,
    data,
    x_value,
    y_value,
    plate_col_name="Plate_Name",
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
    )  # x=x_value, y=y_value, hue=plate_col_name,
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


def allgroups_sort_key(value):
    """Custom sort key function for 'AllGroups' column."""
    import re

    match = re.match(r"P(\d+)", value)
    if match:
        first_number = match.group(1)
        if first_number.isdigit():
            return int(first_number)
        else:
            return 99999
    return 99999


def annotate_pairs_with_calculated_pvalues(
    ax,
    data,
    pivot_data,
    x_value,
    y_value,
    plate_col_name="Plate_Name",
    test_name="tukey",
    pairs=None,
    order=None,
    plot="violinplot",
    show_test_name=False,
    p_correction="fdr_bh",
):
    """Add statistical annotations to a plot using a multiple comparisons test in the statsmodels, scipy, or scikit-posthocs modules.
    see https://statannotations.readthedocs.io/en/latest/custom-test.html for more examples
    Also see https://www.graphpad.com/guides/prism/latest/statistics/stat_summary_of_multiple_comparison.htm for a list of posthoc tests and when to use them

    Args:
        ax (Matplotlib Axes Object): the axis of the graph to annoate
        data (DataFrame): dataframe from a grouped feature df containing the groups aggregated by plate to analyze in "tidy" format
        pivot_data (DataFrame): the grouped feature df in matrix format
        x_value (str): independent variable on x axis
        y_value (str): dependent variable on y axis
        plate_col_name (str, optional): the col containing the experimental plate. Defaults to "Plate_Name".
        test_name (str, optional): the statistical test to perform. Accepts values of "tukey", "anova", or "tukey_v2", for ANOVA with Tukey's HSD, "tukey_v3" for ANOVA with Tukey HSD with Tukey-Kramer correction, "games-howell" or "games" for ANOVA with Games-Howell posthoc, "rmanova" for repeated-measures ANOVA using Welch's ttest with the specified p-value correction, "kruskal" or "dunn" for classic nonparametric multiple comparisons with Dunn's postc, "conover" for kruskal with Conover's posthoc, "nemenyi" for kruskal (or friedman) with Nemenyi's posthoc for repeated measures, "pairwise_ttest" for corrected ttests, "pairwise_mwu" for nonparametic multiple comparisons. Defaults to "tukey".
        pairs (list of str, optional): The pairs of x_value for the comparisons. Defaults to None, is automatically calculated otherwise based on the getpairs() function.
        order (listlike, optional): _description_. Defaults to None.
        plot (str, optional): the type of plot to annotate. Defaults to "violinplot".
        p_correction (str, optional): the p-value correction to use if applicable. Deaults to Benjamini/Hochberg "fdr_bh" (non-negative) method ; graphpad reccomneds as its less hemmoraging to your power. Also accepts "holm", "sidak", "bonferroni", "holm-sidak" and Benjamini/Yekutieli "fdr-by" for negative values. See https://scikit-posthocs.readthedocs.io/en/latest/generated/scikit_posthocs.posthoc_mannwhitney.html for other options

    Returns:
        _type_: _description_
    """
    from statannotations.Annotator import Annotator

    if pairs is None:
        pairs = getpairs(data, x_value, order=order)

    # nonparametric tests
    if test_name in ["kruskal", "dunn", "kruskal-wallis"]:
        used_pairs, p_values = kruskal_with_dunn_posthoc(
            data,
            x_value=x_value,
            y_value=y_value,
            order=order,
            desired_pairs=pairs,
            p_correction=p_correction,
            display_results=True,
        )
    elif test_name in ["drubin", "drubin_posthoc"]:
        used_pairs, p_values = kruskal_with_drubin_posthoc(
            data,
            x_value=x_value,
            y_value=y_value,
            order=order,
            desired_pairs=pairs,
            p_correction=p_correction,
            display_results=True,
        )
    elif test_name in ["conover", "con", "kruskal-conover"]:
        used_pairs, p_values = kruskal_with_conover_posthoc(
            data,
            x_value=x_value,
            y_value=y_value,
            order=order,
            desired_pairs=pairs,
            p_correction=p_correction,
            display_results=True,
        )
    elif test_name in ["nemenyi", "kruskal-nemenyi"]:
        used_pairs, p_values = kruskal_with_nemenyi_posthoc(
            data,
            x_value=x_value,
            y_value=y_value,
            order=order,
            desired_pairs=pairs,
            display_results=True,
            p_correction=p_correction,
        )
    # parametric tests
    elif test_name in ["anova", "tukey", "tukeyhsd"]:
        # perform anova and tukey's post-hoc test
        used_pairs, p_values = pvalues_anova_and_tukeyhsd_posthoc(
            data, pivot_data, x_value, y_value, order=order, desired_pairs=pairs
        )
    elif test_name in ["games", "games-howell"]:
        used_pairs, p_values = pvalues_anova_with_games_howell_pingouin(
            data,
            pivot_data,
            x_value,
            y_value,
            order=order,
            desired_pairs=pairs,
            display=True,
        )
    elif test_name in ["tahmane", "tahmane-t2"]:
        used_pairs, p_values = anova_with_tahmane_posthoc(
            data,
            x_value,
            y_value,
            order=order,
            desired_pairs=pairs,
            display_results=True,
        )
    elif test_name in ["tukey_v2", "tukey_posthocs"]:
        used_pairs, p_values = anova_with_tukey_posthoc(
            data,
            x_value=x_value,
            y_value=y_value,
            plate_number_col=plate_col_name,
            order=order,
            desired_pairs=pairs,
            display_results=True,
        )
    elif test_name in ["tukey_v3", "tukey_pingouin"]:
        used_pairs, p_values = pvalues_anova_with_tukey_pingouin(
            data,
            pivot_data,
            x_value=x_value,
            y_value=y_value,
            plate_number_col=plate_col_name,
            order=order,
            desired_pairs=pairs,
            display=True,
        )
    elif test_name in ["pairwise_ttest" or "multiple_ttest"]:
        used_pairs, p_values = pvalues_anova_with_pairwise_tests_pingouin(
            data,
            x_value=x_value,
            y_value=y_value,
            order=order,
            desired_pairs=pairs,
            pval_correction=p_correction,
            parametric=True,
            display=True,
        )
    elif test_name in [
        "pairwise_mwu"
        or "multiple_mwu"
        or "pairwise_mannwhitney"
        or "multiple_mannwhitney"
        or "pairwise_wilcoxon"
        or "multiple_wilcoxon"
    ]:
        used_pairs, p_values = pvalues_anova_with_pairwise_tests_pingouin(
            data,
            x_value=x_value,
            y_value=y_value,
            order=order,
            desired_pairs=pairs,
            pval_correction=p_correction,
            parametric=False,
            display=True,
        )
    elif test_name in ["ttest_posthoc", "welch's_posthoc", "tt_posthoc"]:
        used_pairs, p_values = anova_with_corr_ttest_posthoc(
            data,
            x_value=x_value,
            y_value=y_value,
            order=order,
            desired_pairs=pairs,
            p_corr=p_correction,
            display_results=True,
        )
    else:
        raise ValueError(
            f"Test name '{test_name}' is invalid. Use 'tukey', 'anova', 'kruskal', 'games-howell', 'drubin', 'tukey_v2', or 'dunn'."
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
        # see https://raw.githubusercontent.com/trevismd/statannotations/3f020ae631ca88a091b6ee3e9a9fd32158920879/usage/example_tuning_y_offsets_w_arguments.png
        annotator.configure(
            text_format="full",
            test_short_name=test_name,
            pvalue_format_string="{:.3f}",
            fontsize="small",
            # pvalue_format = [[1e-5, "1e-5"], [1e-4, "1e-4"], [1e-3, "0.001"], [1e-2, "0.01"], [5e-2, "0.05"]],
            loc="outside",
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
    plate_col_name="Plate_Name",
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


def get_hard_code_plate_colours(
    df, plates=[1, 2, 3, 4, 5, 6, 7], plate_col_name="Plate_Number"
):
    """
    Returns a dictionary mapping each unique plate number to a hard-coded color.
    This ensures color consistency for each plate in seaborn/matplotlib plots.
    """
    import matplotlib
    import seaborn as sns

    # sns.color_palette("pastel").as_hex()
    hard_palette_pastel = [
        "#a1c9f4",
        "#ffb482",
        "#8de5a1",
        "#ff9f9b",
        "#d0bbff",
        "#debb9b",
        "#fab0e4",
    ]
    unique_reps = sorted(df[plate_col_name].dropna().unique())
    # If more plates than colors, repeat palette or use seaborn color_palette
    if len(unique_reps) > len(hard_palette_pastel):
        hard_palette_pastel = sns.color_palette("tab20", len(unique_reps)).as_hex()
        plates = unique_reps
    colour_dict = {rep: hard_palette_pastel[i] for i, rep in enumerate(plates)}
    return colour_dict


def annotate_legend_with_shapiro(
    ax,
    group_avg_df,
    group_col_name,
    shapiro_col_name="Shapiro_normality",
    palette="pastel",
    title="Plate",
):
    """add an annotation to the legend of an axis if there is normality via shapiro test

    Args:
        ax (_type_): _description_
        group_avg_df (_type_): _description_
        plate_col_name (_type_): _description_
    """
    import matplotlib.lines as mlines

    unique_plates = group_avg_df[group_col_name].unique()
    unique_plates = sorted(unique_plates)
    L = plt.legend()
    custom_labels = []
    for rep in unique_plates:
        label = str(rep)
        shapiro_val = group_avg_df[group_avg_df[group_col_name] == rep][
            shapiro_col_name
        ].iloc[0]
        if shapiro_val:
            label += " (normal)"
        custom_labels.append(label)

    # Create custom legend handles (using the same colors as swarmplot)
    palette = get_hard_code_plate_colours(
        group_avg_df
    )  # sns.color_palette(palette, n_colors=len(unique_plates))
    group_avg_df["Plate_Number"] = pd.Categorical(
        group_avg_df["Plate_Number"], categories=unique_plates
    )
    print(palette[1])
    handles = [
        mlines.Line2D(
            [],
            [],
            color=palette[rep],
            marker="o",
            linestyle="None",
            markersize=12,
            markeredgecolor="black",
            label=custom_labels[i],
        )
        for i, rep in enumerate(unique_plates)
    ]
    ax.legend_.set_title(title)
    ax.legend(handles=handles, title=title, loc="best")
    return ax


def annotate_legend_platesonly(
    ax,
    group_avg_df,
    group_col_name,
    palette="pastel",
    title="Plate",
):
    """add an annotation to the legend of an axis to show color coded plates

    Args:
        ax (_type_): _description_
        group_avg_df (_type_): _description_
        plate_col_name (_type_): _description_
    """
    import matplotlib.lines as mlines

    unique_plates = group_avg_df[group_col_name].unique()
    unique_plates = sorted(unique_plates)
    L = plt.legend()
    custom_labels = []
    for rep in unique_plates:
        label = str(rep)
        custom_labels.append(label)

    # Create custom legend handles (using the same colors as swarmplot)
    palette = sns.color_palette(palette, n_colors=len(unique_plates))
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
        for i in range(len(unique_plates))
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
    plate_col_name,
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
        hue=plate_col_name,
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
            group_avg_df, x_value, y_value, plate_col_name
        )
        try:
            ax = annotate_pairs_with_calculated_pvalues(
                ax,
                group_avg_df,
                group_avg_pivot_table,
                x_value,
                y_value,
                plate_col_name=plate_col_name,
                test_name=test,
                order=order,
                plot="violinplot",
                show_test_name=show_test_on_plot,
            )
        except Exception as e:
            print(f"Error annotating with statistical test: {e}")
            # ax = annotate_with_anova_tukey(ax, pairs, group_avg_df_pivot, x_value, y_value, plate_col_name=plate_col_name, order=order, plot="violinplot")
        # elif test == "kruskal":
        #     ax = annotate_with_kruskal(
        #         ax,
        #         pairs,
        #         group_avg_pivot_table,
        #         x_value,
        #         y_value,
        #         order=order,
        #         plate_col_name=plate_col_name,
        #         plot="violinplot",
        #     )
        if shapiro:
            ax = annotate_legend_with_shapiro(ax, group_avg_df, plate_col_name)

    return ax


def superplot_for_area_threshold_comparisons(
    data_df_1,
    group_avg_df_1,
    data_df_2,
    group_avg_df_2,
    x_value="AllGroups",
    y_value="Cell_AreaShape_Area",
    plate_col_name="Plate_Number",
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
        plate_col_name (str, optional): _description_. Defaults to "Plate_Number".
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
        hist = sns.kdeplot(data_df_1, x=y_value, hue=plate_col_name, palette="pastel")
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
        plate_col_name,
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
        plate_col_name,
        order=order,
        annotate=annotate,
        test=test,
    )

    axes[0].set_title(title1)
    axes[1].set_title(title2)

    if legend:
        axes[0] = annotate_legend_with_shapiro(axes[0], group_avg_df_1, plate_col_name)
        axes[1] = annotate_legend_with_shapiro(axes[1], group_avg_df_2, plate_col_name)
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
            group_avg_df_1, x_value, y_value, plate_col_name
        )
        group_avg_df_2_pivot = average_groups_pivot(
            group_avg_df_2, x_value, y_value, plate_col_name
        )
        group_avg_df_1_pivot.to_csv(
            os.path.join(pivot_dir, f"area_pivot_{title1}.csv")
        )  # can plop this into graphpad and see what it tells me
        group_avg_df_2_pivot.to_csv(os.path.join(pivot_dir, f"area_pivot_{title2}.csv"))


def super_splitviolinplot_helper_singleplot(
    data_df,
    group_avg_df,
    ax,
    x_value,
    y_value,
    title,
    plate_col_name,
    pairs=None,
    order=None,
    annotate=False,
    test=None,
    shapiro=True,
    show_test_on_plot=False,
    pallete=None,
    p_correction="bonferroni",
):
    group_avg_df = group_avg_df.copy()
    if pairs is None:
        pairs = getpairs(data_df, x_value, order=order)

    if pallete is None:
        pallete = get_hard_code_plate_colours(group_avg_df)
    print(pairs)
    sns.violinplot(
        data=data_df,
        x=x_value,
        y=y_value,  # hue=x_value,
        # palette="Set2",
        split=True,  # using split violin plots - only one side, basically looks like a histogram
        inner="quart",
        color="gainsboro",
        dodge=False,
        # fill = True
        width=1,
        linewidth=1.5,
        order=order,
        ax=ax,
        cut=0.5,
        common_norm=True,
    )
    # add in the colour scheme
    unique_plates = group_avg_df[plate_col_name].unique()
    group_avg_df[plate_col_name] = pd.Categorical(
        group_avg_df[plate_col_name], categories=unique_plates
    )
    sns.swarmplot(
        data=group_avg_df,
        x=x_value,
        y=y_value,
        hue=plate_col_name,
        order=order,
        palette=pallete,
        size=10,
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
        meanprops={"color": "dimgray", "ls": "-", "lw": 4},
        medianprops={"visible": False},
        whiskerprops={"visible": False},
        zorder=2,
        showfliers=False,
        showbox=False,
        showcaps=False,
        ax=ax,
    )
    ax.set_title(title)

    # use pivot table to get the average values for each group
    if annotate and test is not None:
        group_avg_pivot_table = average_groups_pivot(
            group_avg_df, x_value, y_value, plate_col_name
        )
        try:
            ax = annotate_pairs_with_calculated_pvalues(
                ax,
                group_avg_df,
                group_avg_pivot_table,
                x_value,
                y_value,
                plate_col_name=plate_col_name,
                test_name=test,
                order=order,
                plot="violinplot",
                show_test_name=show_test_on_plot,
                p_correction=p_correction,
            )
        except Exception as e:
            print(f"Error annotating with statistical test: {e}")
            # ax = annotate_with_anova_tukey(ax, pairs, group_avg_df_pivot, x_value, y_value, plate_col_name=plate_col_name, order=order, plot="violinplot")
        # elif test == "kruskal":
        #     ax = annotate_with_kruskal(
        #         ax,
        #         pairs,
        #         group_avg_pivot_table,
        #         x_value,
        #         y_value,
        #         order=order,
        #         plate_col_name=plate_col_name,
        #         plot="violinplot",
        #     )
        if shapiro:
            ax = annotate_legend_with_shapiro(ax, group_avg_df, plate_col_name)

    return ax


def make_superswarmplot_with_annotation(
    data_df,
    x_value,
    y_value,
    plate_col_name="Plate_Number",
    pairs=None,
    order=None,
    annotate=False,
    test="tukey",
    ytitle=None,
    xtitle=None,
    ylim=None,
    pallete=None,
    figsize=(10, 9),
    context="talk",
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
    #     data_df, group=x_value, feature=y_value, plates=plate_col_name
    # )
    pairs = getpairs(data_df, x_value, order)
    print(pairs)

    # Remove the n=1 plate in the passage group code
    if x_value == "AllGroups":
        order = get_all_group_order()
        # feature_df = feature_df[feature_df[group] != "P22-24"]

    group_avg_df = average_groups_by_plate(
        data_df, x_value=x_value, y_value=y_value, plates=plate_col_name
    )

    group_avg_df_shapiro = apply_shapiro_wilk_test_to_df(
        group_avg_df, feature_meas=y_value
    )

    group_avg_df_pivot = average_groups_pivot(
        group_avg_df=group_avg_df_shapiro,
        x_value=x_value,
        y_value=y_value,
        plate_col_name=plate_col_name,
    )

    sns.set_theme(style="ticks")
    plt.figure(figsize=figsize)  # , dpi=dpi)
    sns.set_context(context, font_scale=0.8)
    if pallete is None:
        pallete = get_hard_code_plate_colours(data_df)

    sns.swarmplot(
        data=data_df,
        x=x_value,
        y=y_value,
        hue=plate_col_name,
        order=order,
        palette=pallete,
        dodge=False,
    )
    group_avg_df[plate_col_name]
    ax = sns.swarmplot(
        data=group_avg_df,
        x=x_value,
        y=y_value,
        hue=plate_col_name,
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
        meanprops={"color": "dimgray", "ls": "-", "lw": 4},
        medianprops={"visible": False},
        whiskerprops={"visible": False},
        zorder=2,
        showfliers=False,
        showbox=False,
        showcaps=False,
        ax=ax,
    )

    if ax.legend_ is not None:
        ax = annotate_legend_with_shapiro(ax, group_avg_df_shapiro, plate_col_name)
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
                plate_col_name=plate_col_name,
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


def ridge_label(x, color, label):
    ax = plt.gca()
    ax.text(
        -0.1,
        -0.2,
        label,
        fontweight="bold",
        color=color,
        ha="left",
        va="center",
        transform=ax.transAxes,
    )


def seaborn_ridgeplot(
    df,
    value_col,
    group_col,
    palette=None,
    bw_adjust=1,
    xlabel=None,
    xlim=(None, None),
    title=None,
    fill_alpha=1,
    linewidth=1.5,
    figsize=(20, 30),
    save=True,
    out_dir="",
    show_percentiles=True,
    truncate_outliers=True,
    norm=False,
):
    """
    Make a ridgeline (joyplot) using seaborn FacetGrid and kdeplot
    Args:
        df: DataFrame
        value_col: str, column with numeric values
        group_col: str, column with group/category
        palette: seaborn palette or list/dict of colors
        bw_adjust: float, KDE bandwidth adjust
        xlabel: str or None
        title: str or None
        fill_alpha: float, alpha for fill
        linewidth: float, line width for outline
        figsize: tuple, figure size
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

    unique_groups = df[group_col].unique()
    if palette is None:
        palette = sns.cubehelix_palette(len(unique_groups), rot=-0.25, light=0.7)
    else:
        palette = palette

    if truncate_outliers and xlim == (None, None):
        try:
            if norm:
                top_fence = df[value_col].mean() + 10 * df[value_col].std()
                bottom_fence = None  # np.percentile(data_df[y_value], 0.000001)
            else:
                top_fence = np.percentile(df[value_col], 99.9)
                bottom_fence = np.percentile(df[value_col], 0.01)
                # handle errors where the data is very skewed and the percentile is inf or nan
                if top_fence == 0 or np.isnan(top_fence) or np.isinf(top_fence):
                    top_fence = None
                if (
                    bottom_fence == 0
                    or np.isnan(bottom_fence)
                    or np.isinf(bottom_fence)
                ):
                    bottom_fence = None
            xlim = (bottom_fence, top_fence)
        except ValueError as e:
            print(e)
            top_fence = None
            bottom_fence = None
            xlim = (None, None)
        print(f"Truncating outliers at: {xlim}")
        xmin, xmax = xlim
        plot_df = df.copy()
        if xmin is not None:
            plot_df = plot_df[plot_df[value_col] >= xmin]
        if xmax is not None:
            plot_df = plot_df[plot_df[value_col] <= xmax]
    else:
        plot_df = df.copy()
    # Initialize the FacetGrid object
    g = sns.FacetGrid(
        plot_df,
        row=group_col,
        hue=group_col,
        aspect=15,
        height=0.5,
        palette=palette,
        xlim=xlim,
    )
    # Draw the densities in a few steps
    g.map(
        sns.kdeplot,
        value_col,
        bw_adjust=bw_adjust,
        clip_on=False,
        fill=True,
        alpha=fill_alpha,
        linewidth=linewidth,
    )
    g.map(sns.kdeplot, value_col, clip_on=False, color="w", lw=2, bw_adjust=bw_adjust)

    if show_percentiles:
        percentiles = [5, 12.5, 25, 50, 75, 87.5, 95]
        for ax in g.axes.flatten():
            # group label text (FacetGrid puts "group_col = <value>" in the title)
            title_text = ax.get_title()
            if " = " in title_text:
                group_val = title_text.split(" = ", 1)[1]
            else:
                group_val = title_text

            group_data = df[df[group_col] == group_val][value_col].dropna()
            if group_data.empty:
                continue

            # pick the most representative line on the axis (the KDE line)
            lines = ax.get_lines()
            if not lines:
                continue
            # choose the line with the largest x-range (robust against multiple lines)
            kde_line = max(lines, key=lambda l: np.ptp(l.get_xdata()))
            xs = kde_line.get_xdata()
            ys = kde_line.get_ydata()
            # line_colour = kde_line.get
            # compute median and interpolate its KDE height
            median = np.median(group_data)
            height = np.interp(median, xs, ys, left=0.0, right=0.0)

            # draw a solid thicker median line from y=0 up to the KDE height
            ax.vlines(
                median, 0, height, color="black", linewidth=3, linestyle=":", zorder=4
            )
            ax.set_xlim(xlim)
            # draw lighter dashed lines for the percentiles (optional)
            group_percentiles = np.percentile(group_data, percentiles)
            for p in group_percentiles:
                ax.vlines(p, 0, height, color="dimgray", ls=":", alpha=0.6, linewidth=2)
    # passing color=None to refline() uses the hue mapping
    g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

    g.map(ridge_label, value_col)

    # Set the subplots to overlap
    g.figure.subplots_adjust(hspace=-0.25)
    g.figure.set_size_inches(figsize)
    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[], ylabel="", xlim=xlim)
    g.despine(bottom=True, left=True)
    if xlabel:
        plt.xlabel(xlabel, fontweight="bold", fontsize=14)
    else:
        plt.xlabel(value_col, fontweight="bold", fontsize=14)
    if title:
        g.figure.suptitle(title, ha="right", fontsize=18, fontweight="bold")
    plt.xlim(xlim)
    plt.tight_layout()
    if save:
        plt.savefig(f"{Path(out_dir, f'{value_col}_{group_col}_joyplot')}.png")
    plt.show()


def single_feature_super_splitviolinplot(
    data_df,
    x_value="AllGroups",
    y_value="Cell_AreaShape_Area",
    plate_col_name="Plate_Number",
    out_dir=Path(""),
    xtitle=None,
    ytitle=None,
    order=None,
    legend=True,
    annotate=False,
    test=None,
    show_hist=False,
    remove_outliers=False,
    rm_outliers_method="mad",
    ylim=None,
    reps_to_exclude=[],
    shapiro=True,
    show=True,
    context="talk",
    figsize=(8, 6),
    truncate_outliers=False,
    norm=False,
    pallete=None,
    p_correction="bonferroni",
):
    """Make a superplot to do multiple comparisons for a feature between different conditions
    Args:
        data_df_1 (_type_): _description_
        group_avg_df_1 (_type_): _description_
        data_df_2 (_type_): _description_
        group_avg_df_2 (_type_): _description_
        x_value (str, optional): _description_. Defaults to "AllGroups".
        y_value (str, optional): _description_. Defaults to "Cell_AreaShape_Area".
        plate_col_name (str, optional): _description_. Defaults to "Plate_Number".
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
    if truncate_outliers:
        try:
            bottom_fence = None  # np.percentile(data_df[y_value], 0.000001)
            if norm:
                top_fence = data_df[y_value].mean() + 10 * data_df[y_value].std()
            else:
                top_fence = np.percentile(data_df[y_value], 99.9)
                # handle errors where the data is very skewed and the percentile is inf or nan
                if top_fence == 0 or np.isnan(top_fence) or np.isinf(top_fence):
                    top_fence = None
            axlim = (bottom_fence, top_fence)
        except ValueError as e:
            print(e)
            top_fence = None
            bottom_fence = None
            axlim = (None, None)
    else:
        axlim = (None, None)

    df_sorted = data_df.sort_values(
        by=[x_value], key=lambda x: x.map(passage_groups_sort_key)
    ).reset_index(drop=True)
    if show_hist:
        hist = sns.kdeplot(
            df_sorted, x=y_value, hue=plate_col_name, palette=pallete, multiple="layer"
        )
        plt.xlim(axlim)
        plt.savefig(f"{Path(out_dir, f'{y_value}_{plate_col_name}_histogram')}.png")
        plt.show()
        plt.close()
        hist2 = seaborn_ridgeplot(
            df_sorted,
            value_col=y_value,
            group_col=x_value,
            palette="Set2",
            save=True,
            out_dir=out_dir,
            show_percentiles=True,
        )
        plt.close()
    fig, ax = plt.subplots(figsize=figsize)
    sns.set_context(context=context, font_scale=1.2)
    sns.set_theme(style="ticks")

    feature_df = df_sorted[[x_value, y_value, plate_col_name]].copy()
    if reps_to_exclude:
        feature_df = feature_df[~feature_df[plate_col_name].isin(reps_to_exclude)]
        print(f"removing plates: {reps_to_exclude}")

    # remove outluers
    if remove_outliers is True:
        if rm_outliers_method == "mad":
            feature_df = flag_outliers_by_group_mad(feature_df, x_value, y_value)
            feature_df = feature_df[~feature_df["Outlier"]]
        elif rm_outliers_method == "gesd":
            feature_df = flag_outliers_by_group_gesd(
                feature_df, x_value, y_value, noutliers=50
            )
            feature_df = feature_df[~feature_df["Outlier_GESD"]]
        elif rm_outliers_method == "gesd_2":
            print(feature_df.shape)
            feature_df = remove_outliers_by_group_gesd(
                feature_df, x_value, y_value, noutliers=500
            )
            print(feature_df.shape)
        elif rm_outliers_method == "tietjen":
            feature_df = remove_outliers_by_group_tietjen(
                feature_df, x_value, y_value, noutliers=50
            )
        elif rm_outliers_method == "iqr":
            feature_df = remove_outliers_iqr(feature_df)
        else:
            ValueError(f"{rm_outliers_method} is not a valid outlier removal method")
        # display(feature_df)

    # group by plate and condition
    group_avg_df = feature_df.groupby([x_value, plate_col_name], as_index=False).mean()

    # sort the group avg df by the "AllGroups" order
    group_avg_df_sorted = group_avg_df.sort_values(
        by=[x_value], key=lambda x: x.map(allgroups_sort_key)
    ).reset_index(drop=True)

    ax = super_splitviolinplot_helper_singleplot(
        feature_df,
        group_avg_df_sorted,
        ax,
        x_value,
        y_value,
        title=" ",
        plate_col_name=plate_col_name,
        pairs=pairs,
        order=order,
        annotate=annotate,
        test=test,
        shapiro=False,
        pallete=pallete,
        p_correction=p_correction,
    )

    if legend:
        ax.legend(
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            frameon=True,
            title=plate_col_name,
        )
        if shapiro:
            group_avg_df_shapiro = apply_shapiro_wilk_test_to_df(
                group_avg_df_sorted,
                feature_meas=y_value,
                plate_col_name="Plate_Number",
                alpha=0.05,
            )
            # display(group_avg_df_shapiro)
            ax = annotate_legend_with_shapiro(ax, group_avg_df_shapiro, plate_col_name)
    else:
        ax.legend_.remove()
    if ytitle is not None:
        ax.set_ylabel(ytitle)
    else:
        ax.set_ylabel(y_value.replace("_", " "))
    if xtitle is not None:
        ax.set_xlabel(xtitle)

    # Set the ylim and don't throw an inf
    if ylim is None:
        ylim = axlim
    if ylim[0] is not None and ylim[1] is not None:
        if not (
            np.isnan(ylim[0])
            or np.isnan(ylim[1])
            or np.isinf(ylim[0])
            or np.isinf(ylim[1])
        ):
            ax.set_ylim(ylim)
    plt.tight_layout()
    sns.despine()
    plt.savefig(os.path.join(out_dir, f"{y_value}_{test}.png"))
    if show:
        plt.show()


## Posthocs Outlier tests
def flag_outliers_by_group_mad(df, group_col, feature_col):
    """
    Adds a boolean 'Outlier' column to df, True if the value is an outlier within its group.
    """
    df = df.copy()
    df["Outlier"] = df.groupby(group_col)[feature_col].transform(
        lambda x: pg.madmedianrule(x)
    )
    return df


def flag_outliers_by_group_gesd(df, group_col, feature_col, noutliers=20, report=True):
    """
    Adds a boolean 'Outlier' column to df, True if the value is an outlier within its group.
    """

    df = df.copy()
    df["Outlier_GESD"] = df.groupby(group_col)[feature_col].transform(
        lambda x: sp.outliers_gesd(x, outliers=noutliers, hypo=True, report=report)
    )
    return df


def flag_outliers_by_group_tietjen(df, group_col, feature_col, noutliers=5):
    """
    Adds a boolean 'Outlier_Grubbs' column to df, True if you reject the null hypothesis that the extreme value is an outlier.
    """

    df = df.copy()
    df["Outlier_Tietjen"] = df.groupby(group_col)[feature_col].transform(
        lambda x: sp.outliers_tietjen(x, k=noutliers, hypo=True)
    )
    return df


def remove_outliers_by_group_gesd(df, group_col, feature_col, noutliers=5):
    """
    Filters out outliers in feature_col within each group of group_col using the GEST test.
    Returns a DataFrame with outliers removed.
    """

    df = df.copy()
    filtered_groups = []
    for group_val, group_df in df.groupby(group_col):
        mask = sp.outliers_gesd(group_df[feature_col], outliers=noutliers, hypo=False)
        filtered_group = group_df[group_df[feature_col].isin(mask)]
        filtered_groups.append(filtered_group)
    final_df = pd.concat(filtered_groups, axis=0)
    return final_df


def remove_outliers_by_group_tietjen(df, group_col, feature_col, noutliers=5):
    """
    Filters out outliers in feature_col within each group of group_col using Tietjen's test.
    Returns a DataFrame with outliers removed.
    """

    df = df.copy()
    filtered_groups = []
    for group_val, group_df in df.groupby(group_col):
        mask = sp.outliers_tietjen(group_df[feature_col], k=noutliers, hypo=False)
        filtered_group = group_df[group_df[feature_col].isin(mask)]
        filtered_groups.append(filtered_group)
    final_df = pd.concat(filtered_groups, axis=0)
    return final_df
