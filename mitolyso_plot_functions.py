''' 
Helper functions to build superplots
Allie Spangaro, Toronto Metropolitan University
'''
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from helpers import *

import scipy
import seaborn as sns

'''
See this awesome course note here: https://biapol.github.io/PoL-BioImage-Analysis-TS-Early-Career-Track/day3a_plotting/03_Statistic_Annotations_in_Seaborn_Bonus.html

NOTE for custom annotations in statannotations:
    Use results from a previous stat test e.g.
    stat_results_GMF = [mannwhitneyu(Gentoo_values_female['bill_length_mm'], Gentoo_values_male['bill_length_mm'], alternative="two-sided"),]
    pvalues = [result.pvalue for result in stat_results_GMF]
    formatted_pvalues = [f"p={p:.2e}" for p in pvalues]
    annotator.set_custom_annotations(formatted_pvalues)
'''

def create_superplot_fortwo(data, x_value, y_value, replicates, save_name):

    sns.set_style("whitegrid")

    # Calculate the mean for each group
    group_averages = data.groupby([x_value, replicates], as_index=False).agg({y_value: "mean"})
    group_ave_pivot = group_averages.pivot_table(columns=x_value, values=y_value, index=replicates)

    # Perform paired t-test
    statistic, pvalue = scipy.stats.ttest_rel(group_ave_pivot.iloc[:, 0], group_ave_pivot.iloc[:, 1])
    pvalue_str = str(float(round(pvalue, 3)))
    if pvalue < 0.05:
        pvalue_str = pvalue_str + "*"

    # Create the superplot
    sns.swarmplot(x=x_value, y=y_value, hue=replicates, data=data) # plot the data
    ax = sns.swarmplot(x=x_value, y=y_value, hue=replicates, size=15, edgecolor="k", linewidth=2, data=group_averages) # plot the averages from each replicate
    ax.legend_.remove()  # remove the legend
    
    x1, x2 = 0, 1
    y, h, col = data[y_value].max() + 2, 2, 'k'

    # Add the p-value annotation
    plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
    plt.text((x1 + x2) * .5, y + h * 2, "p = " + pvalue_str, ha='center', va='bottom', color=col)

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
    
    ReplicateAverages = df.groupby(['Treatment','Replicate'], as_index=False).agg({'Speed': "mean"}); 
    ReplicateAvePivot = ReplicateAverages.pivot_table(columns='Treatment', values='Speed', index="Replicate")
    # Calculate 'appropriate' p-value considering n=3
    good_pvalue = ttest_rel(ReplicateAvePivot['Control'], ReplicateAvePivot['Drug']).pvalue
    
    # Just to copy the colors in the paper (https://doi.org/10.1083/jcb.202001064) I extract the RGB values from the original figure
    paper_palette = [
        (0.792156862745098, 0.5529411764705883, 0.1411764705882353), 
        (0.36470588235294116, 0.6274509803921569, 0.7490196078431373), 
        (0.5803921568627451, 0.592156862745098, 0.592156862745098)
    ]

    # Instead of making the background swarmplot white I kept the corresponding categorical color
    # but lower the opacity down by 50%
    # this is the largest difference with the original figure
    alpha_palette = [
        (r,g,b,0.5)
        for (r,g,b) in paper_palette
    ]

    sns.swarmplot( # plot all data points by replicate as swarmplot
        data=df, x='Treatment', y='Speed', hue='Replicate',
        size=4, zorder=0, palette=alpha_palette,
        linewidth=1, legend=False, ax=ax
    )
    
    sns.pointplot( # one dot representing mean of each replicate separated by color and marker type
        data=df, x='Treatment', y='Speed', hue='Replicate',
        palette=paper_palette, linestyle="none", errorbar=None, markers=['o','s','^'], dodge=True,
        markeredgecolor='k', markeredgewidth=1.1, legend=False, ax=ax
    )
    
    sns.pointplot( # standard error bars of mean values of each replicate
        data=ReplicateAverages, x='Treatment', y='Speed', linestyle="none", 
        capsize=.3, errorbar='se', err_kws={'linewidth': 1.4},
        marker="_", markersize=50, markeredgewidth=1.4, color='black', legend=False, ax=ax
    )

    # Plot adjustments
    sns.despine()
    ax.set_ylabel(r'Speed ($\mu$m/min)')
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0,60)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(20))

    x1, x2 = 0, 1
    y, h = df['Speed'].max() + 2, 2

    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c='black')
    ax.text((x1+x2)*.5, y+h*2, "P = {:.3f}".format(good_pvalue), ha='center', va='bottom')
    ax.set_facecolor('#F0F0F7')
    return ax
    
def remove_outliers_iqr(df, col = None):
    """Remove outliers based on interquartile range

    Args:
        df (DataFrame): _description_
        col (string, optional): the name of the column to remove from. Defaults to None, removes all by default.

    Returns:
        _type_: _description_
    """    
    cols = df.select_dtypes('number').columns  # limits to a (float), b (int) and e (timedelta)
    df_sub = df.loc[:, cols]

    iqr = df_sub.quantile(0.75, numeric_only=False) - df_sub.quantile(0.25, numeric_only=False)
    
    #calculate  extreme outlisers by dividing median by iqr
    lim = np.abs((df_sub - df_sub.median()) / iqr) < 2.22

    # replace outliers with nan
    df.loc[:, cols] = df_sub.where(lim, np.nan)
    df.dropna(subset=cols, inplace=True) # drop rows with NaN in numerical columns
    return df

def shapiro_pvalue(group_avg_df, replicate, feature_meas, replicate_col_name ="Replicate_Number", debug=False):
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

def apply_shapiro_wilk_test_to_df(group_avg_df,feature_meas, replicate_col_name ="Replicate_Number", alpha = 0.05):
    """Function to applies the shapiro wilk test row-by-row onto an aggregated dataframe by replicate

    Args:
        group_avg_df (DataFrame): the aggregated dataframe
        feature_meas (string): _description_
        replicate_col_name (str, optional): the name of the replicate column. Defaults to "Replicate_Number".
        alpha (float): the p value threshold. Defaults to p=0.05

    Returns:
        DataFrame: The aggregated dataframe with a "Shaprio_pvalue" column and a boolean "Shapiro_normality" column
    """    
    #apply the shapiro-wilk test to a dataframe
    group_avg_df = group_avg_df.dropna()
    group_avg_df["Shapiro_pvalue"] = group_avg_df.apply(
        lambda row: shapiro_pvalue(
            group_avg_df,
            replicate=row[replicate_col_name],
            feature_meas=feature_meas
        ), axis=1)
    #reject null hypothesis if p < 0.05 - i.e. significant chance that the data is not normally distributed
    group_avg_df["Shapiro_normality"] = group_avg_df.apply(
        lambda row: 
            row['Shapiro_pvalue'] > alpha,
        axis=1)
    return group_avg_df

def oneway_anova(data, group_name, feature_meas):
  from scipy.stats import f_oneway

  data = data.dropna(subset=[group_name, feature_meas])
  data = data[data[feature_meas] != 0]
  
  groups = data[group_name].unique()
  data = [data[data[group_name] == group][feature_meas].dropna() for group in groups]
  anova_result = f_oneway(*data)
  
  print(f"ANOVA F-statistic: {anova_result.statistic}, ANOVA p-value: {anova_result.pvalue}")
  return anova_result

def tukey_test(data, test_groups, feature):
    """
    Perform a oneway anova test and a pairwise tukey post hoc test using averaged values per replicate
    Returns a dataframe
    """
    from scipy.stats import f_oneway
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    df = data.copy()

    # groups = getpairs(temp_copy, 'Passage Group')
    # calculate tukey HSD

    tukey = pairwise_tukeyhsd(endog=df[feature], groups=df[test_groups], alpha=0.05)

    # Extract relevant results
    results = np.array(tukey.summary().data)[:, [0, 1, 3, 6]]
    df_results = pd.DataFrame(
        results, columns=["Group 1", "Group 2", "p-value", "Reject"]
    ).drop([0])
    df_results.reset_index(drop=True, inplace=True)
    df_results[["Group 1", "Group 2"]] = df_results[["Group 1", "Group 2"]]
    df_results["p-value"] = df_results["p-value"].astype(float)

    return df_results

def make_superviolinplot_with_anova_tukey(
    data,
    group,
    feature_meas,
    replicates,
    order,
    ylabel=None,
    pallete="pastel",
    remove_outliers=True,
    save_path="plots/new_plots/",
    figsize=(12, 8),
    dpi=300):
    """_summary_

    Args:
        data (_type_): _description_
        group (_type_): _description_
        feature_meas (_type_): _description_
        replicates (_type_): _description_
        order (_type_): _description_
        ylabel (_type_, optional): _description_. Defaults to None.
        pallete (str, optional): _description_. Defaults to "pastel".
        remove_outliers (bool, optional): _description_. Defaults to True.
        save_path (str, optional): _description_. Defaults to "plots/new_plots/".
        figsize (tuple, optional): _description_. Defaults to (12, 8).
        dpi (int, optional): _description_. Defaults to 300.
    """    
    #gather data with the grouping functions in helpers.py
    pairs = getpairs(data, group, order)
    feature_df = make_single_feature_df(data, group=group, feature=feature_meas, replicates=replicates)
    group_avg_df = average_groups_by_plate(feature_df, x_value=group, y_value=feature_meas, replicates=replicates)
    group_avg_df_pivot = average_groups_pivot(group_avg_df, x_value=group, y_value=feature_meas, replicates=replicates)
    
    if remove_outliers:
        feature_df = remove_outliers_iqr(feature_df)

    sns.set_theme(style="ticks")
    plt.figure(figsize=figsize, dpi=dpi)
    sns.set_context("talk", font_scale=0.5)

    sns.violinplot(
        data=feature_df,
        x=group,
        y=feature_meas,
        order=order,
        fill=False,
        color='gainsboro',
        cut=1,
        native_scale=True,
        linecolor='k',
        inner=None,
    )

    ax = sns.swarmplot(
        data=group_avg_df,
        x=group,
        y=feature_meas,
        hue=replicates,
        order=order,
        palette=pallete,
        size=10,
        edgecolor="k",
        linewidth=1,
        dodge=0.5
    )

    sns.boxplot(
        data=group_avg_df,
        x=group,
        y=feature_meas,
        showmeans=True,
        meanline=True,
        meanprops={'color': 'dimgray', 'ls': '-', 'lw': 2.5},
        medianprops={'visible': False},
        whiskerprops={'visible': False},
        zorder=1,
        showfliers=False,
        showbox=False,
        showcaps=False,
        ax=ax
    )

    ax.legend_.remove()
    sns.despine()
    plt.gcf()
    plt.xlabel(group)
    if ylabel is None:
        plt.ylabel(feature_meas.replace('_', ' '))
    else:
        plt.ylabel(ylabel)

    from statannotations.Annotator import Annotator
    from statannotations.stats.StatTest import StatTest

    anova = oneway_anova(group_avg_df, group, feature_meas)
    tukey_results = tukey_test(group_avg_df, test_groups=group, feature=feature_meas)
    
    annotator = Annotator(ax, pairs, data=group_avg_df_pivot, order=order)
    annotator.configure(text_format='star', loc='inside', verbose=2, hide_non_significant=True)
    annotator.set_pvalues_and_annotate(tukey_results['p-value'])

    plt.savefig(f"{save_path}{feature_meas}_anova_superviolinplot.png", dpi=dpi)
    plt.show()


def make_superviolinplot_with_kruskal(data, group, feature_meas, replicates, xtitle=None, ytitle = None, pallete='pastel', ylim = None, order = None, remove_outliers = False):
    """_summary_

    Args:
        data (DataFrame): _description_
        group (string): _descriThe ption_
        feature_meas (string): _description_
        replicates (_type_): _description_
        xtitle (_type_, optional): _description_. Defaults to None.
        ytitle (_type_, optional): _description_. Defaults to None.
        pallete (str, optional): _description_. Defaults to 'pastel'.
        ylim (_type_, optional): _description_. Defaults to None.
        order (_type_, optional): _description_. Defaults to None.
        remove_outliers (bool, optional): _description_. Defaults to False.
    """    
    if order is None:
        order = data[group].dropna().unique().tolist()
    
    if ytitle is None:
        ytitle = feature_meas.replace('_', ' ')
    if xtitle is None:
        xtitle = group.replace('_', ' ')
       
    feature_df = make_single_feature_df(data, group=group, feature=feature_meas, replicates=replicates)
     
    if remove_outliers is True:
        feature_df = remove_outliers_iqr(feature_df)        
    
    pairs = getpairs(feature_df, group, order)


    group_avg_df = average_groups_by_plate(feature_df, x_value=group, y_value=feature_meas, replicates=replicates)
    group_avg_df_pivot = average_groups_pivot(group_avg_df, x_value=group, y_value=feature_meas, replicates=replicates)

    sns.set_theme(style="ticks")
    sns.set_context("talk", font_scale=0.5)

    plt.figure(dpi=300)

    sns.violinplot(data=feature_df, x=group,
                y=feature_meas,
                order=order,
                fill = False,
                color= 'gainsboro',
                cut=2,
                native_scale=True,
                linecolor='k',
                inner= None,
                #inner_kws=dict(box_width = 5)
                )

    ax = sns.swarmplot(data=group_avg_df, x=group,
                y=feature_meas,
                hue = replicates,
                order=order,
                palette=pallete,
                size=10, 
                edgecolor="k", 
                linewidth=1,
                dodge=0.5)
    
    #use a boxplot to draw the mean line - thinking outside the box :)
    sns.boxplot(data = group_avg_df, x = group,
                y = feature_meas,
                showmeans=True,
                meanline=True,
                meanprops={'color': 'dimgray', 'ls': '-', 'lw': 2.5},
                medianprops={'visible': False},
                whiskerprops={'visible': False},
                zorder=1,
                showfliers=False,
                showbox=False,
                showcaps=False,
                ax = ax)

    ax.legend_.remove()

    sns.despine()
    plt.gcf()#.set_size_inches(10, 6)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.ylim(ylim)

    from statannotations.Annotator import Annotator
    annotator = Annotator(ax, pairs, data=group_avg_df_pivot, order=order) 
    annotator.configure(test='Kruskal', 
                        text_format='star', 
                        #pvalue_format = 'simple',
                        loc='inside', 
                        hide_non_significant = True,
                        color = 'black',
                        verbose = 2)
    annotator.apply_and_annotate()

    plt.savefig("plots/new_plots/" + feature_meas + '_superviolinplot.png', dpi=300)
    plt.show()
    
def make_superswarmplot_with_kruskal(data, group, feature_meas, replicates, order = None, ytitle = None, xtitle = None, ylim = None, pallete='pastel'):
    """_summary_

    Args:
        data (_type_): _description_
        group (_type_): _description_
        feature_meas (_type_): _description_
        replicates (_type_): _description_
        order (_type_, optional): _description_. Defaults to None.
        ytitle (_type_, optional): _description_. Defaults to None.
        xtitle (_type_, optional): _description_. Defaults to None.
        ylim (_type_, optional): _description_. Defaults to None.
        pallete (str, optional): _description_. Defaults to 'pastel'.
    """    
    #Set values to defaults if a parameter is not loaded for order or y axis parameters
    if ytitle is None:
        ytitle = feature_meas.replace('_', ' ')
    if ylim is None:
        ylim = (-1,12)
    if order is None:
        order = data[group].dropna().unique().tolist()
    if xtitle is None:
        xtitle = group
    
    feature_df = make_single_feature_df(data, group=group, feature=feature_meas, replicates=replicates)
    pairs = getpairs(data, group, order)
    print(pairs)

    group_avg_df = average_groups_by_plate(feature_df, x_value=group, y_value=feature_meas, replicates=replicates)
    group_avg_df_pivot = average_groups_pivot(group_avg_df, x_value=group, y_value=feature_meas, replicates=replicates)

    sns.set_theme(style="ticks")
    sns.set_context("talk", font_scale=0.6)

    plt.figure(dpi=300)

    sns.swarmplot(data=feature_df, x=group,
                y=feature_meas,
                hue = replicates,
                order=order,
                palette=pallete,
                )

    ax = sns.swarmplot(data=group_avg_df, x=group,
                y=feature_meas,
                hue = replicates,
                order=order,
                palette=pallete,
                size=10, 
                edgecolor="k", 
                linewidth=1,
                dodge=0.5)
    
    sns.boxplot(data = group_avg_df, x = group,
                y = feature_meas,
                showmeans=True,
                meanline=True,
                meanprops={'color': 'dimgray', 'ls': '-', 'lw': 2.5},
                medianprops={'visible': False},
                whiskerprops={'visible': False},
                zorder=1,
                showfliers=False,
                showbox=False,
                showcaps=False,
                ax = ax)
    
    if ax.legend_ is not None:
        ax.legend_.remove()

    sns.despine()
    plt.gcf()#.set_size_inches(10, 6)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.ylim(ylim)

    from statannotations.Annotator import Annotator
    annotator = Annotator(ax, pairs, data=group_avg_df_pivot, order=order) 
    annotator.configure(test='Kruskal', 
                        text_format='star', 
                        loc='inside', 
                        hide_non_significant = True,
                        color = 'black',
                        verbose = 2)
    annotator.apply_and_annotate()

    plt.savefig(xtitle + '_'+ feature_meas + '_superswarmplot.png', dpi=300)
    plt.show()
# Example usage
ylim = (0,100)

def make_superswarmplot_with_anova(data, group, feature_meas, replicates, order = None, ytitle = None, ylim = None, xtitle= None, pallete='pastel'):
    """_summary_

    Args:
        data (_type_): _description_
        group (_type_): _description_
        feature_meas (_type_): _description_
        replicates (_type_): _description_
        order (_type_, optional): _description_. Defaults to None.
        ytitle (_type_, optional): _description_. Defaults to None.
        ylim (_type_, optional): _description_. Defaults to None.
        xtitle (_type_, optional): _description_. Defaults to None.
        pallete (str, optional): _description_. Defaults to 'pastel'.
    """    
    #Set values to defaults if a parameter is not loaded for order or y axis parameters
    if ytitle is None:
        ytitle = feature_meas.replace('_', ' ')
    if ylim is None:
        ylim = (-1,12)
    if order is None:
        order = data[group].dropna().unique().tolist()
    if xtitle is None:
        xtitle = group
    
    feature_df = make_single_feature_df(data, group=group, feature=feature_meas, replicates=replicates)
    pairs = getpairs(data, group, order)
    print(pairs)

    #Remove the n=1 replicate in the passage group code
    if group == "Passage Group" and order==None:
        order = ['P6-8', 'P9-10', 'P11-13', 'P14-16', 'P17-18', 'P19-21', 'P22-24','P25-26','P27-28','P29+', 'Doxo']
        #feature_df = feature_df[feature_df[group] != "P22-24"]

    group_avg_df = average_groups_by_plate(feature_df, x_value=group, y_value=feature_meas, replicates=replicates)
    group_avg_df_pivot = average_groups_pivot(group_avg_df, x_value=group, y_value=feature_meas, replicates=replicates)

    sns.set_theme(style="ticks")
    sns.set_context("talk", font_scale=0.6)

    plt.figure(dpi=300)

    sns.swarmplot(data=feature_df, x=group,
                y=feature_meas,
                hue = replicates,
                order=order,
                palette=pallete,
                )

    ax = sns.swarmplot(data=group_avg_df, x=group,
                y=feature_meas,
                hue = replicates,
                order=order,
                palette=pallete,
                size=10, 
                edgecolor="k", 
                linewidth=1,
                dodge=0.5)
    
    sns.boxplot(data = group_avg_df, x = group,
                y = feature_meas,
                showmeans=True,
                meanline=True,
                meanprops={'color': 'dimgray', 'ls': '-', 'lw': 2.5},
                medianprops={'visible': False},
                whiskerprops={'visible': False},
                zorder=1,
                showfliers=False,
                showbox=False,
                showcaps=False,
                ax = ax)
    if ax.legend_ is not None:
        ax.legend_.remove()

    sns.despine()
    plt.gcf()#.set_size_inches(10, 6)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.ylim(ylim)
    
    from statannotations.stats.StatTest import StatTest
    custom_long_name = 'One-way ANOVA statistical test'
    custom_short_name = 'One-way ANOVA'
    custom_func = stats.f_oneway
    custom_test = StatTest(custom_func, custom_long_name, custom_short_name)

    from statannotations.Annotator import Annotator
    annotator = Annotator(ax, pairs, data=group_avg_df_pivot, order=order) 
    annotator.configure(test=custom_test, 
                        text_format='star', 
                        loc='inside', 
                        hide_non_significant = True,
                        color = 'black',
                        verbose = 2)
    annotator.apply_and_annotate()

    plt.savefig(xtitle + '_'+ feature_meas + '_superswarmplot.png', dpi=300)
    plt.show()
    
def make_superswarmplot_with_calculated_pval(data, group, feature_meas, replicates, order = None, ytitle = None, xtitle = None, ylim = None, pallete='pastel', stattest_results= None):
    """_summary_

    Args:
        data (_type_): _description_
        group (_type_): _description_
        feature_meas (_type_): _description_
        replicates (_type_): _description_
        order (_type_, optional): _description_. Defaults to None.
        ytitle (_type_, optional): _description_. Defaults to None.
        xtitle (_type_, optional): _description_. Defaults to None.
        ylim (_type_, optional): _description_. Defaults to None.
        pallete (str, optional): _description_. Defaults to 'pastel'.
        stattest_results (_type_, optional): _description_. Defaults to None.
    """    
    #Set values to defaults if a parameter is not loaded for order or y axis parameters
    if ytitle is None:
        ytitle = feature_meas.replace('_', ' ')
    if ylim is None:
        ylim = (-1,12)
    if order is None:
        order = data[group].dropna().unique().tolist()
        
    if xtitle is None:
        xtitle = group
    
    feature_df = make_single_feature_df(data, group=group, feature=feature_meas, replicates=replicates)
    pairs = getpairs(data, group, order)
    print(pairs)

    #Remove the n=1 replicate in the passage group code
    if group == "Passage Group":
        order = ['P6-8', 'P9-10', 'P11-13', 'P14-16', 'P17-18', 'P19-21', 'P22-24','P25-26','P27-28','P29+', 'Doxo']
        #feature_df = feature_df[feature_df[group] != "P22-24"]

    group_avg_df = average_groups_by_plate(feature_df, x_value=group, y_value=feature_meas, replicates=replicates)
    group_avg_df_pivot = average_groups_pivot(group_avg_df, x_value=group, y_value=feature_meas, replicates=replicates)

    sns.set_theme(style="ticks")
    sns.set_context("talk", font_scale=0.6)

    plt.figure(dpi=300)

    sns.swarmplot(data=feature_df, x=group,
                y=feature_meas,
                hue = replicates,
                order=order,
                palette=pallete,
                )

    ax = sns.swarmplot(data=group_avg_df, x=group,
                y=feature_meas,
                hue = replicates,
                order=order,
                palette=pallete,
                size=10, 
                edgecolor="k", 
                linewidth=1,
                dodge=0.5)
    
    sns.boxplot(data = group_avg_df, x = group,
                y = feature_meas,
                showmeans=True,
                meanline=True,
                meanprops={'color': 'dimgray', 'ls': '-', 'lw': 2.5},
                medianprops={'visible': False},
                whiskerprops={'visible': False},
                zorder=1,
                showfliers=False,
                showbox=False,
                showcaps=False,
                ax = ax)
    
    if ax.legend_ is not None:
        ax.legend_.remove()

    sns.despine()
    plt.gcf()#.set_size_inches(10, 6)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.ylim(ylim)
    
    if stattest_results is not None:
        from statannotations.stats.StatTest import StatTest
        custom_long_name = 'One-way ANOVA statistical test'
        custom_short_name = 'One-way ANOVA'
        custom_func = stats.f_oneway
        custom_test = StatTest(custom_func, custom_long_name, custom_short_name)

        from statannotations.Annotator import Annotator
        annotator = Annotator(ax, pairs, data=group_avg_df_pivot, order=order) 
        annotator.configure(
                            text_format='star', 
                            loc='inside', 
                            hide_non_significant = True,
                            color = 'black',
                            verbose = 2)
        annotator.set_pvalues_and_annotate(stattest_results['p-value'])

    plt.savefig(xtitle + '_'+ feature_meas + '_superswarmplot.png', dpi=300)
    plt.show()
    
