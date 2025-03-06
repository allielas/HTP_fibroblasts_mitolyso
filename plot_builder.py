''' 
Helper functions to build superplots
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
    
    
def remove_outliers_iqr(df, col = None):
    cols = df.select_dtypes('number').columns  # limits to a (float), b (int) and e (timedelta)
    df_sub = df.loc[:, cols]

    iqr = df_sub.quantile(0.75, numeric_only=False) - df_sub.quantile(0.25, numeric_only=False)
    
    #calculate  extreme outlisers by dividing median by iqr
    lim = np.abs((df_sub - df_sub.median()) / iqr) < 2.22

    # replace outliers with nan
    df.loc[:, cols] = df_sub.where(lim, np.nan)
    df.dropna(subset=cols, inplace=True) # drop rows with NaN in numerical columns
    return df

def make_superplot_with_kruskal(data, group, feature_meas, replicates, ytitle = None, pallete='pastel', ylim = None, remove_outliers = remove_outliers):
    
    order = ['P6-8', 'P9-10', 'P11-13', 'P14-16', 'P17-18', 'P20-21']#, 'P22-24']
    
    if ytitle is None:
        ytitle = feature_meas.replace('_', ' ')

       
    feature_df = make_single_feature_df(data, group=group, feature=feature_meas, replicates=replicates)
     
    if remove_outliers is True:
        feature_df = remove_outliers_iqr(feature_df)
        display(feature_df)
        
    
    pairs = getpairs(feature_df, group, order)

    #Remove the n=1 replicate
    feature_df = feature_df[feature_df[group] != "P22-24"]


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
                cut=1,
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
    plt.xlabel(group)
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

    plt.savefig(feature_meas + '_superviolinplot.png', dpi=300)
    plt.show()
    
def make_superswarmplot_with_kruskal(data, group, feature_meas, replicates, order = None, ytitle = None, xtitle = None, ylim = None, pallete='pastel'):
    
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
        order = ['P6-8', 'P9-10', 'P11-13', 'P14-16', 'P17-18', 'P20-21']#, 'P22-24']
        feature_df = feature_df[feature_df[group] != "P22-24"]

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
        order = ['P6-8', 'P9-10', 'P11-13', 'P14-16', 'P17-18', 'P20-21']#, 'P22-24']
        feature_df = feature_df[feature_df[group] != "P22-24"]

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
        order = ['P6-8', 'P9-10', 'P11-13', 'P14-16', 'P17-18', 'P20-21']#, 'P22-24']
        feature_df = feature_df[feature_df[group] != "P22-24"]

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
    
