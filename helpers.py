''' 
Helper functions for data analysis and visualization
'''
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def passage_group(passage_num):
    '''
    Group passages into groups for plotting
    returns string of the group that the passage number belongs to
    '''
    #use this function to group passages into groups for plotting
    passage = 'P' + str(passage_num)
    passage_group_dict = {}
    passage_groups = ['P6-8','P9-10','P11-13','P14-16','P17-18','P20-21','P22-24']
    for group in passage_groups:
        if group == 'P6-8':
            passage_group_dict['P6'] = 'P6-8'
            passage_group_dict['P7'] = 'P6-8'
            passage_group_dict['P8'] = 'P6-8'
        elif group == 'P9-10':
            passage_group_dict['P9'] = 'P9-10'
            passage_group_dict['P10'] = 'P9-10'
        elif group == 'P11-13':
            passage_group_dict['P11'] = 'P11-13'
            passage_group_dict['P12'] = 'P11-13'
            passage_group_dict['P13'] = 'P11-13'
        elif group == 'P14-16':
            passage_group_dict['P14'] = 'P14-16'
            passage_group_dict['P15'] = 'P14-16'
            passage_group_dict['P16'] = 'P14-16'
        elif group == 'P17-18':
            passage_group_dict['P17'] = 'P17-18'
            passage_group_dict['P18'] = 'P17-18'
        elif group == 'P20-21':
            passage_group_dict['P20'] = 'P20-21'
            passage_group_dict['P21'] = 'P20-21'
        elif group == 'P22-24':
            passage_group_dict['P22'] = 'P22-24'
            passage_group_dict['P23'] = 'P22-24'
            passage_group_dict['P24'] = 'P22-24'
            
    #return the group that the passage number belongs to
    return passage_group_dict[passage]

def plate_df_setup(curr_plates, curr_plate_datafolders, parent_dir, csv_names = ['Cell.csv', 'Nuclei.csv','MergedMitoPerCell.csv','MergedLysoPerCell.csv']):
    '''
    Combine the cellprofiler feature data from different plates into a single DataFrame
    Returns a DataFrame with the combined data
    '''
    #Initialize a list to store the combined DataFrames
    plate_dfs = {}
    
    for i,plate in enumerate(curr_plates):
        # Construct the full path to the folder
        folder_path = os.path.join(parent_dir, plate)
        
        # Construct the full path to the metadata file and CSV file
        map_file = os.path.join(folder_path, 'metadata/map.csv')
        csv_folder_path = os.path.join(folder_path, curr_plate_datafolders[i])
        
        #Make a list of the csv file paths for each compartment
        compartment_paths = []
        
        for file in csv_names:
            cp_file = os.path.join(csv_folder_path, file)
            if os.path.exists(cp_file) and file in csv_names:
                compartment_paths.append(cp_file)
        
        #Join the file dataframes
        if 'Cell.csv' in compartment_paths[0]:
            pre_cell_df = pd.read_csv(compartment_paths[0])
        else:
            return FileNotFoundError('Cell.csv not found in the folder')  
    
        for j,compartment in enumerate(compartment_paths):
            if j == 0 and 'Cell.csv' in compartment:
                continue
            
            compartment_df = pd.read_csv(compartment)
            excluded_columns = ['ImageNumber', 'ObjectNumber']
            
            prefix = csv_names[j].replace('.csv','') + '_'
            
            keys_df = compartment_df[excluded_columns]
            excluded_keys_df = compartment_df.drop(columns=excluded_columns)
            
            prefixed_compartment_df = excluded_keys_df.add_prefix(prefix)
            combined_prefixed_compartment_df = pd.concat([keys_df, prefixed_compartment_df], axis=1)
            
            pre_cell_df = pre_cell_df.merge(combined_prefixed_compartment_df, on=['ImageNumber','ObjectNumber'], how='left')
            
        #Join the metadata with the data
        if os.path.exists(cp_file) and os.path.exists(map_file):
            # Read the metadata file and merge with dataframes (map.csv)
            platemap_df = pd.read_csv(map_file)
            cell_df = pre_cell_df.merge(platemap_df, on=['Metadata_Well', 'Metadata_WellRow', 'Metadata_WellColumn', 'Metadata_Field'], how='left')            
            
            #Add a column to the cell_df to group passages and identify the plate replicate
            cell_df["Passage Group"] = cell_df['PassageNumber'].apply(passage_group)
            cell_df["Metadata_Plate"] = plate
            cell_df["Replicate_Number"] = i+1
            # Append the merged DataFrame to the list
            plate_dfs[plate] = cell_df
            
        
    # Combine all the different replicate DataFrames into a single DataFrame
    combined_replicates_df = pd.concat(plate_dfs.values(), ignore_index=True)
    
    #Filter DataFrames to only include cells that were stained with LAMP1-488 and MitoRed
    combined_replicates_df_mitolyso = combined_replicates_df[combined_replicates_df['Staining'].str.startswith("LAMP1-488 + MitoRed")]
    return combined_replicates_df_mitolyso


def define_cell_features(df):
    # Get the columns of the dataframe
    columns_list = df.columns.tolist()
    columns_list = [col for col in columns_list if 'Metadata' not in col and 'FileName' not in col and 'PathName' not in col]
    return columns_list

def make_feature_dict(columns_list):
    # Add the different types of features to a dictionary 
    feature_dict = {
        'intensity': [],
        'texture': [],
        'areashape': [],
        'granularity': [],
        'radialdistribution': [],
        'arearatios': [],
        'count': [],
        'distance' : [],
        'metadata': []
    }
    for col in columns_list:
        if 'Texture' in col:
            feature_dict['texture'].append(col)
        elif 'Intensity' in col:
            feature_dict['intensity'].append(col)
        elif 'Math_' in col:
            feature_dict['arearatios'].append(col)
        elif 'Count' in col:
            feature_dict['count'].append(col)
        elif 'AreaShape' in col:
            feature_dict['areashape'].append(col)
        elif 'Distance' in col:
            feature_dict['distance'].append(col)
        elif 'Granularity' in col:
            feature_dict['granularity'].append(col)
        elif 'RadialDistribution' in col:
            feature_dict['radialdistribution'].append(col)
        else:
            feature_dict['metadata'].append(col)

    return feature_dict

def getpairs(df, group, order = []):
    from itertools import combinations
    # Get the unique values of the categorical column, Order the unique values according to the specified order
    unique_values = df[group].dropna().unique()
    
    ordered_values = [value for value in order if value in unique_values]
    
    pairs = list(combinations(ordered_values, 2))
    return pairs

#Function to find the ratio between two columns in the two dataframes and return the ratio as a column
def ratioCalc(df1, df2, col1, col2):
    #Deprecate this function
  int1 = df1[col1]
  int2 = df2[col2]

  temp_copy1 = outlier_removal(df1, int1)
  temp_copy2 = outlier_removal(df2, int2)

  intensity_ratio = temp_copy1[int1] / temp_copy2[int2]
  return df[intensity_ratio]

def standardize_group(df, columns):
    from sklearn.preprocessing import StandardScaler
   #Import the scaler and transform all time values to that of a standard distribution - only use for ML, not very desceiptive
    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(df[columns])
    return scaled_df
    
def group_by_time(df, feature_list):
    #Group columns by time and apply groupby function to the DF
    df_groupby = df.groupby('Time').apply(lambda x: standardize_group(x,feature_list))
    return df_groupby

def normalize_to_control(df, feature):
    # Take the t0 df - lowest passage data point
    t0_df = df[df['Time']==0]
    treatment_df = df[[feature, 'Time']].copy()
    #calculate the mean
    mean_zero = t0_df[feature].mean()
    
    #now update the column to have all rows dividied by the mean of time 0
    treatment_df["norm_" + feature] = treatment_df[feature] / mean_zero  
    #return the normalized feature columnn
    return treatment_df["norm_" + feature]

def outlier_removal(df, nuclei_df, column):
    # Create a copy of the column and the 'Time' column, along with parent nuclei
    mini_df = pd.DataFrame({
        column: df[column].copy(),
        'Time': df['Time'].copy(),
        'Parent_Nuclei': df['Parent_Nuclei'].copy(),
        'ImageNumber': df['ImageNumber'].copy()
    })

    # remove stuff within 1 SD above of the mean of the oldest passage
    nuc_oldest_mean = []
    nuc_oldest_std_dev = []
    try:
        nuc_oldest_mean = mini_df[mini_df['Time'] == 6][column].mean()
        nuc_oldest_std_dev = mini_df[mini_df['Time'] == 6][column].std()
    except:
        nuc_oldest_mean = mini_df[mini_df['Time'] == 4][column].mean()
        nuc_oldest_std_dev = mini_df[mini_df['Time'] == 4][column].std()
    nuc_threshold = nuc_oldest_mean + (nuc_oldest_std_dev)
    
    #print('threshold ', nuc_threshold, 'stdev=',nuc_t4_std_dev, 'mean=', nuc_t4_mean)
    # Filter the nuclei dataframe to remove rows with AreaShape_Area above the threshold
    filtered_nuclei_df = nuclei_df[nuclei_df['AreaShape_MeanRadius'] <= nuc_threshold]
    
    # Filter the cell dataframe to keep only rows where Parent_Nuclei is in the filtered DF
    filtered_cell_df = mini_df.merge(filtered_nuclei_df[['Number_Object_Number', 'ImageNumber']], 
                                     left_on=['Parent_Nuclei', 'ImageNumber'], 
                                     right_on=['Number_Object_Number', 'ImageNumber'],
                                     how='inner')

    # Filter out values less than 0 
    final_filtered_df = filtered_cell_df[filtered_cell_df[column] > 0].dropna()
    return final_filtered_df

def normalize_features(df, feature_list):
    # Normalize the features to the control (Time 0) for each plate
    norm_df = df.copy()
    for feature in feature_list:
        #print('Normalizing feature: ', feature, '...', norm_df[feature].values[0])
        norm_df[feature] = normalize_to_control(df, feature)
        #print('Normalized feature: ', feature, '...', norm_df[feature].values[0])
    return norm_df


def apply_feature_normalization(df, feature_dict, curr_plates):
    # Normalize the features to the control (Time 0) for each plate
    norm_cell_df = df.copy()
    for plate in curr_plates:
        curr_plate_df = norm_cell_df[norm_cell_df['Metadata_Plate'] == plate].copy()
        for feature_type in feature_dict:
            #get the normalized features, locate the corresponding features on the plate, and replace them on that plate to the plate
            curr_plate_features_df = normalize_features(curr_plate_df, feature_dict[feature_type])
            curr_plate_df.loc[:, feature_dict[feature_type]] = curr_plate_features_df[feature_dict[feature_type]]
        norm_cell_df.loc[norm_cell_df['Metadata_Plate'] == plate] = curr_plate_df
    return norm_cell_df

def mean_intesity_per_compartment_per_cell(df, compartment, tag):
    # Calculate the mean intensity of each compartment per cell
    #mean_intesity_per_compartment = integrated / (children*mean_area)
    colname = 'MeanIntensity_Per_' + compartment + '_Per_Cell'
    integrated = 'Intensity_IntegratedIntensity_' + tag
    children = 'Children_' + compartment + '_Count'
    mean_area = 'Mean_'+ compartment + '_AreaShape_Area'
    df[colname] = df.apply(lambda x: x[integrated] / (x[children] * x[mean_area]), axis=1)
    return df[colname]

def proportion_area_occupied_per_cell(df, compartment):
    #proportion of area occupied = children * mean organelle area / cell area
    colname = 'Total_Area_Proportion_' + compartment + '_Per_Cell'
    
    #children = 'Children_' + compartment + '_Count'
    #mean_organelle_area = 'Mean_'+ compartment + '_AreaShape_Area'
    organelle_area = compartment + '_AreaShape_Area'
    cell_area = 'AreaShape_Area'
    #df[colname] = df.apply(lambda x: (x[children] * x[mean_organelle_area]) / x[cell_area], axis=1)
    df[colname] = df.apply(lambda x: (x[organelle_area]) / x[cell_area], axis=1)
    return df[colname]


def tukey_test(data, test_groups, feature):
  '''
  Perform a oneway anova test and a pairwise tukey post hoc test using averaged values per replicate
  Returns a dataframe
  '''
  from stats import f_oneway
  from statsmodels.stats.multicomp import pairwise_tukeyhsd
  
  df = data.copy()
  
  #groups = getpairs(temp_copy, 'Passage Group')
  #calculate tukey HSD
  
  tukey = pairwise_tukeyhsd(endog=df[feature], groups=df[test_groups], alpha=0.05)

  # Extract relevant results
  results = np.array(tukey.summary().data)[:, [0, 1, 3, 6]]
  df_results = pd.DataFrame(results, columns=['Group 1', 'Group 2', 'p-value', 'Reject']).drop([0])
  df_results.reset_index(drop=True, inplace=True)
  df_results[['Group 1', 'Group 2']] = df_results[['Group 1', 'Group 2']]
  df_results['p-value'] = df_results['p-value'].astype(float)
  
  return df_results



def make_superviolinplot_with_kruskal(data, group, feature_meas, replicates, ytitle = None, pallete='bright', ylim = None):
    
    order = ['P6-8', 'P9-10', 'P11-13', 'P14-16', 'P17-18', 'P20-21']#, 'P22-24']
    
    if ytitle is None:
        ytitle = feature_meas.replace('_', ' ')
    if ylim is None:
        ylim = (-1,12)
        
    
    feature_df = make_single_feature_df(data, group=group, feature=feature_meas, replicates=replicates)
    pairs = getpairs(feature_df, group, order)

    #Remove the n=1 replicate
    feature_df = feature_df[feature_df[group] != "P22-24"]

    group_avg_df = average_groups_by_plate(feature_df, x_value=group, y_value=feature_meas, replicates=replicates)
    group_avg_df_pivot = average_groups_pivot(group_avg_df, x_value=group, y_value=feature_meas, replicates=replicates)

    sns.set_theme(style="ticks")
    sns.set_context("talk", font_scale=0.6)

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

    sns.pointplot(data=group_avg_df, x=group,
                y=feature_meas,
                color='dimgray',
                order=order,
                dodge=False,
                markers='_',
                linestyle=None,
                errorbar=None,
                ax=ax)

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

def proportion_area_occupied_per_cell_fromtotal(df, compartment):
    #proportion of area occupied = children * mean organelle area / cell area
    colname = 'Total_Area_Proportion_' + compartment + '_Per_Cell'
    
    #children = 'Children_' + compartment + '_Count'
    #mean_organelle_area = 'Mean_'+ compartment + '_AreaShape_Area'
    organelle_area = compartment + '_AreaShape_Area'
    cell_area = 'AreaShape_Area'
    #df[colname] = df.apply(lambda x: (x[children] * x[mean_organelle_area]) / x[cell_area], axis=1)
    df[colname] = df.apply(lambda x: (x[organelle_area]) / x[cell_area], axis=1)
    return df[colname]

def mean_intesity_per_compartment_per_cell_fromtotal(df,compartment, name, tag):
    # Calculate the mean intensity of each compartment per cell
    #mean_intesity_per_compartment = integrated / (children*mean_area)
    colname = 'MeanIntensity_Per_' + compartment + '_Per_Cell'
    integrated = 'Intensity_IntegratedIntensity_' + tag
    #children = 'Children_' + compartment + '_Count'
    #mean_area = 'Mean_'+ compartment + '_AreaShape_Area'
    total_organelle_area = name + '_AreaShape_Area'
    df[colname] = df.apply(lambda x: x[integrated] / x[total_organelle_area], axis=1)
    return df[colname]

def average_groups_by_plate(df, x_value, y_value, replicates):
    '''
    Group the DataFrame by the specified columns and calculate the mean of the y_value column.
    Returns the averaged dataframe for plotting
    '''
    df = df.dropna(subset=[x_value, y_value, replicates])
    df = df[df[y_value] != 0]

    df.reset_index(drop=True, inplace=True)
    
    group_averages = df.groupby([x_value, replicates], as_index=False, observed=True).agg({y_value: "mean"})
    
    # Reset the index to get a clean DataFrame
    average_df = group_averages.reset_index()
   
    return average_df

def make_single_feature_df(data, group, feature, replicates):
  pd.options.mode.copy_on_write = True
  
  subset=[group, feature, replicates]
  
  df = data.dropna(subset = subset).reset_index(drop=True)
  df = df[df[feature] != 0]
  
  df_subset = df[subset]
  df_subset[group] = df[group].astype('category')
  df_subset.reset_index(drop = True, inplace = True)
  
  return df_subset 

def oneway_anova(data, group_name, feature_meas):
  from scipy.stats import f_oneway

  data = data.dropna(subset=[group_name, feature_meas])
  data = data[data[feature_meas] != 0]
  
  groups = data[group_name].unique()
  data = [data[data[group_name] == group][feature_meas].dropna() for group in groups]
  anova_result = f_oneway(*data)
  
  print(f"ANOVA F-statistic: {anova_result.statistic}, ANOVA p-value: {anova_result.pvalue}")
  return anova_result

def average_groups_by_plate_v0(df, x_value, y_value, replicates):
    '''
    Group the DataFrame by the specified columns and calculate the mean of the y_value column.
    Returns the averaged dataframe for plotting
    '''
    df = df.dropna(subset=[x_value, y_value, replicates])
    #df.reset_index(drop=True, inplace=True) - don't need this?
    
    group_averages = df.groupby([x_value, replicates], as_index=False, observed=True).agg({y_value: "mean"})
    
    # Reset the index to get a clean DataFrame
    average_df = group_averages.reset_index()
   
    return average_df


def average_groups_pivot(df, x_value, y_value, replicates):
    group_ave_pivot = df.pivot_table(columns=x_value, values=y_value, index=replicates)
    return group_ave_pivot
    