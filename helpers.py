def average_groups_by_plate(df, x_value, y_value, replicates):
    # Group by the specified columns and calculate the mean of the y_value column
    group_averages = df.groupby([x_value, replicates], as_index=False).agg({y_value: "mean"})
    
    # Reset the index to get a clean DataFrame
    average_df = group_averages.reset_index(drop=True)
   
    return average_df