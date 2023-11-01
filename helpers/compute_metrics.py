# compute metrics from csv files 

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


csv_folder = '/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/SchwanommaDS/UKE/code/results/ISBI_25D/'

csv_files = os.listdir(csv_folder)

# remove files that are not csv
csv_files = [f for f in csv_files if f.endswith('.csv')]
csv_files.sort()

# create a single  master dataframe 
master_df = pd.DataFrame()

for file in csv_files:


    df = pd.read_csv(csv_folder + file)

    print(df.columns,"here")
    
    df = df[['Group', 'dice_Schwannoma', 'hausdorff_Schwannoma','assd','rve']]
    
    print(df.columns)

    # add to master dataframe
    master_df = pd.concat([master_df, df], axis=0)

# reset index
master_df.reset_index(inplace=True, drop=True)

# compute mean and std of dice_SchwannomaCanal column by grouping by "Group"
# mean and std of dice

dice_df = master_df.groupby(['Group']).mean().reset_index()
dice_df['std'] = master_df.groupby(['Group']).std().reset_index()['dice_Schwannoma']

# just keep mean and std
#dice_df = dice_df[['Group', 'dice_SchwannomaCanal', 'std']] 

# compute hausdorff distance by grouping by "Group"
# mean and std of dice
hd_df = master_df.groupby(['Group']).mean().reset_index()
hd_df['std'] = master_df.groupby(['Group']).std().reset_index()['hausdorff_Schwannoma']


# compute assd by grouping by "Group"
# mean and std of dice
assd_df = master_df.groupby(['Group']).mean().reset_index()
assd_df['std'] = master_df.groupby(['Group']).std().reset_index()['assd']

# compute rve by grouping by "Group"
# mean and std of dice
rve_df = master_df.groupby(['Group']).mean().reset_index()
rve_df['std'] = master_df.groupby(['Group']).std().reset_index()['rve']


# just keep mean and std
#hd_df = hd_df[['Group', 'hausdorff_SchwannomaCanal', 'std']]


# save to csv
# only keep mean and std of dice
dice_df = dice_df[['Group', 'dice_Schwannoma', 'std']]
dice_df.to_csv(csv_folder + 'dice.csv', index=False)

# only keep mean and std of hausdorff
hd_df = hd_df[['Group', 'hausdorff_Schwannoma', 'std']]
hd_df.to_csv(csv_folder + 'hausdorff.csv', index=False)

# only keep mean and std of assd
assd_df = assd_df[['Group', 'assd', 'std']]
assd_df.to_csv(csv_folder + 'assd.csv', index=False)

# only keep mean and std of rve
rve_df = rve_df[['Group', 'rve', 'std']]
rve_df.to_csv(csv_folder + 'rve.csv', index=False)


"""

# print unique groups
print(master_df['Group'].unique())

# remove _dataset_DS1 , _dataset_DS2 , _dataset_DS3 , _dataset_DS4 , _dataset_DS5 from Group
master_df['Group'] = master_df['Group'].str.replace('_dataset_DS1', '')
master_df['Group'] = master_df['Group'].str.replace('_dataset_DS2', '')
master_df['Group'] = master_df['Group'].str.replace('_dataset_DS3', '')
master_df['Group'] = master_df['Group'].str.replace('_dataset_DS4', '')
master_df['Group'] = master_df['Group'].str.replace('_dataset_DS5', '')


# compute dice by grouping by "Group"
# mean and std of dice
dice_df = master_df.groupby(['Group']).mean().reset_index()
dice_df['std'] = master_df.groupby(['Group']).std().reset_index()['dice_SchwannomaCanal']

# only keep mean and std
dice_df = dice_df[['Group', 'dice_SchwannomaCanal', 'std']]

# compute hausdorff distance by grouping by "Group"
# mean and std of dice
hd_df = master_df.groupby(['Group']).mean().reset_index()
hd_df['std'] = master_df.groupby(['Group']).std().reset_index()['hausdorff_SchwannomaCanal']

# only keep mean and std
hd_df = hd_df[['Group', 'hausdorff_SchwannomaCanal', 'std']]

# save to csv
dice_df.to_csv(csv_folder + 'dice_no_dataset.csv', index=False)
hd_df.to_csv(csv_folder + 'hausdorff_no_dataset.csv', index=False)
"""






