import os 
import pandas as pd 
import numpy as np
import random
import re
ds_folder = '/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/SchwanommaDS/VS-Seg-cropped_all_slices_with_tumor/'
save_csv_folder = '/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/SchwanommaDS/UKE/labelled_ds/csv_public_ds_100x100_2D_all_slices/'

# get all folders in the dataset folder
folders = os.listdir(ds_folder)


def extract_number(filename):
    match = re.search(r'(\d+)\.png', filename)
    if match:
        return int(match.group(1))
    return -1

# create a dataframe by dividing the dataset into folds such that each fold has 20% of the total number of folders 5 times

np.random.seed(42)


for ds_number in range(5):

    folders = np.array(folders)
    np.random.shuffle(folders)

    # take the first 20% of the folders as test set

    test_folders = folders[:int(0.2*len(folders))] 

    # take the remaining 80% of the folders as train and val set with 5 folds

    train_val_folders = folders[int(0.2*len(folders)):]
    split_folders = np.array_split(train_val_folders, 5)

    # create a dataframe with 3 columns: train, val, test

    for i in range(5):

        val_folder = list(split_folders[i])
        train_folder = list(set(train_val_folders) - set(val_folder)) 

        # create train df with 2 columns: T1, T2, mask
        train_df = pd.DataFrame(columns=['T1', 'T2', 'mask'])
        for folder in train_folder:

   

            mask_files = os.listdir(os.path.join(ds_folder, folder, 'mask'))
            t1_files = os.listdir(os.path.join(ds_folder, folder, 't1'))
            t2_files = os.listdir(os.path.join(ds_folder, folder, 't2'))

                      
            # sort the files
            mask_files = sorted(mask_files, key=extract_number)
            t1_files = sorted(t1_files, key=extract_number)
            t2_files = sorted(t2_files, key=extract_number)

           

            # create path to the files
            mask_files = [os.path.join(ds_folder, folder, 'mask', f) for f in mask_files]
            t1_files = [os.path.join(ds_folder, folder, 't1', f) for f in t1_files]
            t2_files = [os.path.join(ds_folder, folder, 't2', f) for f in t2_files]


            # create a dictionary with 3 keys: T1, T2, mask
            data = {'T1': t1_files, 'T2': t2_files, 'mask': mask_files}

            # create a dataframe from the dictionary
            df = pd.DataFrame(data)

            # add the dataframe to the train_df
            train_df = pd.concat([train_df, df], axis=0)

        # reset index
        train_df.reset_index(inplace=True, drop=True)

        # create val df with 2 columns: T1, T2, mask

        val_df = pd.DataFrame(columns=['T1', 'T2', 'mask'])

        for folder in val_folder:



            mask_files = os.listdir(os.path.join(ds_folder, folder, 'mask'))
            t1_files = os.listdir(os.path.join(ds_folder, folder, 't1'))
            t2_files = os.listdir(os.path.join(ds_folder, folder, 't2'))


            # sort the files
            mask_files = sorted(mask_files, key=extract_number)
            t1_files = sorted(t1_files, key=extract_number)
            t2_files = sorted(t2_files, key=extract_number)
            

            # create path to the files
            mask_files = [os.path.join(ds_folder, folder, 'mask', f) for f in mask_files]
            t1_files = [os.path.join(ds_folder, folder, 't1', f) for f in t1_files]
            t2_files = [os.path.join(ds_folder, folder, 't2', f) for f in t2_files]
            
            

            # create a dictionary with 3 keys: T1, T2, mask
            data = {'T1': t1_files, 'T2': t2_files, 'mask': mask_files}

            # create a dataframe from the dictionary
            df = pd.DataFrame(data)


            # add the dataframe to the val_df
            val_df = pd.concat([val_df, df], axis=0)

        # reset index
        val_df.reset_index(inplace=True, drop=True)

        # create test df with 2 columns: T1, T2, mask

        test_df = pd.DataFrame(columns=['T1', 'T2', 'mask'])

        for folder in test_folders:



            mask_files = os.listdir(os.path.join(ds_folder, folder, 'mask'))
            t1_files = os.listdir(os.path.join(ds_folder, folder, 't1'))
            t2_files = os.listdir(os.path.join(ds_folder, folder, 't2'))


            # sort the files
            mask_files = sorted(mask_files, key=extract_number)
            t1_files = sorted(t1_files, key=extract_number)
            t2_files = sorted(t2_files, key=extract_number)
            

            # create path to the files
            mask_files = [os.path.join(ds_folder, folder, 'mask', f) for f in mask_files]
            t1_files = [os.path.join(ds_folder, folder, 't1', f) for f in t1_files]
            t2_files = [os.path.join(ds_folder, folder, 't2', f) for f in t2_files]
            
            

            # create a dictionary with 3 keys: T1, T2, mask
            data = {'T1': t1_files, 'T2': t2_files, 'mask': mask_files}

            # create a dataframe from the dictionary
            df = pd.DataFrame(data)


            # add the dataframe to the test_df
            test_df = pd.concat([test_df, df], axis=0)

        # reset index
        test_df.reset_index(inplace=True, drop=True)


        # create directory if it doesn't exist
        os.makedirs(save_csv_folder + 'DS' + str(ds_number+1), exist_ok=True)

        # save the dataframes as csv files

        train_df.to_csv(save_csv_folder + 'DS' + str(ds_number+1) +  '/train_set_' + str(i) + '.csv', index=False)

        val_df.to_csv(save_csv_folder + 'DS' + str(ds_number+1) +  '/val_set_' + str(i) + '.csv', index=False)

        test_df.to_csv(save_csv_folder + 'DS' + str(ds_number+1) +  '/test_set_' + str(i) + '.csv', index=False)

        # store meta data of number of folders in each set
        meta_data = {'train': len(train_folder), 'val': len(val_folder), 'test': len(test_folders)}
        meta_data = pd.DataFrame(meta_data, index=[0])
        meta_data.to_csv(save_csv_folder + 'DS' + str(ds_number+1) +  '/meta_data_' + str(i) + '.csv', index=False)
        







            

            
            
        
        