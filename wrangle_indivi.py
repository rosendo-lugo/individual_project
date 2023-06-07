import pandas as pd
import numpy as np
from env import get_db_url
import os
import opendatasets as od

import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------------
def check_file_exists(fn):
    """
    This function will:
    - Kaggle dataset
    - check if file exists in my local directory, if not, pull from url
    - read the given `url`
    - return dataframe
    """
    cwd = os.getcwd()
    folder = "/american-companies-bankruptcy-prediction-dataset"

    if os.path.exists(cwd + folder):
        os.chdir(cwd+folder)
        if os.path.isfile(fn):
            print('csv file found and loaded')
            return pd.read_csv(fn, index_col=0)
        else:
            print("Not same file name.")   
    else:
        print('creating df and exporting csv')
        od.download("https://www.kaggle.com/datasets/utkarshx27/american-companies-bankruptcy-prediction-dataset")
        os.chdir(cwd+folder)
        df = pd.read_csv(fn)

        return df     
# ----------------------------------------------------------------------------------
def get_data():
    # How to import a database from URL
    # url = "https://www.kaggle.com/datasets/utkarshx27/american-companies-bankruptcy-prediction-dataset#:~:text=calendar_view_week-,american_bankruptcy,-.csv"
    
    url = "kaggle datasets download -d utkarshx27/american-companies-bankruptcy-prediction-dataset"

    filename = 'american_bankruptcy.csv'
    df = check_file_exists(filename, url)    
    
    # # rename columns
    # df.columns
    # df = df.rename(columns={'bedroomcnt':'bedrooms', 'bathroomcnt':'bathrooms',
    #                         'calculatedfinishedsquarefeet':'area','taxvaluedollarcnt':'property_value',
    #                         'fips':'county','transaction_0':'transaction_year',
    #                         'transaction_1':'transaction_month','transaction_2':'transaction_day'})
    
    # drop any nulls in the dataset
    df = df.dropna()    
    
    # sets all the column names to lowercase
    df.columns = df.columns.str.lower()
    
    return df 


# ----------------------------------------------------------------------------------
def get_zillow_data():
    # How to import a database from MySQL
    url = get_db_url('zillow')
    query = '''
    select *
    from properties_2017 p
        join predictions_2017 p2 using (parcelid)
    where p.propertylandusetypeid = 261 and 279
            '''
    filename = 'zillow.csv'
    df = check_file_exists(filename, query, url)
    
    # filter to just 2017 transactions
    df = df[df['transactiondate'].str.startswith("2017", na=False)]
    
    # split transaction date to year, month, and day
    df_split = df['transactiondate'].str.split(pat='-', expand=True).add_prefix('transaction_')
    df = pd.concat([df.iloc[:, :40], df_split, df.iloc[:, 40:]], axis=1)
    
    # Drop duplicate rows in column: 'parcelid', keeping max transaction date
    df = df.drop_duplicates(subset=['parcelid'])
    
    # rename columns
    df.columns
    df = df.rename(columns={'bedroomcnt':'bedrooms', 'bathroomcnt':'bathrooms',
                            'calculatedfinishedsquarefeet':'area','taxvaluedollarcnt':'property_value',
                            'fips':'county','transaction_0':'transaction_year',
                            'transaction_1':'transaction_month','transaction_2':'transaction_day'})
    
    # total outliers removed are 6029 out of 52442
    # # Look at properties less than 1.5 and over 5.5 bedrooms (Outliers were removed)
    # df = df[~(df['bedrooms'] < 1.5) & ~(df['bedrooms'] > 5.5)]

    # Look at properties less than .5 and over 4.5 bathrooms (Outliers were removed)
    df = df[~(df['bathrooms'] < .5) & ~(df['bathrooms'] > 4.5)]

    # Look at properties less than 1906.5 and over 2022.5 years (Outliers were removed)
    df = df[~(df['yearbuilt'] < 1906.5) & ~(df['yearbuilt'] > 2022.5)]

    # Look at properties less than -289.0 and over 3863.0 area (Outliers were removed)
    df = df[~(df['area'] < -289.0) & ~(df['area'] > 3863.0)]

    # Look at properties less than -444576.5 and over 1257627.5 property value (Outliers were removed)
    df = df[~(df['property_value'] < -444576.5) &  ~(df['property_value'] > 1257627.5)]
    
    # replace missing values with "0"
    df = df.fillna({'bedrooms':0,'bathrooms':0,'area':0,'property_value':0,'county':0})
    
    # drop any nulls in the dataset
    df = df.dropna()
    
    # drop all duplicates
    df = df.drop_duplicates(subset=['parcelid'])
    
    # change the dtype from float to int  
    df[['bedrooms','area','property_value','yearbuilt','transaction_month','transaction_day']] = df[['bedrooms','area','property_value','yearbuilt','transaction_month','transaction_day']].astype(int)
    
    # rename the county codes inside county
    df['county'] = df['county'].map({6037: 'LA', 6059: 'Orange', 6111: 'Ventura'})
    
    # get dummies and concat to the dataframe
    dummy_tips = pd.get_dummies(df[['county']], dummy_na=False, drop_first=[True, True])
    df = pd.concat([df, dummy_tips], axis=1)
    
    # dropping these columns for right now until I find a use for them
    df = df.drop(columns =['parcelid','transactiondate','transaction_year','transaction_month','transaction_day'])
    
    # Define the desired column order
    new_column_order = ['bedrooms','bathrooms','area','yearbuilt','county','county_Orange','county_Ventura','property_value',]

    # Reindex the DataFrame with the new column order
    df = df.reindex(columns=new_column_order)

    # write the results to a CSV file
    df.to_csv('df_prep.csv', index=False)

    # read the CSV file into a Pandas dataframe
    prep_df = pd.read_csv('df_prep.csv')
    
    return df.set_index('customer_id'), prep_df