# standart imports
import pandas as pd
from datetime import timedelta, datetime
import numpy as np
import matplotlib.pyplot as plt




def prep_superstore_data(df):
    '''
    This function will clean the data by 
    converting the sale_date from object to datetime64
    and setting it as index. Also, will add three columns
    (dayofweek, month and sales_total). It also rearrange the
    columns for better reading. Finally, it will display two
    plot distributions for sale_amount and item_price. 
    '''
    
    # replace the hours, minutes, seconds and GMT with nothing
    df.sale_date = df.sale_date.str.replace(' 00:00:00 GMT', '')
    
    # convert the date from object to datetime
    df.sale_date = pd.to_datetime(df.sale_date, format = '%a, %d %b %Y')
    
    # This is another way to do it
    # change the date from object to datetime
    # df.sale_date = df.sale_date.astype('datetime64')
    
    # set sale_date as the index
    df = df.set_index('sale_date')
    
    # sort the index (sale_date)
    df = df.sort_index()
    
    # use dayofweek attribute and save
    df['dayofweek'] = df.index.day_name()

    # pull out weekday name & save
    df['month'] = df.index.month_name()
    
    # add a column named sales_total which is a derived from sale_amount (total items) and item_price
    df['sales_total'] = df.sale_amount * df.item_price
    
    # rearrange the columns
    df = df[['dayofweek', 'month', 'sale_amount', 'item_price', 'sales_total', 'sale_id',
             'item', 'item_id', 'item_name', 'item_brand', 'item_upc12','item_upc14',
             'store_id', 'store', 'store_address', 'store_city', 'store_state', 'store_zipcode']]
    
    # plot sale amount using a histgram graph
    plt.hist(df['sale_amount'])
    plt.title('Sale Amount')
    plt.ylabel('Frequency')
    plt.xlabel('Amount')
    plt.show()
    
    # plot item price using a histgram graph    
    plt.hist(df['item_price'])
    plt.title('Item Price')
    plt.ylabel('Frequency')
    plt.xlabel('Price')
    plt.show()
    return df

# ---------------------------------------------------------------------------------

def prep_ops_data(df):
    '''
    This function will clean and prep the open power system database.
    It first reset the index so that the date is back to a column.
    In the process all the names of the columns were set to lowercase,
    the date columns was converted from an object to datetime64. The date 
    column was then reset back to the index and sorted. Also, two new columns
    were added to the dataframe. Finally, all missing values were set to zero. 
    '''
    
    # reset the index
    df = df.reset_index()
    
    # convert all the column names to lowercase
    df.columns = df.columns.str.lower()
    
    # change the date from object to datetime and save it back to the date column
    df.date = df.date.astype('datetime64')
    
    # set sale_date as the index
    df = df.set_index('date')

    # sort the index (sale_date)
    df = df.sort_index()
    
    # pull out monthly name & save
    df['month'] = df.index.month_name()

    # use year attribute and save
    df['year'] = df.index.year
    
    # replace all the nulls with a zero
    df = df.replace(np.nan,0)
    
    # perform a plot distribution of all the features
    for col in df.columns:
        print(col)
        plt.hist(df[col])
        plt.show()
    return df