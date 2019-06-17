# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 22:34:00 2019

@author: tommy
"""

import pandas as pd
import numpy as np



#%%
df = pd.DataFrame({'data1': np.random.randint(low = -20, high = 50, 
                                              size = 10000),
                   'data2': np.random.randint(low = 20, high = 150, 
                                              size = 10000),
                   'data3': np.random.randint(low = 1000, high = 10000, 
                                              size = 10000),
                   'group1': np.random.randint(low = 1, high = 6, 
                                              size = 10000),
                   'group2': np.random.randint(low = 1, high = 10, 
                                              size = 10000),
                   'group3': np.random.randint(low = 1, high = 15, 
                                              size = 10000)
                                        })
                
df.iloc[0,0] = None
#%%
                   
def cap_values(df, cols):
    '''
    cap values to the 98th and 2nd percentiles
    
    Parameters
    ----------
    df: dataframe
        "tidy" data frame with rows as observations and columns as attributes 
    cols: list
        column names of the columns that should be capped
        
    Returns
    -------
    dataframe
        same input df but with additional cols for the capped values of the cols 
    '''
    
    # the min and max value for each one of the cols
    min_values = df.loc[:, cols].quantile(.02)  
    max_values = df.loc[:, cols].quantile(.98) 
    
    # create list of cols with the capped values
    capped_cols = []  
    i = 0
    for col in cols:
        capped_cols.append(np.select([df[col] < min_values[i], 
                                      df[col] > max_values[i]], 
                                     [min_values[i], max_values[i]], 
                                     default = df[col]))
        i = i + 1
    
    # create col names for capped values
    capped_col_names = [col + '_capped' for col in cols]
    
    # create df with the capped values. (list of arrays so have to zip and dict)
    df2 = pd.DataFrame(dict(zip(capped_col_names, capped_cols)))
    
    # add the cols with capped values to the input df
    df2 = pd.concat([df, df2], axis=1)
    
    return(df2, capped_col_names)


def grouped_standardize(df, group_cols, std_cols): 
    '''
    standarize columns with groupings
    
    Parameters
    ----------
    df: dataframe
        "tidy" data frame with rows as observations and columns as attributes 
    cols: list
        column names of the columns that should be standardized
        
    Returns
    -------
    dataframe
        same input df but with additional cols for the standardized values 
    '''
    
    z_vals = []
    
    # group standardize the cols 
    for col in std_cols:
        mu = df.groupby(group_cols)[col].transform('mean')
        sd = df.groupby(group_cols)[col].transform('std')
        z_vals.append((df[col] - mu) / sd)
    
    # create df of z values. (list of series so using concat)
    df2 = pd.concat(z_vals, axis = 1)
    
    # create list of col names and rename the cols in the df
    z_col_names = [col + '_z' for col in std_cols]
    df2.columns = z_col_names
    
    # add the cols with z values to the input df
    df2 = pd.concat([df, df2], axis=1)
    
    return(df2, z_col_names)

def cap_vals_3(df, cols): 
    '''
    limit values to -3 to 3
    
    Parameters
    ----------
    df: dataframe
        "tidy" data frame with rows as observations and columns as attributes 
    cols: list
        column names of the columns that should be capped 
        
    Returns
    -------
    dataframe
        same input df but with additional cols for the capped cols 
    '''
    
    # col names for the capped values
    cols_capped = [col + '_cap3' for col in cols]
    
    for col, col2 in zip(cols, cols_capped):
       df[col2] = np.select([df[col] < -3, df[col] > 3], [-3, 3,], 
                            default = df[col])
    
    return(df, cols_capped)
#%%

# cap values to the 2nd and 98th percentile
cols = ['data1', 'data2', 'data3']  
df2, cols = cap_values(df, cols)

# group standardize (z score) the capped values
#capped_cols = [col + '_capped' for col in cols]
group_cols = ['group1', 'group2']
df2, cols = grouped_standardize(df2, group_cols, cols)

# cap z scores to -+ 3
#capped_z = [col + '_z' for col in capped_cols]
df2, cols = cap_vals_3(df2, cols)

# group standardize the capped z scores with diff grouping
# capped_z_3 = [col + '_cap3' for col in capped_z]
group_cols2 = ['group1', 'group3']
df2, cols = grouped_standardize(df2, group_cols2, cols)

# cap the z scores to -+3
# capped_z_3_z = [col + '_z' for col in capped_z_3]
df2, cols = cap_vals_3(df2, cols)

#%%

df2['mod_score'] = .2 * df2.data1_capped_z_cap3_z_cap3 \
                    + .3 * df2.data2_capped_z_cap3_z_cap3






























