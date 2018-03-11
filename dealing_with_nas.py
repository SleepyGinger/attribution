import numpy as np
import pandas as pd

#Add unknown values present in the variables
unknown_values = ['Unknown', 'nan','NaN','NA', 'None', '--None--', 'NaT'] 


def convert_all_nas(df):
    
    df=df.replace(unknown_values, np.nan)
    
    return df

#checks for null values and returns either missing_values list or no_missing_values list
def missingvalues_check(df, no_nas=False):
    
    has_missing_values = []
    no_missing_values = []
    
    for i in list(df.columns):
        if df[i].hasnans == True:
            has_missing_values.append(i)
        elif df[i].hasnans == False:
            no_missing_values.append(i)

    if no_nas==True:
        return no_missing_values
    else:
        return has_missing_values

#checks if percentaqge of null values is less that 20% and creates a list of these features.
def NAcalculator(df, has_Na): 
    
    percentage = []
    some_missing_values = []
    
    for i in has_Na:
        perc = "{0:.2f}".format((df[i].isnull().sum()) / float(len(df[i])) * 100)
        #print i + " " + perc +"%"
        percentage.append((i,perc))  
        if perc < 20:
            some_missing_values.append(i)
    
    df=pd.DataFrame(data=percentage, columns = ['feature_name', '%']).sort_values(['%'], ascending=False)       

    return df
