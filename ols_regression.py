import numpy as np
import statsmodels.formula.api as smf
import pandas as pd


def ols_regression(df, results, extended_results=False):
    
    features = []
    rejected = []
    
    #making sure we are working with numbers
    for i in df:
        if df[i].dtype == float:
            rejected.append(i)
        elif df[i].dtype == int:
            features.append(i)
        else:
            rejected.append(i)
            
    if extended_results==True:
        all_results=['result_binary', 'result_raw', 'result_scaler', 
                     'result_L2_norm', 'result_minmaxscaler', 'result_dummy']
    elif extended_results==False:
        all_results=results

    colNames=('Feature', 'Result', 'R-squared', 'Adj. R-squared', 'F-statistic', 
              'Log-Likelihood', 'Akaike', 'Bayesian')
    
    results_df = pd.DataFrame(columns=colNames)

    for i in features:
        for e in all_results:
            
            rsquared=[]
            rsquared_adj=[]
            fvalue=[]
            Log_Likelihood=[]
            aic=[]
            bic=[]   
        
            y = np.asarray(df[e])
            
            x = np.asarray(df[i])

            OLS_Regression = smf.OLS(y,x).fit()

            rsquared.append(format(OLS_Regression.rsquared, '.2f'))
            rsquared_adj.append(format(OLS_Regression.rsquared_adj, '.2f'))
            fvalue.append(format(OLS_Regression.fvalue, '.2f'))
            Log_Likelihood.append(format(OLS_Regression.llf, '.2f'))
            aic.append(format(OLS_Regression.aic, '.2f'))
            bic.append(format(OLS_Regression.bic, '.2f'))
            
            series = [[i, e, rsquared, rsquared_adj, 
                               fvalue, Log_Likelihood,aic, bic]]
            
            temp_df = pd.DataFrame(data=series, columns = colNames)
            
            results_df = results_df.append(temp_df, ignore_index=True)
            
    return results_df.sort_values(['R-squared'], ascending=False)