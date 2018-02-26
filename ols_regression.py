import numpy as np
import statsmodels.formula.api as smf


def ols_regression(df, results):
    features = list(df)
    rsquared=[]
    rsquared_adj=[]
    fvalue=[]
    Log_Likelihood=[]
    aic=[]
    bic=[]
    

    for i in features:
        X = df[i]
        Y = df[results]
        x = np.asarray(X)
        y = np.asarray(Y)

        OLS_Regression = smf.OLS(y,x).fit()
        rsquared.append(format(OLS_Regression.rsquared, '.2f'))
        rsquared_adj.append(format(OLS_Regression.rsquared_adj, '.2f'))
        fvalue.append(format(OLS_Regression.fvalue, '.2f'))
        Log_Likelihood.append(format(OLS_Regression.llf, '.2f'))
        aic.append(format(OLS_Regression.aic, '.2f'))
        bic.append(format(OLS_Regression.bic, '.2f'))
        
    return pd.DataFrame(zip(features, rsquared, rsquared_adj, fvalue, Log_Likelihood,aic, bic), 
                        columns=['Feature', 'R-squared', 'Adj. R-squared', 'F-statistic', 
                        'Log-Likelihood', 'Akaike', 'Bayesian']).sort_values(['R-squared'], ascending=False)