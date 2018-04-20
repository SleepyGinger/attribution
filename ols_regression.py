import numpy as np
import statsmodels.formula.api as smf
import pandas as pd


def regressions(df, results, extended_results=False):
    
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
    ols_df = pd.DataFrame(columns=colNames)

        
    def ols(features, all_results, ols_df):

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

                ols_df = ols_df.append(temp_df, ignore_index=True)
            
        return ols_df.sort_values(['R-squared'], ascending=False)
    
    linearcol=('Feature', 'Result', 'Coefficients', 'MSE', 'Variance')
    linear_df = pd.DataFrame(columns=linearcol)

    def linear(features, all_results, linear_df):
        for i in features:
            for e in all_results:
                
                coefficients = []
                mse = []
                variance = []

                y = np.asarray(df[e]).reshape(-1, 1)

                x = np.asarray(df[i]).reshape(-1, 1)

                na_features=missingvalues_check(df, no_nas=True)
                X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

                regr.fit(X_train, y_train)

                y_pred = regr.predict(X_test)

                coefficients.append(regr.coef_)
                mse.append(mean_squared_error(y_test, y_pred))
                variance.append(r2_score(y_test, y_pred))

                series = [[i, e, coefficients, mse, variance]]

                ltemp_df = pd.DataFrame(data=series, columns = linearcol)

                linear_df = linear_df.append(ltemp_df, ignore_index=True)
            
        return linear_df
    
    ltemp_df=linear(features, all_results, linear_df)
    
    return ltemp_df