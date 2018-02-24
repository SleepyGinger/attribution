import pandas as pd
import numpy as np

from sklearn import preprocessing


def extend_results(og_df, results):
    results_array = og_df[results]
    
    def raw(results_array):
        raw = np.array(results_array)
        
        return raw
    
    def binary(results_array):
        binary=[]

        for x in results_array:
            if x == 0:
                binary.append(int(0))
            elif x > 0:
                binary.append(int(1))
            else:
                binary.append(results_array)
                
        return binary
                
    def minmaxscaler(results_array):
        min_max_scaler = preprocessing.MinMaxScaler()
        standardized = min_max_scaler.fit_transform(results_array)
        
        return standardized
    
    def L2_norm(results_array):
        norm = np.linalg.norm(results_array, axis=0, keepdims = True)
        L2_norm=results_array/norm
        
        return L2_norm
        
    def scaler(results_array):
        scale = preprocessing.scale(results_array)
        
        return scale
    
    def dummy(results_array):
        le = preprocessing.LabelEncoder()
        dum = pd.DataFrame(results_array).apply(le.fit_transform)
    
        return dum.values.T.tolist()

    new_results=np.concatenate([[binary(results_array), raw(results_array), scaler(results_array), L2_norm(results_array), minmaxscaler(results_array)], dummy(results_array)])
    new_df=np.transpose(pd.DataFrame(new_results))
    new_df.columns=['result_binary', 'result_raw', 'result_scaler', 'result_L2_norm', 'result_minmaxscaler', 'result_dummy']
    try:
        df=df.reset_index()
        extended_df=pd.concat([og_df, new_df], axis=1)
    except:
        extended_df=pd.concat([og_df, new_df], axis=1)

    return extended_df
