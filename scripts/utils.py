from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd

def balance_dataset_with_smote(X, y):
    # Initialize the SMOTE object
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    
    # Apply SMOTE to the dataset
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    return X_resampled, y_resampled

## sample code to run the balance dataset using smote
# print(X_train.shape,y_train.shape)
# get_numpy_value_counts(y_train)
# X_train, y_train = balance_dataset_with_smote(X_train, y_train)
# print(X_train.shape,y_train.shape)
# get_numpy_value_counts(y_train)

def get_numpy_value_counts(arr):
    unique, counts = np.unique(arr, return_counts=True)
    # print(np.asarray((unique, counts)).T)
    return np.asarray((unique, counts)).T


# check distribution of values
def check_values_distributino(arr):
    """
    arr must be numpy array and have shape (num_samples,num_features), e.g. (8000,4)
    """
    for j in range(arr.shape[1]):
        print(min(arr[:,j]),max(arr[:,j]))
        plt.hist(arr[:, j], bins=70, edgecolor='k', alpha=0.7)
        plt.show()


# functions to add target column to df

def categorize_by_percentile(values,percentile_list,labels):
    # Compute percentiles
    percentiles = np.percentile(values, percentile_list)
    # Use pd.cut to bin the values
    categories = pd.cut(values, bins=[-np.inf] + percentiles.tolist() + [np.inf], labels=labels, include_lowest=True)
    
    return categories

def add_target_cols(df,percentile_list,labels):
    new_df = pd.DataFrame()
    
    for stock_name in df.stock_name.unique():
        stock_df = df[df['stock_name']==stock_name]
        stock_df["Stock_Position"] = categorize_by_percentile(stock_df["Log_Return_shift"].to_numpy(),percentile_list,labels)        
        label_mapping = {labels[i]:i for i in range(len(labels))}
        stock_df["Target"] = stock_df["Stock_Position"].apply(lambda x:label_mapping[x])
        new_df = pd.concat([new_df,stock_df],ignore_index=True)
    
    return new_df

# # sample code to create target column
# # percentile_list = [20,40,60,80]
# percentile_list = [33,66]
# # labels = ['strong sell', 'sell', 'hold', 'buy', 'strong buy']
# labels = ['sell', 'hold', 'buy']
# df = add_target_cols(df,percentile_list,labels)