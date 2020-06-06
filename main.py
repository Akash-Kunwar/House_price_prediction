import pandas as pd
import matplotlib.pyplot  as plt
import numpy as np
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit 
from pandas.plotting import scatter_matrix
from sklearn.impute  import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

housing =pd.read_csv('data.csv')

#Data Set Description
# print(housing.head())
# print(housing.info())
# print(housing.describe())
# print(housing['CHAS'].value_counts())

#Histogram
# housing.hist(bins=30,figsize=(15,10))
# plt.show()

#Train test spliting
# def split_tain_test(data,test_ratio):
#     np.random.seed(42)
#     shuffled =np.random.permutation(len(data))
#     test_size=int(len(data)*test_ratio)
#     test_indices=shuffled[:test_size]
#     train_inices=shuffled[test_size:]
#     return data.iloc[train_inices],data.iloc[test_indices]
# train_sets,test_sets=split_tain_test(housing,0.2)

#using Sklearn
# train_set,test_set=train_test_split(housing,test_size=0.2,random_state=42)
# print(len(train_set),len(test_set))


#Distributing Equaly using Sklearn.StratifiedShuffleSplit
split=StratifiedShuffleSplit (n_splits=1,test_size=0.2,random_state=42)
for train_index,test_index in split.split(housing,housing['CHAS']):
    s_test_set=housing.loc[test_index];
    s_train_set=housing.loc[train_index];
# print(s_test_set.describe())

# take only training data set for further processing
housing=s_train_set.copy()
housing=s_train_set.drop("MEDV",axis=1)
housing_labels=s_train_set["MEDV"].copy()

#To take care of missing attributes
    # 1-Get rid of missing data points
    # 2-Get rid of thw whole attribute
    # 3-Set the value as (0,mean,mode,median) 

# housing.dropna(subset=["RM "]).shape() #option 1
# housing.drop("RM ",axis=1) #option 2

median=housing["RM "].median() #option 3
housing.fillna(median)
imputer=SimpleImputer(strategy="median")
imputer.fit(housing)

X=imputer.transform(housing)

housing_tr=pd.DataFrame(X,columns=housing.columns)

# print(housing_tr.info())





#Correlation Matrix
# corr_matrix=housing.corr()
# print(corr_matrix['MEDV'].sort_values(ascending=False));
# attr=["MEDV","RM ","TAXRM"]
# scatter_matrix(housing[attr],figsize=(12,8))
# plt.show()

#Feature Scaling
# Primarily, Two types of methods are there
# 1. Min-Max(Normalisation)
    # (value-min)/(max-min)
    # Sklearn provides class MinMaxScaler()
# 2. Standardisation
    # (value-mean)/standar_deviation
    # Sklearn provides class StandardScaler()



# Creating Pipeline
my_pipe=Pipeline([
    ('imputer',SimpleImputer(strategy="median")),
    # ('std_scaler',StandardScaler()),
])

housing_num_tr=my_pipe.fit_transform(housing_tr)
# print(housing_num_tr)



#Model Selection
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# model=LinearRegression()
# model=DecisionTreeRegressor()
model=RandomForestRegressor()

model.fit(housing_num_tr,housing_labels)



some_data=housing.iloc[:5]
some_labels=housing_labels.iloc[:5]
prepared_data=my_pipe.transform(some_data)

# print(model.predict(prepared_data))
# print(list(some_labels))



#EVALUATING THE MODEL

from sklearn.metrics import mean_squared_error 
housing_predictions=model.predict(housing_num_tr)
mse=mean_squared_error(housing_labels,housing_predictions)
rmse=np.sqrt(mse)
# print(rmse)


# Using better evaluation technique -Cross validation
from sklearn.model_selection  import cross_val_score

scores=cross_val_score(model,housing_num_tr,housing_labels,scoring="neg_mean_squared_error",cv=10)

rmse_scores=np.sqrt(-scores)

# print(rmse_scores)

def print_scores(scores):
    print('scores',scores)
    print('Mean',scores.mean())
    print('standard deviation',scores.std())

# print_scores(rmse_scores)


#SAVING THE MODEL

from joblib import dump, load
dump(model, 'model.pkl')  

#Testing the data
X_test=s_test_set.drop("MEDV",axis=1)
Y_test=s_test_set["MEDV"].copy()
X_test_prepare=my_pipe.transform(X_test)
final_predictions=model.predict(X_test_prepare)
final_mse = mean_squared_error(Y_test,final_predictions)
final_rmse=np.sqrt(final_mse)
print(prepared_data[0])