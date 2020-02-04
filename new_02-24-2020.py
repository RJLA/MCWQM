from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
import os
import pandas as pd
import joblib
from sklearn.model_selection import cross_val_score

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression


def cross_validate(X,
                   y,
                   key,
                   model,
                   name,
                   ws,
                   folds = 5):

    pipeline = Pipeline([('sc', 
                  MinMaxScaler()),
                ('%s'%key,
                 model)])

    kf = KFold(n_splits = folds, 
               shuffle = True, 
               random_state = None) 
    
    scores_fold = cross_val_score(pipeline, 
                                  X, 
                                  y, 
                                  cv = kf,
                                 scoring = 'r2')

    pipeline.fit(X, y)

    base_model_path = os.path.join(ws,
                                'tuned_models')

    if not os.path.exists(base_model_path):
        os.makedirs(base_model_path)
        
    joblib.dump(pipeline, 
                os.path.join(base_model_path,
                             '%s.sav'%name))
    
    mean_acc = sum(scores_fold) / len(scores_fold) 
    
    print('Scores per fold %s' %scores_fold)
    print('Mean accuracy: %.2f' %mean_acc)

def train_level_1(sub_level_n):
    
    tuned_models_dir = os.path.join(os.getcwd(),
                        'tuned_models')

    key_list = ['lgb',
    'xgb',
    'cat',
    'rf',
    'dt',
    'svr',
    'knn',
    'lss',
    'enet',
    'rdg',
    'lr']

    model_list = [
        LGBMRegressor(n_estimators = 1000),
        XGBRegressor(n_estimators = 300,
                                n_jobs = -1,
                                objective='reg:squarederror'),
        CatBoostRegressor(),                                
        RandomForestRegressor(n_estimators = 500,
                                      n_jobs = -1),
        DecisionTreeRegressor(),
        SVR(), 
        KNeighborsRegressor(n_jobs = -1),
        Lasso(max_iter = 20000),
        ElasticNet(max_iter = 20000),
        Ridge(),
        LinearRegression()]
    
    
    ###################### sub level ######################

    level_1_predictions = [] 
    
    for sub_level in range(sub_level_n):
    

        for key, model in zip(key_list,
                             model_list):
            print()
            print('Sub level %s' %sub_level)
            print(key)
            mean_acc = cross_validate(X_train,
                          y_train,
                          key,
                          model,
                          '%s_sub%s_level_1'%(key,
                                            sub_level),
                          ws = os.getcwd(),
                          folds = 10)

            model_trained = joblib.load(os.path.join(tuned_models_dir, 
                                                     '%s_sub%s_level_1.sav'%(key,
                                                                       sub_level)))

            prediction_level_1 = model_trained.predict(X_train)
            level_1_features.append(prediction_level_1)

    df_level_1 = pd.DataFrame(level_1_features).T  
    return df_level_1, y_train

df_main = pd.read_csv('all_data_raw.csv')

X = df_main.iloc[:,2:]

X['G_over_R'] = X['green'] / X['red'] #G/R
X['G_over_B'] = X['green'] / X['blue'] #G/B
X['B_over_R'] = X['blue'] / X['red'] #B/R

X['B_over_NIR_plus_G'] = (X['blue'] / X['re_1']) + X['green'] #B/NIR+G 

X['SABI'] = (X['re_1'] - X['red']) / (X['blue'] + X['green'])

X['kab1'] = (1.67-3.94)*np.log(X['blue'])+3.78*np.log(X['green'])
X['NDWI'] = (X['green'] - X['re_1']) / (X['green'] + X['re_1']) 
X['MNDWI'] = (X['green'] - X['swir_2']) / (X['green'] + X['swir_2'])
X['NDMI'] = (X['red'] - X['re_1']) / (X['red'] + X['re_1'])

#new features
X['NDCI'] = (X['re_1'] - X['red']) / (X['re_1'] + X['red']) 
X['CHL1'] = X['re_2'] / (X['re_1'] - X['red'])


# # 80% of data used for training, 20% for test
# df_train = df_main.sample(frac = 0.8, 
#                             random_state = None) 
# # # get samples not in training to be used for evaluating the model
# index_not_train = [idx for idx in df_main.index if idx not in df_train.index]
# df_test = df_main.iloc[index_not_train]

# ransac = RANSACRegressor(base_estimator = LinearRegression(), 
#                          max_trials = 100,
#                         min_samples = X.shape[0])

# X_scaled = MinMaxScaler().fit_transform(X) 
# ransac.fit(X_scaled,
#            y_chla)
# inlier = ransac.inlier_mask_
# outliers = np.logical_not(inlier)

# X_inlier = X[inlier]
# y_inlier = y_chla[inlier]
# X_outlier = X[outliers]
# y_outlier = y_chla[outliers]

# df_inlier = pd.concat([X_inlier,
#                       y_inlier],
#                      axis = 1)

# df_inlier.reset_index(inplace = True)

# del df_inlier['index']
# print(df_inlier.shape)

# # #features and targets for training and test data
# X_train = df_train.iloc[:,:-1] 
# y_train = df_train.iloc[:,-1]

# X_test = df_test.iloc[:,:-1] 
# y_test = df_test.iloc[:,-1]