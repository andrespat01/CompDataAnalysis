import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import Lasso, Ridge, ElasticNet, Lars, BayesianRidge # , randomforrest, lars, iterativeimputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.feature_selection import SelectFromModel

def transform_to_categorical_bool(inp_array):
    N = inp_array.shape[0]
    for i in range(N):
        for j in range(4):
            idx = np.argmax(inp_array[i, 95+(j*5) : 95+(j+1)*5 ])
            inp_array[i, 95+(j*5) : 95+(j+1)*5 ] = 0,0,0,0,0
            inp_array[i, 95+(j*5) + idx] = 1
    return inp_array

def custom_scale(scaler, data_X, data_Xnew):
    """ 
        Scale both X and Xnew based only on their continous values.
    """
    scaler.fit(np.concatenate((data_X[:,:95], data_Xnew[:,:95]), axis = 0))
    X_con, Xnew_con = scaler.transform(data_X[:,:95]), scaler.transform(data_Xnew[:,:95])
    data_X_norm = np.concatenate((X_con, data_X[:,95:]), axis = 1)
    data_Xnew_norm = np.concatenate((Xnew_con, data_Xnew[:,95:]), axis = 1)
    all_data = np.concatenate((data_X_norm, data_Xnew_norm), axis = 0)

    return scaler, data_X_norm, all_data

def custom_transform(scaler, data):
    return np.concatenate((scaler.transform(data[:,:95]), data[:,95:]), axis = 1)

if __name__ == "__main__":

    #load data from txt file with pandas
    data = pd.read_csv('case1Data.txt', sep=", ", engine='python')
    y = data['y']

    X = pd.read_csv('case1Data_one_hot.csv').to_numpy()
    X_new = pd.read_csv('case1Data_Xnew_one_hot.csv').to_numpy()
    y_vec = y.values

    K_outer = 10
    K_inner = 5
    cv_outer = KFold(n_splits=K_outer, shuffle=True, random_state=42)
    cv_inner = KFold(n_splits=K_inner, shuffle=True, random_state=42)

    #Model parameters
    alphas_elastic = np.linspace(0.1, 15, 30)
    l1_ratios_elastic = np.linspace(0.1, 1, 20)
    nonzero_coefs_lars = range(16, 36, 2)
    alphas_lasso_feat_select = np.linspace(0.1, 3, 10)
    n_estimators_rf = range(10, 100, 10)

    #Imputation parameters
    imputer_estimators = [(BayesianRidge(), 100), (BayesianRidge(), 115)]

    RMSE = {'ElasticNet': np.zeros(K_outer), 'Lars': np.zeros(K_outer), 
            'RandomForest': np.zeros(K_outer), 'RandomForest_feat_select': np.zeros(K_outer)}
    for k1, (train_index, test_index) in enumerate(cv_outer.split(X)):
        print('Outer fold:', k1+1, '/', K_outer)
        Xtrain, Xtest = X[train_index], X[test_index]
        ytrain, ytest = y_vec[train_index], y_vec[test_index]

        errors_elastic = np.zeros((K_inner, len(imputer_estimators), len(alphas_elastic), len(l1_ratios_elastic)))
        errors_lars = np.zeros((K_inner, len(imputer_estimators), len(nonzero_coefs_lars)))
        errors_rf = np.zeros((K_inner, len(imputer_estimators), len(n_estimators_rf)))
        errors_rf_feat_select = np.zeros((K_inner, len(imputer_estimators), len(n_estimators_rf), len(alphas_lasso_feat_select)))
        for k2, (train_index_inner, test_index_inner) in enumerate(cv_inner.split(Xtrain)):
            print('Inner fold:', k2+1, '/', K_inner)
            Xtrain_inner, Xtest_inner = Xtrain[train_index_inner], Xtrain[test_index_inner]
            ytrain_inner, ytest_inner = ytrain[train_index_inner], ytrain[test_index_inner]
            scaler, Xtrain_inner_norm, all_inner_data = custom_scale(StandardScaler(), Xtrain_inner, X_new)
            Xtest_inner_norm = custom_transform(scaler, Xtest_inner)
            
            for i, (imputer_estimator, n_nearest) in enumerate(imputer_estimators):
                    
                imputer = IterativeImputer(estimator=imputer_estimator, n_nearest_features= n_nearest, random_state=42, max_iter = 100).fit(all_inner_data)
                #Impute data
                Xtrain_inner_imputed = transform_to_categorical_bool(imputer.transform(Xtrain_inner_norm))
                Xtest_inner_imputed = transform_to_categorical_bool(imputer.transform(Xtest_inner_norm))

                #ElasticNet
                for j, alpha in enumerate(alphas_elastic):
                    for l, l1_ratio in enumerate(l1_ratios_elastic):
                        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000).fit(Xtrain_inner_imputed, ytrain_inner)
                        preds_inner_elastic = model.predict(Xtest_inner_imputed)
                        errors_elastic[k2, i, j, l] = np.sum((preds_inner_elastic - ytest_inner)**2)
                #Lars
                for j, nzc in enumerate(nonzero_coefs_lars):
                    model = Lars(n_nonzero_coefs=nzc).fit(Xtrain_inner_imputed, ytrain_inner)
                    preds_inner_lars = model.predict(Xtest_inner_imputed)
                    errors_lars[k2, i, j] = np.sum((preds_inner_lars - ytest_inner)**2)
                #RandomForest_feat_select
                for j, n_est in enumerate(n_estimators_rf):
                    for l, alpha in enumerate(alphas_lasso_feat_select):
                        feat_select = Lasso(alpha=alpha).fit(Xtrain_inner_imputed, ytrain_inner)
                        model_feat_select = SelectFromModel(feat_select, prefit=True)
                        Xtrain_inner_imputed_feat_select = model_feat_select.transform(Xtrain_inner_imputed)
                        Xtest_inner_imputed_feat_select = model_feat_select.transform(Xtest_inner_imputed)
                        model = RandomForestRegressor(n_estimators=n_est).fit(Xtrain_inner_imputed_feat_select, ytrain_inner)
                        preds_inner_rf_feat_select = model.predict(Xtest_inner_imputed_feat_select)
                        errors_rf_feat_select[k2, i, j, l] = np.sum((preds_inner_rf_feat_select - ytest_inner)**2)

                #RandomForest withput feature selection
                for j, n_est in enumerate(n_estimators_rf):
                    model = RandomForestRegressor(n_estimators=n_est).fit(Xtrain_inner_imputed, ytrain_inner)
                    preds_inner_rf = model.predict(Xtest_inner_imputed)
                    errors_rf[k2, i, j] = np.sum((preds_inner_rf - ytest_inner)**2)

        scaler, Xtrain_norm, all_data = custom_scale(StandardScaler(), Xtrain, X_new)
        Xtest_norm = custom_transform(scaler, Xtest)

        model_dict_errors = {'ElasticNet': errors_elastic, 'Lars': errors_lars, 
                             'RandomForest_feat_select': errors_rf_feat_select, 'RandomForest': errors_rf}

        for model, errors in model_dict_errors.items():
            mean_error = errors.mean(axis = 0)
            idx = np.unravel_index(np.argmin(mean_error, axis=None), mean_error.shape)
            imputer = IterativeImputer(estimator=imputer_estimators[idx[0]][0], n_nearest_features=imputer_estimators[idx[0]][1], max_iter = 100, random_state=42).fit(all_data)
            Xtrain_imputed = transform_to_categorical_bool(imputer.transform(Xtrain_norm))
            Xtest_imputed = transform_to_categorical_bool(imputer.transform(Xtest_norm))
            if model == 'ElasticNet':
                best_inner = ElasticNet(alpha=alphas_elastic[idx[1]], l1_ratio=l1_ratios_elastic[idx[2]], max_iter=10000).fit(Xtrain_imputed, ytrain)
            elif model == 'Lars':
                best_inner = Lars(n_nonzero_coefs=nonzero_coefs_lars[idx[1]]).fit(Xtrain_imputed, ytrain)
            elif model == 'RandomForest_feat_select':
                feat_select = Lasso(alpha=1).fit(Xtrain_imputed, ytrain)
                model_feat_select = SelectFromModel(feat_select, prefit=True)
                Xtrain_imputed_feat_select = model_feat_select.transform(Xtrain_imputed)
                Xtest_imputed_feat_select = model_feat_select.transform(Xtest_imputed)
                best_inner = RandomForestRegressor(n_estimators=n_estimators_rf[idx[1]]).fit(Xtrain_imputed_feat_select, ytrain)
            elif model == 'RandomForest':
                best_inner = RandomForestRegressor(n_estimators=n_estimators_rf[idx[1]]).fit(Xtrain_imputed, ytrain)
            
            if model == 'RandomForest_feat_select':
                preds_outer = best_inner.predict(Xtest_imputed_feat_select)
            else:
                preds_outer = best_inner.predict(Xtest_imputed)
            RMSE[model][k1] = np.sqrt(np.mean((preds_outer - ytest)**2))
            print('Optimal', model, ': imputer estimator: ', imputer_estimators[idx[0]], 'model parameters (idx):', idx[1:],
                   ', RMSE:', round(RMSE[model][k1], 3))

    print()
    print('ElasticNet mean RMSE:', round(RMSE['ElasticNet'].mean(), 4), ', std:', round(RMSE['ElasticNet'].std(), 4))
    print('Lars mean RMSE:', round(RMSE['Lars'].mean(), 4), ', std:', round(RMSE['Lars'].std(), 4))
    print('RandomForest_feat_select mean RMSE:', round(RMSE['RandomForest_feat_select'].mean(), 4), 
          ', std:', round(RMSE['RandomForest_feat_select'].std(), 4))
    print('RandomForest mean RMSE:', round(RMSE['RandomForest'].mean(), 4), ', std:', round(RMSE['RandomForest'].std(), 4))

    fig, ax = plt.subplots()
    models = list(RMSE.keys())
    mean = [round(RMSE[model].mean(), 4) for model in models]
    std = [round(RMSE[model].std(), 4) for model in models]
    ax.bar(models, mean, yerr=std, align='center', alpha=0.5, ecolor='black', capsize=10, label='Mean RMSE')
    ax.errorbar(models, mean, yerr=std, fmt='o', color='black', label='$\pm$ 1 std', markersize=0.1)
    ax.legend()
    ax.set_ylabel('RMSE')
    ax.set_title('RMSE for different models')
    ax.yaxis.grid(True)
    #save the figure
    plt.savefig('RMSE_models_iterativeImputer_test_featselect.png')

    #save the RMSE dictionary
    import pickle
    with open('RMSE_dict_iterativeImputer_test_featselect.pkl', 'wb') as f:
        pickle.dump(RMSE, f)