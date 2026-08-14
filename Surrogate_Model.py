import numpy as np
import os 
import pandas as pd
import random
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score
from scipy import stats

def setup_seed(seed):
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.

def mixup(train_data_mat, train_label, alpha = 0.75):
    lam = np.random.beta(alpha, alpha)
    rand_index = np.random.permutation(np.arange(train_data_mat.shape[0]))
    gene_train_data_mat = (1 - lam) * train_data_mat + lam * train_data_mat[rand_index]
    gene_train_label = (1 - lam) * train_label + lam * train_label
    train_data_mat = np.row_stack((train_data_mat,gene_train_data_mat))
    train_label = np.row_stack((train_label, gene_train_label))
    return gene_train_data_mat, gene_train_label

def mixup_time(train_formula_list, train_data_mat, train_label, alpha = 0.75):
    ## packing 
    data_unpacked = np.column_stack((train_data_mat,train_label))
    data_packed = []
    for formula in np.unique(train_formula_list):
        index = np.where(np.array(train_formula_list)==formula)[0]
        data = data_unpacked[index,:]
        data_packed.append(data)
    
    ## generating
    for data in data_packed:
        feature, label = data[:,0:-1], data[:,-1]
        new_feature, new_label = np.copy(feature), np.copy(label)
        for i in range(len(data)-1):
            lam = np.random.beta(alpha, alpha)
            new_feature[i,-1] = lam * feature[i,-1] + (1 - lam) * feature[i+1,-1]
            new_label[i] = lam * label[i] + (1 - lam) * label[i+1] 
        train_data_mat = np.row_stack((train_data_mat,new_feature))
        train_label = np.row_stack((train_label,new_label.reshape(len(new_label),1)))
    return train_data_mat, train_label

def dataset_augmentation(train_formula_list,train_data_mat, train_label, mode = 'time_linear'):
    if mode == 'default':
        train_data_mat, train_label = mixup(train_data_mat, train_label) ## traditional mixup
    elif mode == 'time_linear':
        train_data_mat, train_label = mixup_time(train_formula_list, train_data_mat, train_label) ## our mixup for the drug release prediction
    return train_data_mat, train_label

def get_formula_list(root, task='vitro'):
    if task == 'vitro':
        os.chdir(root + '\\Data sets')
        df = pd.read_excel('literature-derived subset.xlsx', sheet_name='In vitro Release')
        data_mat = df.to_numpy()
        formula_list = np.arange(len(data_mat)//9)
    elif task == 'vivo':
        os.chdir(root + '\\Data sets')
        df = pd.read_excel('literature-derived subset.xlsx', sheet_name='In vivo Release')
        data_mat = df.to_numpy()
        formula_list = np.arange(len(data_mat)//11)
    return formula_list

def dataloading_vitro(formula, root):
    os.chdir(root + '\\Data sets')
    df = pd.read_excel('literature-derived subset.xlsx', sheet_name='In vitro Release')

    column_name = df.columns.tolist()[0:-1]
    column_name[-1] = 'Time' 
    data_mat = df.to_numpy()
    
    train_release_label = []
    train_feature_list = []
    train_formula_list = []
    
    test_release_label = []
    test_feature_list = []
    test_formula_list = []
    
    for train_formula in range(64):
        if train_formula == formula:
            test_feature_list.append(data_mat[train_formula*9:(train_formula+1)*9,:-1])
            test_release_label.append(data_mat[train_formula*9:(train_formula+1)*9,-1:])
            for _ in range(9):
                test_formula_list.append(train_formula)
        else:
            train_feature_list.append(data_mat[train_formula*9:(train_formula+1)*9,:-1])
            train_release_label.append(data_mat[train_formula*9:(train_formula+1)*9,-1:])
            for _ in range(9):
                train_formula_list.append(train_formula)
        
    train_data_mat = np.vstack(train_feature_list).astype('float32')
    train_label = np.vstack(train_release_label).astype('float32')/100
    
    test_data_mat = np.vstack(test_feature_list).astype('float32')
    test_label = np.vstack(test_release_label).astype('float32')/100
    
    train_data_mat, train_label = dataset_augmentation(train_formula_list, train_data_mat, train_label, mode = 'time_linear')
    
    dataset = {'train_data_mat':nan_converter(train_data_mat), 'train_label': train_label.squeeze(),
               'test_data_mat':nan_converter(test_data_mat), 'test_label': test_label.squeeze(),
               'train_formula_list': train_formula_list,'test_formula_list': test_formula_list,
               'column_name': column_name}
    print('dataset loading success')
    return dataset

def dataloading_vivo(formula, root):
    os.chdir(root + '\\Data sets')
    df = pd.read_excel('literature-derived subset.xlsx', sheet_name='In vivo Release')

    column_name = df.columns.tolist()[0:-2]
    column_name[-1] = 'Time' 
    data_mat = df.to_numpy()
    
    train_release_label = []
    train_feature_list = []
    train_formula_list = []
    
    test_release_label = []
    test_feature_list = []
    test_formula_list = []
    
    for train_formula in range(24):
        if train_formula == formula:
            test_feature_list.append(data_mat[train_formula*11:(train_formula+1)*11,:-2])
            test_release_label.append(data_mat[train_formula*11:(train_formula+1)*11,-2:-1])
            for _ in range(11):
                test_formula_list.append(train_formula)
        else:
            train_feature_list.append(data_mat[train_formula*11:(train_formula+1)*11,:-2])
            train_release_label.append(data_mat[train_formula*11:(train_formula+1)*11,-2:-1])
            for _ in range(11):
                train_formula_list.append(train_formula)
        
    train_data_mat = np.vstack(train_feature_list).astype('float32')
    train_label = np.vstack(train_release_label).astype('float32')/100
    
    test_data_mat = np.vstack(test_feature_list).astype('float32')
    test_label = np.vstack(test_release_label).astype('float32')/100
    
    ## optional
    train_data_mat, train_label = dataset_augmentation(train_formula_list, train_data_mat, train_label, mode = 'time_linear')
    
    dataset = {'train_data_mat':nan_converter(train_data_mat), 'train_label': train_label.squeeze(),
               'test_data_mat':nan_converter(test_data_mat), 'test_label': test_label.squeeze(),
               'train_formula_list': train_formula_list,'test_formula_list': test_formula_list,
               'column_name': column_name}
    print('dataset loading success')

    return dataset

def nan_converter(matrix, mode = 'zero'):
    if mode == 'mean':
        column_means = np.nanmean(matrix, axis=0)
        matrix = np.nan_to_num(matrix, nan=column_means)
    else:
        matrix = np.nan_to_num(matrix)
    return matrix

def dataloading_drugload(formula, root):
    os.chdir(root + '\\Data sets')
    df = pd.read_excel('literature-derived subset.xlsx', sheet_name='Drug Loading')

    column_name = df.columns[:-1].tolist()
    data_mat = df.to_numpy()
    release_label = []
    feature_list = []
    for row in range(len(data_mat)):   
        feature_list.append(data_mat[row,:-1])
        release_label.append(data_mat[row,-1])
                
    train_data_mat = np.vstack(feature_list).astype('float32')
    train_label = np.vstack(release_label).astype('float32')/100
    
    test_data_mat = train_data_mat[formula:formula+1,:]
    test_label = train_label[formula:formula+1,:]
    
    train_data_mat = np.delete(train_data_mat, 0, axis = 0)
    train_label = np.delete(train_label, 0, axis = 0)
    
    dataset = {'train_data_mat':nan_converter(train_data_mat), 'train_label': train_label.squeeze(),
               'test_data_mat':nan_converter(test_data_mat), 'test_label': test_label.squeeze(),
               'column_name': column_name}
    print('dataset loading success')
    return dataset

def mse_loss(y_true, y_pred):
    return np.nanmean((y_true - y_pred) ** 2)

def mae_loss(y_true, y_pred):
    return np.nanmean(np.abs(y_true - y_pred))

def enforce_monotonicity(predictions):
    for i in range(1, len(predictions)):
        predictions[i] = max(predictions[i], predictions[i - 1])
    return predictions

def train_pipeline_drugload(formula, seed=18, root = 'D:\\', setting = 'LightGBM', parameter = {'boosting_type': 'gbdt', 'learning_rate':0.1, 'n_estimators':300}):
    print('we are processing task ' + 'drugload' + ' seed ' + str(seed) + ' formula ' + str(formula) + ' with setting' + setting)
    dataset = dataloading_drugload(formula, root)
    if setting == 'LightGBM':
        print(parameter)
        #reg = LGBMRegressor(force_col_wise=True, boosting_type = 'gbdt', learning_rate = 0.1, n_estimators = 300)
        reg = LGBMRegressor(force_col_wise=True, boosting_type = parameter['boosting_type'], learning_rate = parameter['learning_rate'], n_estimators = parameter['n_estimators'])
    elif setting == 'RF':
        from sklearn.ensemble import RandomForestRegressor
        reg = RandomForestRegressor(n_estimators=100, random_state=seed, criterion='squared_error', min_samples_split=2,
                                    min_samples_leaf=2, bootstrap=True, oob_score=True, ccp_alpha=0)
    elif setting == 'KNN':
        from sklearn.neighbors import KNeighborsRegressor
        reg = KNeighborsRegressor(weights='distance', algorithm='ball_tree', leaf_size = 50, p=1)
    elif setting == 'SVR':
        from sklearn.svm import SVR
        reg = SVR(kernel='rbf', gamma='scale', C = 1, epsilon=0.2, shrinking=False)
    elif setting == 'Ridge':
        from sklearn.linear_model import Ridge
        reg = Ridge(random_state=seed)
    elif setting == 'Linear':
        from sklearn.linear_model import LinearRegression
        reg = LinearRegression()
    elif setting == 'Lasso':
        from sklearn.linear_model import Lasso
        reg = Lasso(positive= True, alpha = 0.25, random_state=seed)
    elif setting == 'DT':
        from sklearn.tree import DecisionTreeRegressor
        reg = DecisionTreeRegressor(criterion='squared_error', min_samples_split=4,
                                    min_samples_leaf=4, max_features=None,ccp_alpha=0,
                                    random_state=seed)
    elif setting == 'EDT':
        from sklearn.tree import ExtraTreeRegressor
        reg = ExtraTreeRegressor(random_state=seed)
    elif setting == 'XGBoost':
        import xgboost as xgb
        reg = xgb.XGBRegressor(objective ='reg:squarederror', random_state=seed)
    elif setting == 'AdaBoost':
        from sklearn.ensemble import AdaBoostRegressor
        reg = AdaBoostRegressor()
    elif setting == 'GradientBoost':
        from sklearn.ensemble import GradientBoostingRegressor
        reg = GradientBoostingRegressor()
    elif setting == 'Bagging':
        from sklearn.ensemble import BaggingRegressor
        reg = BaggingRegressor()
    else:
        raise ValueError("setting error")
    reg.fit(dataset['train_data_mat'], dataset['train_label'])
    Predicted_release = reg.predict(dataset['test_data_mat'])
    trimed_prediction = np.clip(Predicted_release,0,1)
    mean_absolute_loss = mae_loss(trimed_prediction, dataset['test_label'])
    mean_square_loss = mse_loss(trimed_prediction, dataset['test_label'])
    dummy_prediction = np.nanmean(dataset['train_label']) + np.zeros_like(dataset['test_label'])
    dummy_mean_square_loss = mse_loss(dummy_prediction, dataset['test_label'])
    dummy_mean_absolute_loss = mae_loss(dummy_prediction, dataset['test_label'])

    result_dict = {'mae': mean_absolute_loss, 'mse': mean_square_loss, 'dummy_mae': dummy_mean_absolute_loss, 'dummy_mse': dummy_mean_square_loss}
    return result_dict

#https://lightgbm.readthedocs.io/en/latest/Parameters.html
def train_pipeline(formula, seed=18, task = 'vivo', root = 'D:\\', setting = 'LightGBM', parameter = {'boosting_type': 'dart', 'learning_rate':0.1, 'n_estimators':500}):
    print('we are processing task ' + task + ' seed ' + str(seed) + ' formula ' + str(formula) + ' with setting' + setting)
    setup_seed(seed)
    if task == 'vivo':
        dataset = dataloading_vivo(formula, root)
    else:
        dataset = dataloading_vitro(formula, root)
    if setting == 'LightGBM':
        print(parameter)
        if task == 'vivo':
            #reg = LGBMRegressor(force_col_wise=True, boosting_type = 'dart', learning_rate = 0.1, n_estimators = 500)
            reg = LGBMRegressor(force_col_wise=True, boosting_type = parameter['boosting_type'], learning_rate = parameter['learning_rate'], n_estimators = parameter['n_estimators'])
        else:
            monotone_constraints = np.zeros(dataset['train_data_mat'].shape[1])
            monotone_constraints[-1] = 1
            # reg = LGBMRegressor(force_col_wise=True,monotone_constraints=monotone_constraints, 
            #                     boosting_type = 'dart', learning_rate = 0.1, n_estimators = 500)
            reg = LGBMRegressor(force_col_wise=True,monotone_constraints=monotone_constraints, boosting_type = parameter['boosting_type'], learning_rate = parameter['learning_rate'], n_estimators = parameter['n_estimators'])
    elif setting == 'RF':
        from sklearn.ensemble import RandomForestRegressor
        reg = RandomForestRegressor(n_estimators=100, random_state=seed, criterion='squared_error', min_samples_split=2,
                                    min_samples_leaf=2, bootstrap=True, oob_score=True, ccp_alpha=0)
    elif setting == 'KNN':
        from sklearn.neighbors import KNeighborsRegressor
        reg = KNeighborsRegressor(weights='distance', algorithm='ball_tree', leaf_size = 50, p=1)
    elif setting == 'SVR':
        from sklearn.svm import SVR
        reg = SVR(kernel='rbf', gamma='scale', C = 1, epsilon=0.2, shrinking=False)
    elif setting == 'Ridge':
        from sklearn.linear_model import Ridge
        reg = Ridge(random_state=seed)
    elif setting == 'Linear':
        from sklearn.linear_model import LinearRegression
        reg = LinearRegression()
    elif setting == 'Lasso':
        from sklearn.linear_model import Lasso
        reg = Lasso(positive= True, alpha = 0.25, random_state=seed)
    elif setting == 'DT':
        from sklearn.tree import DecisionTreeRegressor
        if task == 'vivo':
            reg = DecisionTreeRegressor(criterion='squared_error', min_samples_split=4,
                                    min_samples_leaf=4, max_features=None,ccp_alpha=0,
                                    random_state=seed)
        else:
            monotone_constraints = np.zeros(dataset['train_data_mat'].shape[1])
            monotone_constraints[-1] = 1
            reg = DecisionTreeRegressor(criterion='squared_error', min_samples_split=4,
                                    min_samples_leaf=4, max_features=None,ccp_alpha=0,
                                    random_state=seed, monotonic_cst = monotone_constraints)
    elif setting == 'EDT':
        from sklearn.tree import ExtraTreeRegressor
        if task == 'vivo':
            reg = ExtraTreeRegressor(random_state=seed)
        else:
            monotone_constraints = np.zeros(dataset['train_data_mat'].shape[1])
            monotone_constraints[-1] = 1
            reg = ExtraTreeRegressor(random_state=seed, monotonic_cst = monotone_constraints)
    elif setting == 'XGBoost':
        import xgboost as xgb
        reg = xgb.XGBRegressor(objective ='reg:squarederror', random_state=seed)
    elif setting == 'AdaBoost':
        from sklearn.ensemble import AdaBoostRegressor
        reg = AdaBoostRegressor()
    elif setting == 'GradientBoost':
        from sklearn.ensemble import GradientBoostingRegressor
        reg = GradientBoostingRegressor()
    elif setting == 'Bagging':
        from sklearn.ensemble import BaggingRegressor
        reg = BaggingRegressor()
    else:
        raise ValueError("setting error")

    reg.fit(dataset['train_data_mat'], dataset['train_label'])

    Predicted_release = reg.predict(dataset['test_data_mat'])
    trimed_prediction = np.clip(Predicted_release,0,np.inf)

    mean_square_loss = mse_loss(trimed_prediction, dataset['test_label'])
    mean_absolute_loss = mae_loss(trimed_prediction, dataset['test_label'])
    r2 = r2_score(trimed_prediction, dataset['test_label'])
    spearman = stats.spearmanr(trimed_prediction, dataset['test_label']).statistic
    pearson = stats.pearsonr(trimed_prediction, dataset['test_label']).statistic
    ## dummy prediction
    dummy_prediction = np.nanmean(dataset['train_label']) + np.zeros_like(dataset['test_label'])
    dummy_mean_square_loss = mse_loss(dummy_prediction, dataset['test_label'])
    dummy_mean_absolute_loss = mae_loss(dummy_prediction, dataset['test_label'])
    
    result_dict = {'mae': mean_absolute_loss, 'mse': mean_square_loss, 'r2':r2, 'spearman': spearman, 'pearson' : pearson,
                   'dummy_mae': dummy_mean_absolute_loss, 'dummy_mse': dummy_mean_square_loss }
    return result_dict

def main_function_drugload(seed=18, root = 'D:\\', setting = 'LightGBM', parameter = {'boosting_type': 'gbdt', 'learning_rate':0.1, 'n_estimators':300}):
    setting_list = ['LightGBM', 'RF', 'KNN', 'SVR', 'Ridge', 'Linear', 'Lasso', 'DT', 
                    'EDT', 'XGBoost', 'AdaBoost', 'GradientBoost', 'Bagging']
    formula_nums = len(pd.read_excel(root + '\\Data sets\\literature-derived subset.xlsx', sheet_name='Drug Loading'))
    formula_list = np.arange(formula_nums)
    mae_mat = np.zeros((len(setting_list), formula_nums))
    mse_mat = np.zeros((len(setting_list),formula_nums))
    dummy_mae_mat = np.zeros((len(setting_list),formula_nums))
    dummy_mse_mat = np.zeros((len(setting_list),formula_nums))
    
    for setting, index in zip(setting_list, np.arange(len(setting_list))):
        for i , formula in zip(np.arange(formula_nums), formula_list): ## compute the average results under 100 random seeds.
            result_dict = train_pipeline_drugload(formula, seed=seed, root = root, setting = setting, parameter = parameter)
            mae_mat[index, i] = result_dict['mae']
            mse_mat[index,i] = result_dict['mse']
            dummy_mae_mat[index,i] = result_dict['dummy_mae']
            dummy_mse_mat[index,i] = result_dict['dummy_mse']
    print('Average MAE', np.nanmean(mae_mat, axis=1), '_std_', np.nanstd(mae_mat, axis=1))
    print('Average MSE', np.nanmean(mse_mat, axis=1), '_std_', np.nanstd(mse_mat, axis=1))
    result_dict ={'mae_mat': mae_mat, 'mse_mat': mse_mat, 'dummy_mae_mat': dummy_mae_mat, 'dummy_mse_mat': dummy_mse_mat}


    os.chdir(root + '\\result')
    filename = 'seed' + str(seed) + 'drugload' + '_loocv_' + 'allmodel.npy' 
    np.save(filename, result_dict)
    
    return 

def main_function(seed=18, task='vivo', root = 'D:\\', setting = 'LightGBM', parameter = {'boosting_type': 'dart', 'learning_rate':0.1, 'n_estimators':500}):
    setting_list = ['LightGBM', 'RF', 'KNN', 'SVR', 'Ridge', 'Linear', 'Lasso', 'DT', 
                    'EDT', 'XGBoost', 'AdaBoost', 'GradientBoost', 'Bagging']
    formula_list = get_formula_list(root=root, task=task)
    formula_nums = len(formula_list)
    mae_mat = np.zeros((len(setting_list), formula_nums))
    mse_mat = np.zeros((len(setting_list), formula_nums))
    spearman_mat = np.zeros((len(setting_list), formula_nums))
    r2_mat = np.zeros((len(setting_list), formula_nums))
    pearson_mat = np.zeros((len(setting_list), formula_nums))
    dummy_mae_mat = np.zeros((len(setting_list), formula_nums))
    dummy_mse_mat = np.zeros((len(setting_list), formula_nums))
    for setting, index in zip(setting_list, np.arange(len(setting_list))):
        for i , formula in zip(np.arange(formula_nums), formula_list): ## compute the average results under 100 random seeds.
            result_dict = train_pipeline(formula, seed=seed, task=task, root = root, setting = setting, parameter = parameter)
            mae_mat[index, i] = result_dict['mae']
            mse_mat[index, i] = result_dict['mse']
            spearman_mat[index, i] = result_dict['spearman']
            pearson_mat[index, i] = result_dict['pearson']
            r2_mat[index, i] = result_dict['r2']
            dummy_mae_mat[index, i] = result_dict['dummy_mae']
            dummy_mse_mat[index, i] = result_dict['dummy_mse']

    result_dict ={'mae_mat': mae_mat, 'mse_mat': mse_mat, 'spearman_mat': spearman_mat, 'r2_mat': r2_mat,
                  'pearson_mat': pearson_mat, 'dummy_mae_mat': dummy_mae_mat, 'dummy_mse_mat': dummy_mse_mat}
    
    os.chdir(root + '\\result')
    filename = 'seed' + str(seed) + task + '_loocv_' + 'allmodel.npy' 
    np.save(filename, result_dict)
    
    return 

def main_function_drugload_search(seed=18, root = 'D:\\', setting = 'LightGBM', parameter = {'boosting_type': 'gbdt', 'learning_rate':0.1, 'n_estimators':300}):
    setting_list = ['LightGBM']
    formula_nums = len(pd.read_excel(root + '\\Data sets\\literature-derived subset.xlsx', sheet_name='Drug Loading'))
    formula_list = np.arange(formula_nums)
    mae_mat = np.zeros((len(setting_list), formula_nums))
    mse_mat = np.zeros((len(setting_list),formula_nums))
    dummy_mae_mat = np.zeros((len(setting_list),formula_nums))
    dummy_mse_mat = np.zeros((len(setting_list),formula_nums))
    
    for setting, index in zip(setting_list, np.arange(len(setting_list))):
        for i , formula in zip(np.arange(formula_nums), formula_list): ## compute the average results under 100 random seeds.
            result_dict = train_pipeline_drugload(formula, seed=seed, root = root, setting = setting, parameter = parameter)
            mae_mat[index, i] = result_dict['mae']
            mse_mat[index,i] = result_dict['mse']
            dummy_mae_mat[index,i] = result_dict['dummy_mae']
            dummy_mse_mat[index,i] = result_dict['dummy_mse']
    print('Average MAE', np.nanmean(mae_mat, axis=1), '_std_', np.nanstd(mae_mat, axis=1))
    print('Average MSE', np.nanmean(mse_mat, axis=1), '_std_', np.nanstd(mse_mat, axis=1))
    result_dict ={'mae_mat': mae_mat, 'mse_mat': mse_mat, 'dummy_mae_mat': dummy_mae_mat, 'dummy_mse_mat': dummy_mse_mat}
    
    return result_dict

def main_function_search(seed=18, task='vivo', root = 'D:\\', setting = 'LightGBM', parameter = {'boosting_type': 'dart', 'learning_rate':0.1, 'n_estimators':500}):
    setting_list = ['LightGBM']
    formula_list = get_formula_list(root=root, task=task)
    formula_nums = len(formula_list)
    mae_mat = np.zeros((len(setting_list), formula_nums))
    mse_mat = np.zeros((len(setting_list), formula_nums))
    spearman_mat = np.zeros((len(setting_list), formula_nums))
    r2_mat = np.zeros((len(setting_list), formula_nums))
    pearson_mat = np.zeros((len(setting_list), formula_nums))
    dummy_mae_mat = np.zeros((len(setting_list), formula_nums))
    dummy_mse_mat = np.zeros((len(setting_list), formula_nums))
    for setting, index in zip(setting_list, np.arange(len(setting_list))):
        for i , formula in zip(np.arange(formula_nums), formula_list): ## compute the average results under 100 random seeds.
            result_dict = train_pipeline(formula, seed=seed, task=task, root = root, setting = setting, parameter = parameter)
            mae_mat[index, i] = result_dict['mae']
            mse_mat[index, i] = result_dict['mse']
            spearman_mat[index, i] = result_dict['spearman']
            pearson_mat[index, i] = result_dict['pearson']
            r2_mat[index, i] = result_dict['r2']
            dummy_mae_mat[index, i] = result_dict['dummy_mae']
            dummy_mse_mat[index, i] = result_dict['dummy_mse']

    result_dict ={'mae_mat': mae_mat, 'mse_mat': mse_mat, 'spearman_mat': spearman_mat, 'r2_mat': r2_mat,
                  'pearson_mat': pearson_mat, 'dummy_mae_mat': dummy_mae_mat, 'dummy_mse_mat': dummy_mse_mat}
    
    return result_dict

def save2excel(task, seed, root):
    os.chdir(root + '\\result\\')

    filename = 'seed' + str(seed) + task + '_loocv_' + 'allmodel.npy' 

    result_dict = np.load(filename, allow_pickle = True).item()
    import pandas as pd
    setting_list = ['LightGBM', 'RF', 'KNN', 'SVR', 'Ridge', 'Linear', 'Lasso', 'DT', 
                    'EDT', 'XGBoost', 'AdaBoost', 'GradientBoost', 'Bagging']
    result_dataframe = {
    'Methods': np.array(setting_list),
    'Mean MAE': np.nanmean(result_dict['mae_mat'], axis = 1),
    'Std MAE': np.nanstd(result_dict['mae_mat'], axis = 1),
    'Mean MSE': np.nanmean(result_dict['mse_mat'], axis = 1),
    'Std MSE': np.nanstd(result_dict['mse_mat'], axis = 1),
    'Mean Pearson': np.nanmean(result_dict['pearson_mat'], axis = 1),
    'Std Pearson': np.nanstd(result_dict['pearson_mat'], axis = 1),
    'Mean Spearman': np.nanmean(result_dict['spearman_mat'], axis = 1),
    'Std Spearman': np.nanstd(result_dict['spearman_mat'], axis = 1)
    }
    result_dataframe = pd.DataFrame(result_dataframe)
    os.chdir(root + '\\result')

    result_dataframe.to_excel('seed' + str(seed) + task + '_loocv_' + 'model_comparison.xlsx'
                                  , index=False, engine='openpyxl')

    formula_list = get_formula_list(root=root, task=task)
    for setting, index in zip(setting_list, np.arange(len(setting_list))):
        result_dataframe = {
        'Formula_one': np.array(formula_list),
        'MAE': result_dict['mae_mat'][index,:],
        'MSE': result_dict['mse_mat'][index,:],
        'Pearson': result_dict['pearson_mat'][index,:],
        'Spearman': result_dict['spearman_mat'][index,:]
        }
        result_dataframe = pd.DataFrame(result_dataframe)

        average = {'Formula_one': 'Average',
        'MAE': np.nanmean(result_dict['mae_mat'][index,:]),
        'MSE': np.nanmean(result_dict['mse_mat'][index,:]),
        'Pearson': np.nanmean(result_dict['pearson_mat'][index,:]),
        'Spearman': np.nanmean(result_dict['spearman_mat'][index,:])}

        std = {'Formula_one': 'STD',
        'MAE': np.nanstd(result_dict['mae_mat'][index,:]),
        'MSE': np.nanstd(result_dict['mse_mat'][index,:]),
        'Pearson': np.nanstd(result_dict['pearson_mat'][index,:]),
        'Spearman': np.nanstd(result_dict['spearman_mat'][index,:])}

        result_dataframe.loc[len(result_dataframe)] = average
        result_dataframe.loc[len(result_dataframe)] = std
        os.chdir(root + '\\result')
        
        result_dataframe.to_excel('seed' + str(seed) + task + '_loocv_' + setting + '.xlsx'
                                      , index=False, engine='openpyxl')
    return

def save2excel_drugload(seed, root):
    os.chdir(root + '\\result')
    filename = 'seed' + str(seed) + 'drugload' + '_loocv_' + 'allmodel.npy' 
    result_dict = np.load(filename, allow_pickle = True).item()
    import pandas as pd
    setting_list = ['LightGBM', 'RF', 'KNN', 'SVR', 'Ridge', 'Linear', 'Lasso', 'DT', 
                    'EDT', 'XGBoost', 'AdaBoost', 'GradientBoost', 'Bagging']
    result_dataframe = {
    'Methods': np.array(setting_list),
    'Mean MAE': np.nanmean(result_dict['mae_mat'], axis = 1),
    'Std MAE': np.nanstd(result_dict['mae_mat'], axis = 1),
    'Mean MSE': np.nanmean(result_dict['mse_mat'], axis = 1),
    'Std MSE': np.nanstd(result_dict['mse_mat'], axis = 1)
    }
    result_dataframe = pd.DataFrame(result_dataframe)
    os.chdir(root + '\\result')
    result_dataframe.to_excel('seed' + str(seed) + 'drugload' + '_loocv_' + 'model_comparison.xlsx', index=False, engine='openpyxl')
    
    ## save the results of each methods
    formula_nums = len(pd.read_excel(root + '\\Data sets\\literature-derived subset.xlsx', sheet_name='Drug Loading'))
    formula_list = np.arange(formula_nums)
    for setting, index in zip(setting_list, np.arange(len(setting_list))):

        result_dataframe = {
        'Formula_one': np.array(formula_list),
        'MAE': result_dict['mae_mat'][index,:],
        'MSE': result_dict['mse_mat'][index,:],
        }
        result_dataframe = pd.DataFrame(result_dataframe)
        
        average = {'Formula_one': 'Average',
        'MAE': np.nanmean(result_dict['mae_mat'][index,:]),
        'MSE': np.nanmean(result_dict['mse_mat'][index,:])}
        
        std = {'Formula_one': 'STD',
        'MAE': np.nanstd(result_dict['mae_mat'][index,:]),
        'MSE': np.nanstd(result_dict['mse_mat'][index,:])}
        
        result_dataframe.loc[len(result_dataframe)] = average
        result_dataframe.loc[len(result_dataframe)] = std
        os.chdir(root + '\\result')
        result_dataframe.to_excel('seed' + str(seed) + 'drugload' + '_loocv_' + setting + '.xlsx', index=False, engine='openpyxl')
    return

def grid_search(task, seed = 18, root = 'D:\\'):
    print('gridsearch_task:', task)
    n_estimators_list = [100, 300, 500]
    learning_rate_list = [0.1, 0.01, 0.001]
    boosting_type_list = ['gbdt', 'dart']
    if task == 'drugload':
        grid_result = []
        parameter_list = []
        for n_estimators in n_estimators_list:
            for learning_rate in learning_rate_list:
                for boosting_type in boosting_type_list:
                    lightgbm_parameter = {'n_estimators': n_estimators, 'boosting_type': boosting_type, 'learning_rate': learning_rate}
                    parameter_list.append(lightgbm_parameter)
                    print('current n_estimators:', n_estimators)
                    print('current boosting_type:', boosting_type)
                    print('current learning_rate:', learning_rate)
                    result_dict = main_function_drugload_search(seed=18, root = root, parameter=lightgbm_parameter)
                    grid_result.append(result_dict)
        os.chdir(root + '\\result')
        filename = 'seed' + str(seed) + task + '_loocv_' + 'gridsearch.npy' 
        np.save(filename, grid_result)
        
        mae_result = []
        for result_dict in grid_result:
            mae_result.append(np.mean(result_dict['mae_mat']))
        best_index, best_mae = np.argmin(mae_result), np.min(mae_result)
        best_parameter = parameter_list[best_index]
        print(best_parameter)
    else:
        grid_result = []
        parameter_list = []
        for n_estimators in n_estimators_list:
            for learning_rate in learning_rate_list:
                for boosting_type in boosting_type_list:
                    lightgbm_parameter = {'n_estimators': n_estimators, 'boosting_type': boosting_type, 'learning_rate': learning_rate}
                    parameter_list.append(lightgbm_parameter)
                    print('current n_estimators:', n_estimators)
                    print('current boosting_type:', boosting_type)
                    print('current learning_rate:', learning_rate)
                    result_dict = main_function_search(seed=18, task = task, root = root, parameter=lightgbm_parameter)
                    grid_result.append(result_dict)
        os.chdir(root + '\\result')
        filename = 'seed' + str(seed) + task + '_loocv_' + 'gridsearch.npy' 
        np.save(filename, grid_result)
        
        os.chdir(root + '\\result')
        grid_result = np.load('seed' + str(seed) + task + '_loocv_' + 'gridsearch.npy', allow_pickle = True)
        mae_result = []
        for result_dict in grid_result:
            mae_result.append(np.mean(result_dict['mae_mat']))
        best_index, best_mae = np.argmin(mae_result), np.min(mae_result)
        best_parameter = parameter_list[best_index]
        print(best_parameter)
    return best_parameter, best_mae

if __name__ == '__main__':
    
    root = 'D:\projectzrs\EMAF_Demo' ## modify your root path
    ## Leave-one-out cross validation of different forward models. The hyperparameter grid search process is optional.
    #grid_search(task='vitro', seed = 18, root = root)
    
    main_function(seed=18, task='vitro', root = root)
    save2excel(task='vitro', seed = 18, root = root)
    
    main_function(seed=18, task='vivo', root = root)
    save2excel(task='vivo', seed = 18, root = root)
    
    main_function_drugload(seed = 18, root = root)
    save2excel_drugload(seed = 18, root= root)
    
    
    