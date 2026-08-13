import numpy as np
import os 
import pandas as pd
import random
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy import stats

def setup_seed(seed):
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.

def dataleaking_checker(train_list, test_list):
    leak_flag = False
    for i in train_list:
        if i in test_list:
            leak_flag = True
            break
    if leak_flag:
        print('dataleaking')
    else:
        print('no dataleaking')

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

def dataset_augmentation(train_formula_list,train_data_mat, train_label, mode = 'noaug'):
    if mode == 'default':
        train_data_mat, train_label = mixup(train_data_mat, train_label) ## traditional mixup
    elif mode == 'time_linear':
        train_data_mat, train_label = mixup_time(train_formula_list, train_data_mat, train_label) ## our mixup for the drug release prediction
    return train_data_mat, train_label


def get_formula_list(root, task='vitro'):
    if task == 'vitro':
        os.chdir(root + '\\Data sets')
        df = pd.read_excel('In Vitro Release.xlsx')
        data_mat = df.to_numpy()
        formula_list = np.unique(data_mat[:,0]).tolist()
    elif task == 'vivo':
        os.chdir(root + '\\Data sets')
        df = pd.read_excel('In Vivo Release.xlsx')
        df = df.drop(['Time/h','AUC(0-t)','AUC(0-inf)',df.columns[-1]],axis='columns')
        df = df.drop(['Release_3'],axis='columns')
        data_mat = df.to_numpy()
        formula_list = np.unique(data_mat[:,0]).tolist()
    return formula_list

def find_repeat_list(data_mat):
    data_mat_refine = np.column_stack((data_mat[:,0], data_mat[:,2:-2]))
    feature_num = data_mat_refine.shape[1]-1
    data_packed = []
    for formula in np.unique(data_mat[:,0]):
        index = np.where(np.array(data_mat[:,0])==formula)[0][0]
        data = data_mat_refine[index,0:]
        data_packed.append(data)
    
    data_packed = np.vstack(data_packed)
    repeat_list = []
    for i in range(len(data_packed)):
        repeat_list_formula = [data_packed[i,0]]
        for j in range(len(data_packed)):
            if data_packed[i,0] == data_packed[j,0]:
                continue
            if np.sum(data_packed[i,1:]==data_packed[j,1:]) == feature_num:
                repeat_list_formula.append(data_packed[j,0])
        repeat_list.append(repeat_list_formula)
   
    real_repeat_list = []
    for i in repeat_list:
        if len(i)>1:
           real_repeat_list.append(np.sort(i).tolist()) 

    pure_repeat_list = []
    for i in real_repeat_list:
        if i not in pure_repeat_list:
            pure_repeat_list.append(i)
    return pure_repeat_list

def dataloading_vitro(formula, root, Additives = False, aug = True):
    os.chdir(root + '\\Data sets')
    df = pd.read_excel('In Vitro Release.xlsx')
    if Additives == False:
        print('remove additives information')
        df = df.drop(['Additives_2','Additives_3','Additives_4','Additives_5','Additives_6', 'microspheres  properties_2'],axis='columns')
    column_name = df.columns[2:-1].tolist()
    column_name[-1] = 'Time' 
    data_mat = df.to_numpy()
    print(column_name)
    formula_list = np.unique(data_mat[:,0]).tolist()
    formula_list.remove(formula)
    
    formulas_train_list = formula_list
    formulas_test_list = [formula]
    
    pure_repeat_list = find_repeat_list(data_mat)
    remove_flag = False
    for f in pure_repeat_list:
        if formula in f:
            list_to_remove = f
            remove_flag = True
    if remove_flag:
        list_to_remove.remove(formula)
        for f in list_to_remove:
            formulas_train_list.remove(f)
        
    dataleaking_checker(formulas_train_list, formulas_test_list)
    train_release_label = []
    train_feature_list = []
    train_formula_list = []
    for formula in formulas_train_list:
        for row in range(len(data_mat)):   
            if data_mat[row,0] == formula:
                train_feature_list.append(data_mat[row,2:-1])
                train_release_label.append(data_mat[row,-1])
                train_formula_list.append(formula)
                
    test_release_label = []
    test_feature_list = []
    test_formula_list = []
    for formula in formulas_test_list:
        for row in range(len(data_mat)):   
            if data_mat[row,0] == formula:
                test_feature_list.append(data_mat[row,2:-1])
                test_release_label.append(data_mat[row,-1])
                test_formula_list.append(formula)
                
    train_data_mat = np.vstack(train_feature_list).astype('float32')
    train_label = np.vstack(train_release_label).astype('float32')/100
    if aug:
        train_data_mat, train_label = dataset_augmentation(train_formula_list, train_data_mat, train_label, mode = 'time_linear')
    else:
        print('no augmentation!!!')
    test_data_mat = np.vstack(test_feature_list).astype('float32')
    test_label = np.vstack(test_release_label).astype('float32')/100

    dataset = {'train_data_mat':nan_converter(train_data_mat), 'train_label': train_label.squeeze(),
               'test_data_mat':nan_converter(test_data_mat), 'test_label': test_label.squeeze(),
               'train_formula_list': train_formula_list,'test_formula_list': test_formula_list,
               'column_name': column_name}
    print('dataset loading success')
    return dataset

def dataleaking_checker_v2(train_data_mat, test_data_mat):
    train_data_mat, test_data_mat = train_data_mat[:,:-1], test_data_mat[:,:-1]
    feature_num = train_data_mat.shape[1]
    leak_flag = False
    for i in range(len(train_data_mat)):
        for j in range(len(test_data_mat)):
            if np.sum(train_data_mat[i,:]==test_data_mat[j,:]) == feature_num:
                leak_flag = True
                break
    if leak_flag:
        print('dataleaking')
    else:
        print('no dataleaking')
        
def dataloading_vivo(formula, root, Additives = False, aug = True):
    os.chdir(root + '\\Data sets')
    df = pd.read_excel('In Vivo Release.xlsx')
    df = df.drop(['Time/h','AUC(0-t)','AUC(0-inf)',df.columns[-1]],axis='columns')
    df = df.drop(['Release_3'],axis='columns')
    if Additives == False:
        print('remove additives information')
        df = df.drop(['Additives_2','Additives_3','Additives_4','Additives_5','Additives_6', 'microspheres  properties_2'],axis='columns')
    column_name = df.columns[2:-1].tolist()
    column_name[-1] = 'Time' 
    print(column_name)
    data_mat = df.to_numpy()
    
    formula_list = np.unique(data_mat[:,0]).tolist()
    formula_list.remove(formula)
    
    formulas_train_list = formula_list
    formulas_test_list = [formula]
    
    pure_repeat_list = find_repeat_list(data_mat)
    remove_flag = False
    for f in pure_repeat_list:
        if formula in f:
            list_to_remove = f
            remove_flag = True
    if remove_flag:
        list_to_remove.remove(formula)
        for f in list_to_remove:
            formulas_train_list.remove(f)
        
    dataleaking_checker(formulas_train_list, formulas_test_list)
    train_release_label = []
    train_feature_list = []
    train_formula_list = []
    for formula in formulas_train_list:
        for row in range(len(data_mat)):   
            if data_mat[row,0] == formula:
                train_feature_list.append(data_mat[row,2:-1])
                train_release_label.append(data_mat[row,-1])
                train_formula_list.append(formula)
                
    test_release_label = []
    test_feature_list = []
    test_formula_list = []
    for formula in formulas_test_list:
        for row in range(len(data_mat)):   
            if data_mat[row,0] == formula:
                test_feature_list.append(data_mat[row,2:-1])
                test_release_label.append(data_mat[row,-1])
                test_formula_list.append(formula)
                
    train_data_mat = np.vstack(train_feature_list).astype('float32')
    train_label = np.vstack(train_release_label).astype('float32')
    if aug:
        train_data_mat, train_label = dataset_augmentation(train_formula_list, train_data_mat, train_label, mode = 'time_linear')
    else:
        print('no augmentation!!!')
    test_data_mat = np.vstack(test_feature_list).astype('float32')
    test_label = np.vstack(test_release_label).astype('float32')

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

def dataloading_drugload(formula, root, Additives = False):
    index = formula - 1
    os.chdir(root + '\\Data sets')
    df = pd.read_excel('Drug loading data sets.xlsx')
    if Additives == False:
        print('remove additives information')
        df = df.drop(['Additives_2','Additives_3','Additives_4','Additives_5','Additives_6', 'microspheres  properties_2'],axis='columns')
    df = df.drop(['microspheres  properties_4'],axis='columns')
    column_name = df.columns[2:-1].tolist()
    print(column_name)
    data_mat = df.to_numpy()
    release_label = []
    feature_list = []
    for row in range(len(data_mat)):   
        feature_list.append(data_mat[row,2:-1])
        release_label.append(data_mat[row,-1])
                
    train_data_mat = np.vstack(feature_list).astype('float32')
    train_label = np.vstack(release_label).astype('float32')/100
    
    test_data_mat, test_label = train_data_mat[index,:].reshape(1, train_data_mat.shape[1]), train_label[index,:].reshape(1, 1)
    train_data_mat = np.delete(train_data_mat, index, axis=0)
    train_label = np.delete(train_label, index, axis=0)
    
    dataset = {'train_data_mat':nan_converter(train_data_mat), 'train_label': train_label.squeeze(),
               'test_data_mat':nan_converter(test_data_mat), 'test_label': test_label.squeeze(),
               'column_name': column_name}
    print('dataset loading success')
    return dataset

def mse_loss(y_true, y_pred):
    return np.nanmean((y_true - y_pred) ** 2)

def mae_loss(y_true, y_pred):
    return np.nanmean(np.abs(y_true - y_pred))

def result_plot(formula, root, trimed_prediction, dataset, task, seed=18):
    mean_square_loss = mse_loss(trimed_prediction, dataset['test_label'])
    mean_absolute_loss = mae_loss(trimed_prediction, dataset['test_label'])
    
    data = np.column_stack((dataset['test_data_mat'][:,-1],trimed_prediction,dataset['test_label']))
    
    fig, ax = plt.subplots(figsize=(8,5))
    
    ax.plot(data[:,0], data[:,1], label='Predicted', linestyle='--', marker='o', markersize=8, 
            markeredgecolor="black", alpha=0.8)
    
    ax.plot(data[:,0], data[:,2], label='Experimental', linestyle='--', marker='o', markersize=8, 
            alpha=0.8, markeredgecolor="black")
    plt.legend()
    ax.set_xlabel('Time (Days)', fontsize=15, color = 'black', weight='bold')
    if task == 'vivo':
        ax.set_ylabel('Plasma Concentration in vivo', fontsize=15, color = 'black', weight='bold')
    else:
        ax.set_ylabel('Fractional Drug Release', fontsize=15, color = 'black', weight='bold')
    ax.grid(False)
    if task == 'vitro':
        plt.ylim(0,1)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.title(task + '_Formulation.no_'+ str(formula) + '_MSE_' + '(' + str(np.round(mean_square_loss,2)) + ')' + '_MAE_' + '(' + str(np.round(mean_absolute_loss,2)) + ')')

    ax.tick_params(colors='black')
    plt.tight_layout()
    
    file_path = root + '\\figure\\v2_post_case_'+ task + '\\' + 'seed_' + str(seed)
    if not os.path.exists(file_path):
        print('creating new file path', file_path)
        os.makedirs(file_path)
    os.chdir(file_path)
    plt.savefig('case_'+ task + '_formulation_' + str(formula) + '.jpg',bbox_inches = 'tight', transparent=True, dpi = 300)
    plt.close()
    

def curve_postprocessing(trimed_prediction, time):
    diff_list = np.diff(trimed_prediction, n = 1, prepend = 0)
    bad_index_list = np.where(diff_list<0)[0]
    if len(bad_index_list) > 0:
        bad_index = np.where(diff_list<0)[0][0]
        print('bad prediction')
        for index in range(bad_index, len(trimed_prediction)):
            if index > 1:
                slope = (trimed_prediction[index-1] -  trimed_prediction[index-2]) / (time[index-1] -  time[index-2])
                corrected_value = slope * (time[index] -  time[index-1]) + trimed_prediction[index-1]
                trimed_prediction[index] = min(corrected_value, 1)
                
                new_diff_list = np.diff(trimed_prediction, n = 1, prepend = 0)
                new_bad_index = np.where(new_diff_list<0)[0]
                if len(new_bad_index) < 1:
                    break
            else:
                continue
    else:
        print('good prediction')
        trimed_prediction = trimed_prediction
    return trimed_prediction

def enforce_monotonicity(predictions):
    for i in range(1, len(predictions)):
        predictions[i] = max(predictions[i], predictions[i - 1])
    return predictions

def train_pipeline_drugload(formula, seed=18, Additives = False, root = 'D:\\', setting = 'LightGBM'):
    print('we are processing task ' + 'drugload' + ' seed ' + str(seed) + ' formula ' + str(formula) + ' with setting' + setting)
    dataset = dataloading_drugload(formula, root, Additives)
    if setting == 'LightGBM':
        reg = LGBMRegressor(force_col_wise=True, boosting_type = 'gbdt', learning_rate = 0.1, n_estimators = 300)
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
def train_pipeline(formula, seed=18, Additives = False, task = 'vivo', root = 'D:\\', setting = 'LightGBM', aug = True):
    print('we are processing task ' + task + ' seed ' + str(seed) + ' formula ' + str(formula) + ' with setting' + setting)
    setup_seed(seed)
    if task == 'vivo':
        dataset = dataloading_vivo(formula, root, Additives, aug)
    else:
        dataset = dataloading_vitro(formula, root, Additives, aug)
    if setting == 'LightGBM':
        if task == 'vivo':
            reg = LGBMRegressor(force_col_wise=True, boosting_type = 'dart', learning_rate = 0.1, n_estimators = 500)
        else:
            monotone_constraints = np.zeros(dataset['train_data_mat'].shape[1])
            monotone_constraints[-1] = 1
            reg = LGBMRegressor(force_col_wise=True,monotone_constraints=monotone_constraints, 
                                boosting_type = 'dart', learning_rate = 0.1, n_estimators = 500)
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

def main_function_drugload(seed=18,  Additives=False, root = 'D:\\', setting = 'LightGBM'):
    setting_list = ['LightGBM', 'RF', 'KNN', 'SVR', 'Ridge', 'Linear', 'Lasso', 'DT', 
                    'EDT', 'XGBoost', 'AdaBoost', 'GradientBoost', 'Bagging']
    formula_nums = len(pd.read_excel(root + '\\Data sets\\Drug loading data sets.xlsx'))
    formula_list = np.arange(formula_nums) + 1 
    mae_mat = np.zeros((len(setting_list), formula_nums))
    mse_mat = np.zeros((len(setting_list),formula_nums))
    dummy_mae_mat = np.zeros((len(setting_list),formula_nums))
    dummy_mse_mat = np.zeros((len(setting_list),formula_nums))
    
    for setting, index in zip(setting_list, np.arange(len(setting_list))):
        for i , formula in zip(np.arange(formula_nums), formula_list): ## compute the average results under 100 random seeds.
            result_dict = train_pipeline_drugload(formula, seed=seed, Additives=Additives, root = root, setting = setting)
            mae_mat[index, i] = result_dict['mae']
            mse_mat[index,i] = result_dict['mse']
            dummy_mae_mat[index,i] = result_dict['dummy_mae']
            dummy_mse_mat[index,i] = result_dict['dummy_mse']
    print('Average MAE', np.nanmean(mae_mat, axis=1), '_std_', np.nanstd(mae_mat, axis=1))
    print('Average MSE', np.nanmean(mse_mat, axis=1), '_std_', np.nanstd(mse_mat, axis=1))
    result_dict ={'mae_mat': mae_mat, 'mse_mat': mse_mat, 'dummy_mae_mat': dummy_mae_mat, 'dummy_mse_mat': dummy_mse_mat}
    if Additives:
        os.chdir(root + '\\final_result_nospan')
        filename = 'seed' + str(seed) + 'drugload' + '_loocv_' + 'withadd_final_othermodel.npy' 
        np.save(filename, result_dict)
    else:
        os.chdir(root + '\\final_result_nospan')
        filename = 'seed' + str(seed) + 'drugload' + '_loocv_' + 'withoutadd_final_othermodel.npy' 
        np.save(filename, result_dict)
    
    os.chdir(root + '\\final_result_nospan')
    filename = 'seed' + str(seed) + 'drugload' + '_loocv_' + 'withoutadd_final_othermodel.npy' 
    result_dict = np.load(filename, allow_pickle = True).item()
    print('Average MAE', np.nanmean(result_dict['mae_mat'], axis = 1), '_std_', np.nanstd(result_dict['mae_mat'], axis = 1))
    print('Average MAE', np.nanmean(result_dict['mse_mat'], axis = 1), '_std_', np.nanstd(result_dict['mse_mat'], axis = 1))
    return 

def main_function(seed=18, task='vivo', Additives=False, root = 'D:\\', setting = 'LightGBM', aug=True):
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
            result_dict = train_pipeline(formula, seed=seed, Additives=Additives, task=task, root = root, setting = setting, aug=aug)
            mae_mat[index, i] = result_dict['mae']
            mse_mat[index, i] = result_dict['mse']
            spearman_mat[index, i] = result_dict['spearman']
            pearson_mat[index, i] = result_dict['pearson']
            r2_mat[index, i] = result_dict['r2']
            dummy_mae_mat[index, i] = result_dict['dummy_mae']
            dummy_mse_mat[index, i] = result_dict['dummy_mse']

    result_dict ={'mae_mat': mae_mat, 'mse_mat': mse_mat, 'spearman_mat': spearman_mat, 'r2_mat': r2_mat,
                  'pearson_mat': pearson_mat, 'dummy_mae_mat': dummy_mae_mat, 'dummy_mse_mat': dummy_mse_mat}
    if aug:
        os.chdir(root + '\\final_result_nospan')
        filename = 'seed' + str(seed) + task + '_loocv_' + 'withoutadd_final_othermodel.npy' 
        np.save(filename, result_dict)
    else:
        os.chdir(root + '\\final_result_nospan')
        filename = 'seed' + str(seed) + task + '_loocv_' + 'withoutadd_final_othermodel_noaug.npy' 
        np.save(filename, result_dict)
    
    return 

def save2excel(task, seed, root, aug):
    os.chdir(root + '\\final_result_nospan\\')
    if aug:
        filename = 'seed' + str(seed) + task + '_loocv_' + 'withoutadd_final_othermodel.npy'
    else:
        filename = 'seed' + str(seed) + task + '_loocv_' + 'withoutadd_final_othermodel_noaug.npy'
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
    os.chdir(root + '\\final_result_nospan_excel\\othermodel\\loocv')
    if aug:
        result_dataframe.to_excel('seed' + str(seed) + task + '_loocv_' + 'model_comparison.xlsx'
                                  , index=False, engine='openpyxl')
    else:
        result_dataframe.to_excel('seed' + str(seed) + task + '_loocv_' + 'model_comparison_noaug.xlsx'
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
        os.chdir(root + '\\final_result_nospan_excel\\othermodel\\loocv\\details\\' + task)
        if aug:
            result_dataframe.to_excel('seed' + str(seed) + task + '_loocv_' + setting + '.xlsx'
                                      , index=False, engine='openpyxl')
        else:
            result_dataframe.to_excel('seed' + str(seed) + task + '_loocv_' + setting + '_noaug.xlsx'
                                      , index=False, engine='openpyxl')
    return

def save2excel_drugload(seed, root):
    os.chdir(root + '\\final_result_nospan')
    filename = 'seed' + str(seed) + 'drugload' + '_loocv_' + 'withoutadd_final_othermodel.npy'
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
    os.chdir(root + '\\final_result_nospan_excel\\othermodel\\loocv')
    result_dataframe.to_excel('seed' + str(seed) + 'drugload' + '_loocv_' + 'model_comparison.xlsx', index=False, engine='openpyxl')
    
    ## save the results of each methods
    formula_nums = len(pd.read_excel(root + '\\Data sets\\Drug loading data sets.xlsx'))
    formula_list = np.arange(formula_nums) + 1
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
        os.chdir(root + '\\final_result_nospan_excel\\othermodel\\loocv\\details\\drugload')
        result_dataframe.to_excel('seed' + str(seed) + 'drugload' + '_loocv_' + setting + '.xlsx', index=False, engine='openpyxl')
    return

if __name__ == '__main__':
    
    root = 'D:\\xxx\\xxx' ## modify your root path
    main_function(seed=18, task='vitro', Additives=False, root = root, aug = True)
    save2excel(task='vitro', seed = 18, root = root, aug = True)
    
    main_function(seed=18, task='vivo', Additives=False, root = root, aug = False)
    save2excel(task='vivo', seed = 18, root = root, aug = False)
    
    main_function_drugload(seed = 18, Additives = False, root = root)
    save2excel_drugload(seed = 18, root= root)
    