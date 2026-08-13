import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message="The 'delay_after_gen' parameter is deprecated starting from PyGAD 3.3.0.*", category=UserWarning, module='pygad')
warnings.filterwarnings("ignore", message=".*No further splits with positive gain, best gain: -inf.*")
import numpy as np
import os 
import pandas as pd
import random
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score
from scipy import stats
import pygad
import copy

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

def dataset_augmentation(train_formula_list,train_data_mat, train_label, mode = 'noaug'):
    if mode == 'default':
        train_data_mat, train_label = mixup(train_data_mat, train_label) ## traditional mixup
    elif mode == 'time_linear':
        train_data_mat, train_label = mixup_time(train_formula_list, train_data_mat, train_label) ## our mixup for the drug release prediction
    return train_data_mat, train_label

def demo_for_vivomodel_training():
    os.chdir(root + '\\Data sets')
    df = pd.read_excel('literature-derived subset.xlsx', sheet_name='In vivo Release')
    data_mat = df.to_numpy()
    
    train_data_mat = data_mat[:,0:-2]
    train_plasma_concentration = data_mat[:,-2]
    train_accumulated_release = data_mat[:,-1]
    
    reg_plasma = LGBMRegressor(force_col_wise=True, boosting_type = 'dart', learning_rate = 0.1, n_estimators = 500)
    
    monotone_constraints = np.zeros(train_data_mat.shape[1])
    monotone_constraints[-1] = 1
    reg_accumulated = LGBMRegressor(force_col_wise=True, boosting_type = 'dart', 
                                    
                                    learning_rate = 0.1, n_estimators = 500,
                                    
                                    monotone_constraints=monotone_constraints)
    
    reg_plasma.fit(train_data_mat, train_plasma_concentration)
    
    reg_accumulated.fit(train_data_mat, train_accumulated_release)
    
    return reg_plasma, reg_accumulated

def dataloading_vitro(root):
    os.chdir(root + '\\Data sets')
    df = pd.read_excel('literature-derived subset.xlsx', sheet_name='In vitro Release')

    column_name = df.columns.tolist()[0:-1]
    column_name[-1] = 'Time' 
    data_mat = df.to_numpy()
    
    train_release_label = []
    train_feature_list = []
    train_formula_list = []
    
    for formula in range(64):
        train_feature_list.append(data_mat[formula*9:(formula+1)*9,:-1])
        train_release_label.append(data_mat[formula*9:(formula+1)*9,-1:])
        for _ in range(9):
            train_formula_list.append(formula)
        
    train_data_mat = np.vstack(train_feature_list).astype('float32')
    train_label = np.vstack(train_release_label).astype('float32')/100
    
    train_data_mat, train_label = dataset_augmentation(train_formula_list, train_data_mat, train_label, mode = 'time_linear')
    dataset = {'train_data_mat':nan_converter(train_data_mat), 'train_label': train_label.squeeze(),
               'train_formula_list': train_formula_list,'column_name': column_name}
    print('dataset loading success')
    return dataset

def nan_converter(matrix, mode = 'zero'):
    if mode == 'mean':
        column_means = np.nanmean(matrix, axis=0)
        matrix = np.nan_to_num(matrix, nan=column_means)
    else:
        matrix = np.nan_to_num(matrix)
    return matrix

def dataloading_drugload(root):
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
    
    dataset = {'train_data_mat':nan_converter(train_data_mat), 'train_label': train_label.squeeze(),
                'column_name': column_name}
    print('dataset loading success')
    return dataset

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mae_loss(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def enforce_monotonicity(predictions):
    for i in range(1, len(predictions)):
        predictions[i] = max(predictions[i], predictions[i - 1])
    return predictions

#https://lightgbm.readthedocs.io/en/latest/Parameters.html
def vitro_forward_model(root, seed=18, setting = 'optim'):
    setup_seed(seed)
    dataset = dataloading_vitro(root)
        
    monotone_constraints = np.zeros(dataset['train_data_mat'].shape[1])
    monotone_constraints[-1] = 1

    if setting == 'optim':
        reg = LGBMRegressor(force_col_wise=True,monotone_constraints=monotone_constraints, 
                        boosting_type = 'dart', learning_rate = 0.1, n_estimators = 500)
    elif setting == 'default':
        reg = LGBMRegressor(force_col_wise=True,monotone_constraints=monotone_constraints)
    else:
        raise ValueError("setting can only be dafault or optim")
    reg.fit(dataset['train_data_mat'], dataset['train_label'])

    return reg

def drug_forward_model(root, seed=18, setting = 'optim'):
    setup_seed(seed)
    dataset = dataloading_drugload(root)
        
    if setting == 'optim':
        reg = LGBMRegressor(force_col_wise=True, boosting_type = 'gbdt', learning_rate = 0.1, n_estimators = 300)
    elif setting == 'default':
        reg = LGBMRegressor(force_col_wise=True)
    else:
        raise ValueError("setting can only be dafault or optim")
    train_data_mat, train_label = dataset['train_data_mat'], dataset['train_label']
    
    reg.fit(train_data_mat, train_label)

    return reg

def get_range(min_value, max_value, length):
    return np.arange(min_value, max_value, (max_value-min_value)/length)

def load_formula_search_accelerate(root, formula_no = 1):
    os.chdir(root + '\\Data sets')
    df = pd.read_excel('RIS_Design_Task.xlsx')
    df = df[df['Formulation.no'] == formula_no]

    df = df.drop(['Formulation.no', 'Drug_name'] ,axis='columns')

    feature_dict_list = []
    Thl = df['Theoretical drug loading'].values[0]
    length_imp = 100
    length_noimp = 50
    if df['Drug_4'].values[0] > 0:
        for enc_method in [2,3]:
            feature_dict = {}
            feature_dict['Drug_1'] = df['Drug_1'].values[0]
            feature_dict['Drug_2'] = df['Drug_2'].values[0]
            feature_dict['Drug_3'] = df['Drug_3'].values[0]
            feature_dict['Drug_4'] = df['Drug_4'].values[0]
            feature_dict['Drug_5'] = df['Drug_5'].values[0]
            feature_dict['Encapsulation methods'] = enc_method
            
            feature_dict['Prescriptions_3'] = get_range(20, 300, length_imp)
            if enc_method == 3:
                feature_dict['Prescriptions_4'] = get_range(2000, 7000, length_noimp)
            else:
                feature_dict['Prescriptions_4'] = get_range(7000, 25000, length_noimp)
            feature_dict['Prescriptions_5'] = get_range(30, 200, length_noimp)
            feature_dict['Prescriptions_6'] = 0
            feature_dict['Prescriptions_7'] = get_range(1, 100, length_noimp)
            feature_dict['Prescriptions_8'] = get_range(0.5, 2, length_noimp)
            feature_dict['Prescriptions_9'] = get_range(0.5, 2, length_noimp)
            feature_dict['solidification method'] = np.arange(1, 3, 1)
            feature_dict['microspheres  properties_1'] = get_range(10, 50, length_imp)
            feature_dict['microspheres  properties_2'] = Thl
            feature_dict['Accessories_1'] = get_range(0.01, 100, length_noimp)
            feature_dict['Accessories_2'] = get_range(1000, 200000, length_imp)
            feature_dict['Accessories_3'] = np.arange(1, 4, 1) 
            feature_dict_list.append(feature_dict)

    else:
        for enc_method in [1,2]:
            if enc_method == 2:
                feature_dict = {}
                feature_dict['Drug_1'] = df['Drug_1'].values[0]
                feature_dict['Drug_2'] = df['Drug_2'].values[0]
                feature_dict['Drug_3'] = df['Drug_3'].values[0]
                feature_dict['Drug_4'] = df['Drug_4'].values[0]
                feature_dict['Drug_5'] = df['Drug_5'].values[0]
                feature_dict['Encapsulation methods'] = enc_method
                
                
                feature_dict['Prescriptions_3'] = get_range(20, 300, length_imp)
                
                feature_dict['Prescriptions_4'] = get_range(7000, 25000, length_noimp)
                
                feature_dict['Prescriptions_5'] = get_range(30, 200, length_noimp)
                feature_dict['Prescriptions_6'] = 0
                feature_dict['Prescriptions_7'] = get_range(1, 100, length_noimp)
                feature_dict['Prescriptions_8'] = get_range(0.5, 2, length_noimp)
                feature_dict['Prescriptions_9'] = get_range(0.5, 2, length_noimp)
                feature_dict['solidification method'] = np.arange(1, 3, 1)
                feature_dict['microspheres  properties_1'] = get_range(10, 50, length_imp)
                feature_dict['microspheres  properties_2'] = Thl
                feature_dict['Accessories_1'] = get_range(0.01, 100, length_noimp)
                feature_dict['Accessories_2'] = get_range(1000, 200000, length_imp)
                feature_dict['Accessories_3'] = np.arange(1, 4, 1) 
            else:
                feature_dict = {}
                feature_dict['Drug_1'] = df['Drug_1'].values[0]
                feature_dict['Drug_2'] = df['Drug_2'].values[0]
                feature_dict['Drug_3'] = df['Drug_3'].values[0]
                feature_dict['Drug_4'] = df['Drug_4'].values[0]
                feature_dict['Drug_5'] = df['Drug_5'].values[0]
                feature_dict['Encapsulation methods'] = enc_method
                
                
                feature_dict['Prescriptions_3'] = get_range(20, 300, length_imp)
                
                feature_dict['Prescriptions_4'] = get_range(7000, 25000, length_noimp)
                
                feature_dict['Prescriptions_5'] = get_range(30, 200, length_noimp)
                feature_dict['Prescriptions_6'] = get_range(1, 20, length_noimp)
                feature_dict['Prescriptions_7'] = get_range(1, 100, length_noimp)
                feature_dict['Prescriptions_8'] = get_range(0.5, 2, length_noimp)
                feature_dict['Prescriptions_9'] = get_range(0.5, 2, length_noimp)
                feature_dict['solidification method'] = np.arange(1, 3, 1) 
                feature_dict['microspheres  properties_1'] = get_range(10, 50, length_imp)
                feature_dict['microspheres  properties_2'] = Thl
                feature_dict['Accessories_1'] = get_range(0.01, 100, length_noimp)
                feature_dict['Accessories_2'] = get_range(1000, 200000, length_imp)
                feature_dict['Accessories_3'] = np.arange(1, 4, 1)
            feature_dict_list.append(feature_dict)
    return feature_dict_list

def get_full_solution(solution, column_name, column_name_full):
    condidate_formula_dict = {}
    for name in column_name:
        condidate_formula_dict[name] = solution[column_name.index(name)]
    enc_method = solution[column_name.index('Encapsulation methods')]
    if enc_method == 1:
        Thl = solution[column_name.index('microspheres  properties_2')]/100
        P_3 = solution[column_name.index('Prescriptions_3')]
        P_6 = solution[column_name.index('Prescriptions_6')]
        condidate_formula_dict['Prescriptions_1'] = (Thl * P_3 * P_6)/(1-Thl)
        condidate_formula_dict['Prescriptions_2'] = 0
    else:
        Thl = solution[column_name.index('microspheres  properties_2')]/100
        P_3 = solution[column_name.index('Prescriptions_3')]
        condidate_formula_dict['Prescriptions_2'] = (Thl * P_3)/(1-Thl)
        condidate_formula_dict['Prescriptions_1'] = 0
    solution = np.zeros(len(column_name_full))
    for i in range(len(column_name_full)):
        name = column_name_full[i]
        solution[i] = condidate_formula_dict[name]
    return solution

def decide_efficiency_thres(feature_dict_list, drug_model, column_name, column_name_full):
    threshold = []
    for feature_dict in feature_dict_list:
        efficiency_list = []
        for _ in range(20000):
            condidate_formula_dict = {}    
            for i in range(len(column_name)):
                name = column_name[i]
                if isinstance(feature_dict[name], np.ndarray):   
                    condidate_formula_dict[name] = random.choice(feature_dict[name].tolist())
                else:
                    condidate_formula_dict[name] = feature_dict[name]
            enc_method = condidate_formula_dict['Encapsulation methods']
            if enc_method == 1:
                Thl = condidate_formula_dict['microspheres  properties_2']/100
                P_3 = condidate_formula_dict['Prescriptions_3']
                P_6 = condidate_formula_dict['Prescriptions_6']
                condidate_formula_dict['Prescriptions_1'] = (Thl * P_3 * P_6)/(1-Thl)
                condidate_formula_dict['Prescriptions_2'] = 0
            else:
                Thl = condidate_formula_dict['microspheres  properties_2']/100
                P_3 = condidate_formula_dict['Prescriptions_3']
                condidate_formula_dict['Prescriptions_2'] = (Thl * P_3)/(1-Thl)
                condidate_formula_dict['Prescriptions_1'] = 0
            condidate_formula = np.zeros(len(column_name_full))
            for i in range(len(column_name_full)):
                name = column_name_full[i]
                condidate_formula[i] = condidate_formula_dict[name]
            efficiency = np.clip(drug_model.predict(np.array(condidate_formula).reshape(1, len(condidate_formula))), 0, 1)
            efficiency_list.append(efficiency)
        efficiency_list = np.stack(efficiency_list)
        threshold.append(np.percentile(efficiency_list, 80))
    return threshold

def search_formula_drugloading_accelerate(root, genetics_parameter,drug_model, vitro_model, formula_num = 30, formula_no = 1):
    dataset = dataloading_drugload(root)
    column_name = copy.deepcopy(dataset['column_name'])
    column_name_full = copy.deepcopy(column_name)
    column_name.remove('Prescriptions_1')
    column_name.remove('Prescriptions_2')
    feature_dict_list = load_formula_search_accelerate(root, formula_no = formula_no)
    feature_dict = feature_dict_list[0]
    possible_formula = []
    
    threshold = decide_efficiency_thres(feature_dict_list, drug_model, column_name, column_name_full)
        
    for (feature_dict, thres) in zip(feature_dict_list, threshold):
        #feature_dict = feature_dict_list[0]
        possible_formula_1 = []
        while len(possible_formula_1) < int(genetics_parameter['initial_population']/2):
            
            condidate_formula_dict = {}    
            for i in range(len(column_name)):
                name = column_name[i]
                if isinstance(feature_dict[name], np.ndarray):   
                    condidate_formula_dict[name] = random.choice(feature_dict[name].tolist())
                else:
                    condidate_formula_dict[name] = feature_dict[name]
            enc_method = condidate_formula_dict['Encapsulation methods']
            if enc_method == 1:
                Thl = condidate_formula_dict['microspheres  properties_2']/100
                P_3 = condidate_formula_dict['Prescriptions_3']
                P_6 = condidate_formula_dict['Prescriptions_6']
                condidate_formula_dict['Prescriptions_1'] = (Thl * P_3 * P_6)/(1-Thl)
                condidate_formula_dict['Prescriptions_2'] = 0
            else:
                Thl = condidate_formula_dict['microspheres  properties_2']/100
                P_3 = condidate_formula_dict['Prescriptions_3']
                condidate_formula_dict['Prescriptions_2'] = (Thl * P_3)/(1-Thl)
                condidate_formula_dict['Prescriptions_1'] = 0
            
            condidate_formula = np.zeros(len(column_name_full))
            for i in range(len(column_name_full)):
                name = column_name_full[i]
                condidate_formula[i] = condidate_formula_dict[name]
            efficiency = np.clip(drug_model.predict(np.array(condidate_formula).reshape(1, len(condidate_formula))), 0 , 1)
            if efficiency  > thres:
                condidate_formula = np.zeros(len(column_name))
                for i in range(len(column_name)):
                    name = column_name[i]
                    condidate_formula[i] = condidate_formula_dict[name]
                possible_formula_1.append(condidate_formula)
        possible_formula.extend(possible_formula_1)
    possible_formula = np.vstack(possible_formula)
    
    os.chdir(root + '\\Data sets')
    df = pd.read_excel('RIS_Design_Task.xlsx')
    Time, label = np.array(df['Release_2'])[9*(formula_no-1):9*formula_no], np.array(df['Release_3'])[9*(formula_no-1):9*formula_no].squeeze()/100

    def fitness_function(ga_instance, solution, solution_idx):
        enc_method = solution[column_name.index('Encapsulation methods')]
        Drug_4 = solution[column_name.index('Drug_4')]
        if Drug_4 > 0:
            index = enc_method - 2 # enc_method = 2 or 3
        else:
            index = enc_method - 1 # enc_method = 1 or 2
            
        solution = get_full_solution(solution, column_name, column_name_full)
        solution_efficiency = np.clip(drug_model.predict(solution.reshape(1, len(solution))), 0, 1)
        if solution_efficiency < threshold[int(index)]:
            return 0
        else:
            solution_actual_drugload = solution_efficiency * feature_dict['microspheres  properties_2']
            
           # print('solution_actual_drugload:', solution_actual_drugload, 'solution_efficiency:', solution_efficiency, 'Thl:', feature_dict['microspheres  properties_2'])
            solution = np.concatenate((solution, solution_actual_drugload, 100 * solution_efficiency, np.ones(1) + 1))
            solution = np.tile(solution, (len(Time), 1))
            solution = np.column_stack((solution, Time))
            output = vitro_model.predict(solution)
            trimed_prediction = np.clip(output,0,1)
            fitness = 1/mae_loss(trimed_prediction, label)
            return fitness
    
    def solution_purify(final_population, final_fitness):
        puried_population = np.unique(np.column_stack((final_population, final_fitness)), axis=0)
        return puried_population[:,:-1], puried_population[:,-1]
    
    initial_population = possible_formula

    top_solutions_list, mae_list = [], []
    
    all_population_list, all_fitness_list = [] , []
    for type_f in range(2):
        feature_dict = feature_dict_list[type_f]
        
        gene_space = []
        for i in range(len(column_name)):
            name = column_name[i]
            gene_space.append(feature_dict[name])
            
        initial_num = len(possible_formula)//2
        initial_population = possible_formula[type_f*initial_num:(type_f+1)*initial_num,:]
        ga_instance = pygad.GA(num_generations=genetics_parameter['num_generations'],
                            num_parents_mating=genetics_parameter['num_parents_mating'],
                            fitness_func=fitness_function,
                            crossover_type=genetics_parameter['crossover_type'],
                            num_genes=initial_population.shape[1],
                            mutation_probability=genetics_parameter['mutation_probability'],
                            mutation_type=genetics_parameter['mutation_type'],
                            crossover_probability=genetics_parameter['crossover_probability'],
                            initial_population=initial_population,
                            gene_space = gene_space,
                            save_solutions=True)
        ga_instance.run()
        
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        #print("Best Solution: {solution}".format(solution=solution))
        print("Best Solution MAE: {solution_mae}".format(solution_mae=1/solution_fitness))
        
        final_population = ga_instance.population
        final_fitness = ga_instance.last_generation_fitness

        final_population, final_fitness = solution_purify(final_population, final_fitness)
        
        top_indices = np.argsort(final_fitness)[::-1][:min(len(final_population),30)]
        top_solutions = final_population[top_indices]
        top_fitness = final_fitness[top_indices]
        #print("Top 30 solutions after last generation:")
        for idx, solution in enumerate(top_solutions):
            #print("Solution {}: {} with MAE: {}".format(idx+1, solution, 1/top_fitness[idx]))
            top_solutions_list.append(solution)
            mae_list.append(1/top_fitness[idx])
            
        all_population = ga_instance.solutions
        all_fitness = ga_instance.solutions_fitness
        all_population, all_fitness = solution_purify(all_population, all_fitness)
        all_population_list.append(all_population)
        all_fitness_list.append(all_fitness)
    
    
    all_population = np.vstack(all_population_list).tolist()
    all_fitness = np.concatenate((all_fitness_list[0][:,np.newaxis], all_fitness_list[1][:,np.newaxis]))
    for i in range(len(all_population)):
        all_population[i] = get_full_solution(all_population[i], column_name, column_name_full)
    all_population = np.vstack(all_population)
    #plot_fitness_shap(root, column_name_full, all_population, all_fitness, genetics_parameter, formula_no = formula_no, drug_name = Drug_name)

    for i in range(len(top_solutions_list)):
        top_solutions_list[i] = get_full_solution(top_solutions_list[i], column_name, column_name_full)
        
    top_solutions_list = np.vstack(top_solutions_list)
    efficiency = np.clip(drug_model.predict(top_solutions_list), 0, 1)
    actual_drugload = feature_dict['microspheres  properties_2'] * efficiency
    top_solutions_list = np.column_stack((top_solutions_list, actual_drugload, 100 * efficiency, np.ones_like(efficiency) + 1))
    error_mat = np.zeros((len(top_solutions_list), 3))
    curve_mat = np.zeros((len(top_solutions_list), len(Time)))
    for f in range(len(top_solutions_list)):
        current_formula = top_solutions_list[f,:]
        collect_input = np.tile(current_formula, (len(Time), 1))
        collect_input = np.column_stack((collect_input, Time))
        output = vitro_model.predict(collect_input)
        trimed_prediction = np.clip(output,0,1)
        mean_absolute_loss = mae_loss(trimed_prediction, label)
        r2 = r2_score(trimed_prediction, label)
        pearson = stats.pearsonr(trimed_prediction, label).statistic
        error_mat[f,:] = [mean_absolute_loss, r2, pearson]
        curve_mat[f,:] = trimed_prediction
    return top_solutions_list, error_mat, curve_mat

def checker(selected_formula_list, seed=18):    
    for i in range(len(selected_formula_list)):
        if float(selected_formula_list[i]['Drug_4'].values[0]) <= 0:
            P_3 = np.float64(selected_formula_list[i]['Prescriptions_3'].values)
            
            Thl = np.float64(selected_formula_list[i]['microspheres  properties_2'].values)/100
            
            P_6 = np.float64(selected_formula_list[i]['Prescriptions_6'].values)
            
            P_1 = (Thl * P_3 * P_6)/(1-Thl) - np.float64(selected_formula_list[i]['Prescriptions_1'].values)
            
            P_1 = P_1[0:30]
            
            P_2 = (Thl * P_3)/(1-Thl) - np.float64(selected_formula_list[i]['Prescriptions_2'].values)
            
            P_2 = P_2[30:]
            
            print(np.sum(P_1) + np.sum(P_2))
            
            return np.sum(P_1) + np.sum(P_2)
        else:
            P_3 = np.float64(selected_formula_list[i]['Prescriptions_3'].values)
            
            Thl = np.float64(selected_formula_list[i]['microspheres  properties_2'].values)/100
            
            P_6 = np.float64(selected_formula_list[i]['Prescriptions_6'].values)
            
            P_1 = (Thl * P_3 * P_6)/(1-Thl) - np.float64(selected_formula_list[i]['Prescriptions_1'].values)
            
            P_2 = (Thl * P_3)/(1-Thl) - np.float64(selected_formula_list[i]['Prescriptions_2'].values)
            
            print(np.sum(P_1) + np.sum(P_2))
            
            return np.sum(P_1) + np.sum(P_2)

def main_function_acc(genetics_parameter, seed=18, formula_no = 1, root = 'D:\\xxx\\xxx', setting = 'optim'):
    vitro_model = vitro_forward_model(root, seed=seed, setting = setting)
    drug_model = drug_forward_model(root, seed=seed, setting = setting)
    
    selected_formula_acc, error_acc, curve_mat_acc = search_formula_drugloading_accelerate(root, genetics_parameter, drug_model, 
                                                                                       vitro_model, formula_num = 60
                                                                                       , formula_no = formula_no)
    
    vitro_dataset = dataloading_vitro(root)
    
    os.chdir(root + '\\Data sets')
    df = pd.read_excel('RIS_Design_Task.xlsx')
    Time_acc, label_acc = np.array(df['Release_2'])[9*(formula_no-1):9*formula_no], np.array(df['Release_3'])[9*(formula_no-1):9*formula_no].squeeze()/100
    
    column_name = vitro_dataset['column_name'][0:-1]
    column_name.append('MAE')
    column_name.append('R2')
    column_name.append('Pearson')
    parameter_list = list(genetics_parameter.keys())
    parameter_mat = []
    for name in parameter_list:
        column_name.append(name)
        parameter_mat.append(genetics_parameter[name])
        
    parameter_mat = np.array(parameter_mat).reshape(1,len(parameter_mat)) 
    parameter_mat = np.tile(parameter_mat, (len(error_acc),1))
    column_name_acc = copy.deepcopy(column_name)
    column_name_acc.extend(Time_acc)
    column_name_acc.extend(Time_acc)
    selected_formula_acc = pd.DataFrame(data=np.column_stack((selected_formula_acc, error_acc, parameter_mat, curve_mat_acc, 
                                                          np.tile(label_acc, (len(error_acc), 1)))), columns=column_name_acc)
    selected_formula_list = [selected_formula_acc]
    flag = checker(selected_formula_list, seed=18)
    if flag != 0:
        raise ValueError("Checker Warning!!!!!!!!!")
    return selected_formula_acc

def get_best_solution(formula_no, release_type):
    num_parents_list = [4, 8, 16]
    mutation_probability_list = [0.01, 0.05, 0.10]
    solution_list = []
    for num_parents in num_parents_list:
        for mutation_probability in mutation_probability_list:
            file_path = root + '\\condidate_solution\\gridsearch_result_final_flex_fitness\\' + 'parents_' + str(num_parents) + '_mutation_' + str(int(100*mutation_probability))
            os.chdir(file_path)
            file_name = release_type + '_solution_formula_' + str(formula_no) +'.xlsx'
            solution = pd.read_excel(file_name)
            solution_list.append(solution)
            
    
    final_solution = pd.concat(solution_list, ignore_index=True)
    columns_to_check = final_solution.columns[:25] 
    final_solution = final_solution.drop_duplicates(subset=columns_to_check)
    mae_list = np.array(final_solution['MAE'])
    
    selected_index = np.argsort(mae_list)[0:60]
    selected_solution = final_solution.loc[selected_index]
    
    del file_path
    
    final_file_path = root + '\\condidate_solution\\gridsearch_result_final_flex_fitness\\final_solutions'
    if not os.path.exists(final_file_path):
        print('creating new file path', final_file_path)
        os.makedirs(final_file_path)
    os.chdir(final_file_path)
    
    selected_solution.to_excel(release_type + '_solution_formula_' + str(formula_no) +'_best.xlsx', index=False, engine='openpyxl')

def get_parameter_std(solution):
    measure_list = solution.columns[6:20].tolist()
    measure_list.remove('Prescriptions_1')
    measure_list.remove('Prescriptions_2')
    measure_list.remove('Prescriptions_6')
    measure_list.remove('solidification method')
    std_list = []  
    for measure in measure_list:
        data = np.array(solution[measure])
        std_list.append(np.std(data))     
    return std_list, measure_list

def sepe_mae(prediction_mat, groundtruth_mat):
    mae_mat = np.zeros((len(prediction_mat),3))
    for i in range(len(mae_mat)):
        mae_mat[i, 0] = mae_loss(prediction_mat[i,0:3], groundtruth_mat[i,0:3])
        mae_mat[i, 1] = mae_loss(prediction_mat[i,3:6], groundtruth_mat[i,3:6])
        mae_mat[i, 2] = mae_loss(prediction_mat[i,6:9], groundtruth_mat[i,6:9])
    return mae_mat

def get_best_solution_shap(formula_no, release_type):
    from scipy.stats import rankdata
    num_parents_list = [4, 8, 16]
    mutation_probability_list = [0.01, 0.05, 0.10]
    solution_list = []
    all_std_list = []
    for num_parents in num_parents_list:
        for mutation_probability in mutation_probability_list:
            file_path = root + '\\condidate_solution\\' + 'parents_' + str(num_parents) + '_mutation_' + str(int(100*mutation_probability))
            os.chdir(file_path)
            file_name = release_type + '_solution_formula_' + str(formula_no) +'.xlsx'
            solution = pd.read_excel(file_name)
            solution_list.append(solution)
            std_list, measure_list = get_parameter_std(solution)
            all_std_list.append(std_list)
    all_std_list = np.vstack(all_std_list)
    rank_list = np.zeros_like(all_std_list)
    for i in range(all_std_list.shape[1]):
        rank_list[:,i] =  rankdata(-all_std_list[:,i], method='average')
    weight = [0.01, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.03]
    score = np.sum(rank_list*weight, axis = 1)
    
    best_parameter = np.argmax(score)
    selected_solution = solution_list[best_parameter]

    prediction_mat = np.array(selected_solution.iloc[:, len(selected_solution.columns) - 18: len(selected_solution.columns) - 9])
    groundtruth_mat = np.array(selected_solution.iloc[:, len(selected_solution.columns) - 9: len(selected_solution.columns)])
    
    mae_mat = sepe_mae(prediction_mat, groundtruth_mat)
    idx = selected_solution.columns.get_loc('MAE') + 1
    selected_solution.insert(idx, 'MAE_1', mae_mat[:, 0])
    selected_solution.insert(idx + 1, 'MAE_2', mae_mat[:, 1])
    selected_solution.insert(idx + 2, 'MAE_3', mae_mat[:, 2])
    final_file_path = root + '\\condidate_solution\\final_solutions_shap'
    
    print(final_file_path)
    if not os.path.exists(final_file_path):
        print('creating new file path', final_file_path)
        os.makedirs(final_file_path)
    os.chdir(final_file_path)
    selected_solution['microspheres  properties_4'] = selected_solution['microspheres  properties_4'].clip(lower = 0, upper = 100)
    selected_solution['microspheres  properties_3'] = selected_solution['microspheres  properties_2'].values[0] * selected_solution['microspheres  properties_4']/100
    selected_solution.to_excel(release_type + '_solution_formula_' + str(formula_no) +'_best.xlsx', index=False, engine='openpyxl')


if __name__ == '__main__':

    root = 'D:\\EMAF_Demo' ## modify your root path

    num_parents_list = [4, 8, 16]
    mutation_probability_list = [0.01, 0.05, 0.10]
    for num_parents in num_parents_list:
        for mutation_probability in mutation_probability_list:
            print('current number of parents:', num_parents)
            print('current mutation probability:', mutation_probability)
            genetics_parameter = {'crossover_type':'single_point', 'crossover_probability': 0.8, 'mutation_probability': mutation_probability,
                              'num_parents_mating': num_parents, 'num_generations': 100, 'mutation_type': 'random', 'initial_population': 1000}
            file_path = root + '\\condidate_solution\\' + 'parents_' + str(num_parents) + '_mutation_' + str(int(100*mutation_probability))
            formula_no = 1
            print('current formula: ', formula_no)
            selected_formula_acc = main_function_acc(genetics_parameter, seed=18, formula_no = formula_no, root = root, setting = 'optim')
            if not os.path.exists(file_path):
                print('creating new file path', file_path)
                os.makedirs(file_path)
            os.chdir(file_path)
            selected_formula_acc.to_excel('acc_solution_formula_'+ str(formula_no) +'.xlsx', index=False, engine='openpyxl')
                
    get_best_solution_shap(formula_no = 1, release_type = 'acc')
        
            
            
            
            
    
        
    
    
