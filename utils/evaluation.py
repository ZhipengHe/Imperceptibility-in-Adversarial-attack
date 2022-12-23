import numpy as np
import pandas as pd
from enum import Enum
from typing import List
from utils.preprocessing import DfInfo
from scipy.spatial import distance
from sklearn.metrics import accuracy_score


class InstanceType(Enum):
    ScaledInput = "scaled_input_"
    ScaledAdv = "scaled_adv_"
    OriginInput = "origin_input_"
    OriginAdv = "origin_adv_"

'''
Evaluation Functions.
'''

def get_Linf(**kwargs):
    input_array = np.array(kwargs['input'])
    adv_array = np.array(kwargs['adv'])

    return np.linalg.norm(input_array - adv_array, axis=1, ord=np.inf)

def get_L2(**kwargs):
    input_array = np.array(kwargs['input'])
    adv_array = np.array(kwargs['adv'])

    return np.linalg.norm(input_array - adv_array, axis=1, ord=2)


def get_L1(**kwargs):
    input_array = np.array(kwargs['input'])
    adv_array = np.array(kwargs['adv'])

    return np.linalg.norm(input_array - adv_array, axis=1, ord=1)


def get_sparsity(**kwargs):

    # should remove the target column first.
    
    input_df = kwargs['not_dummy_input']
    adv_df = kwargs['not_dummy_adv']

    input_array = np.array(input_df)
    adv_array = np.array(adv_df)

    return np.equal(input_array, adv_array).astype(int).sum(axis=1)
    # return sum(x != y for x, y in zip(input_array, adv_array))

    



def get_realisitic(**kwargs):
    '''
    Checking if the numerical columns are in the range of [0, 1].
    '''
    df_info: DfInfo = kwargs['df_info']
    adv_num_array = np.array(kwargs['adv'][df_info.numerical_cols])
    return np.all(np.logical_and(adv_num_array >= 0, adv_num_array <= 1 ), axis=1)

def get_mad(**kwargs,):
    '''
    Get Mean Absolute Deviation Distance between input and adv. 
    '''

    eps = 1e-8

    input_df = kwargs['input']
    adv_df = kwargs['adv']
    df_info = kwargs['df_info']

    ohe_cat_cols = df_info.get_ohe_cat_cols()
    ohe_num_cols = df_info.get_ohe_num_cols()

    numerical_mads = df_info.get_numerical_mads()

    mad_df = pd.DataFrame({}, columns= df_info.ohe_feature_names)
    mad_df[ohe_cat_cols] = (input_df[ohe_cat_cols] != adv_df[ohe_cat_cols]).astype(int)
    for num_col in ohe_num_cols: 
        mad_df[num_col] = abs(adv_df[num_col] - input_df[num_col]) / (numerical_mads[num_col] + eps)

    if len(ohe_cat_cols) > 0 and len(ohe_num_cols) > 0:
        return (mad_df[ohe_num_cols].mean(axis=1) + mad_df[ohe_cat_cols].mean(axis=1)).tolist()
        # return mad_df.mean(axis=1).tolist()
        # return mad_df.sum(axis=1).tolist() # <=(weird, may be wrong) actually from (https://github.com/ADMAntwerp/CounterfactualBenchmark/blob/9dbf6a9e604ce1a2a0ddfb15025718f2e0effb0a/frameworks/LORE/distance_functions.py) 

    elif len(ohe_num_cols) > 0:
        return mad_df[ohe_num_cols].mean(axis=1).tolist()
    elif len(ohe_cat_cols) > 0:
        return mad_df[ohe_cat_cols].mean(axis=1).tolist()
    else:
        raise Exception("No columns provided for MAD.")

    # return (mad_df[ohe_num_cols].mean(axis=1) + mad_df[ohe_cat_cols].mean(axis=1)).tolist()


def get_mahalanobis(**kwargs,):
    '''
    Get Mahalanobis distance between input and adv.
    '''
    input_df = kwargs['input']
    adv_df = kwargs['adv']
    df_info = kwargs['df_info']

    VI_m = df_info.dummy_df[df_info.ohe_feature_names].cov().to_numpy()

    return [distance.mahalanobis(input_df[df_info.ohe_feature_names].iloc[i].to_numpy(),
                                adv_df[df_info.ohe_feature_names].iloc[i].to_numpy(),
                                VI_m) for i in range(len(input_df))]


def get_perturbation_sensitivity(**kwargs,):

    adv_df = kwargs['adv']
    df_info = kwargs['df_info']

    std = df_info.dummy_df[df_info.ohe_feature_names].std().to_numpy()
    adv_std = adv_df[df_info.ohe_feature_names].std().to_numpy()

    return (1.0 / adv_std)


def get_neighbour_distance(**kwargs,):

    adv_df = kwargs['adv']
    df_info = kwargs['df_info']
    
    adv_arr = adv_df[df_info.ohe_feature_names].to_numpy()
    dataset = df_info.dummy_df[df_info.ohe_feature_names].to_numpy()

    return distance.cdist(adv_arr, dataset, 'minkowski', p=2).min(axis=1).tolist()

    

class EvaluationMatrix(Enum):
    '''
    All evaluation function should be registed here.
    '''
    L1 = "eval_L1"
    L2 = "eval_L2"
    Linf = "eval_Linf"
    Sparsity = "eval_Sparsity"
    Realistic = "eval_Realistic"
    MAD = "eval_MAD"
    Mahalanobis = "eval_Mahalanobis"
    Perturbation_Sensitivity = "eval_Perturbation_Sensitivity"
    Neighbour_Distance = "eval_Neighbour_Distance"

evaluation_name_to_func = {
    # All evaluation function should be registed here as well
    EvaluationMatrix.L1: get_L1,
    EvaluationMatrix.L2: get_L2,
    EvaluationMatrix.Linf: get_Linf,
    EvaluationMatrix.Sparsity: get_sparsity,
    EvaluationMatrix.Realistic: get_realisitic,
    EvaluationMatrix.MAD: get_mad,
    EvaluationMatrix.Mahalanobis: get_mahalanobis,
    EvaluationMatrix.Perturbation_Sensitivity: get_perturbation_sensitivity,
    EvaluationMatrix.Neighbour_Distance: get_neighbour_distance,
}


'''
Util functions.
'''

def get_dummy_version(input_df: pd.DataFrame, df_info: DfInfo):
    '''
    Transform the categorical data to ohe format. (Better for calculating the distance)
    '''

    def get_string_dummy_value(x):
        if isinstance(x, float) and x==x:
            x = int(x)

        return str(x)

    number_of_instances = len(input_df)

    init_row = {}
    for k in df_info.ohe_feature_names:
        init_row[k] = 0

    init_df = pd.DataFrame([init_row]*number_of_instances,
                           columns=df_info.ohe_feature_names)

    for k, v in df_info.cat_to_ohe_cat.items():
        for ohe_f in v:
            init_df[ohe_f] = input_df[k].apply(
                lambda x: 1 if ohe_f.endswith(get_string_dummy_value(x)) else 0).tolist()

    for col in df_info.numerical_cols:
        init_df[col] = input_df[col].tolist()

    return init_df


def get_type_instance(df: pd.DataFrame, instance_type: InstanceType, with_original_name: bool = True):
    '''
    Get certain type of instance in the result data frame. Check `InstanceType` to know all types.
    '''

    df = df.copy(deep=True)
    return_df = df[[
        col for col in df.columns if col.startswith(instance_type.value)]]

    if with_original_name:
        return_df.columns = [col.replace(
            instance_type.value, "") for col in return_df.columns]

    return return_df


def prepare_evaluation_dict(result_df: pd.DataFrame, df_info: DfInfo):
    '''
    Prepare the information needed to perform evaluation.
    '''

    return {
        "input": get_dummy_version(get_type_instance(result_df, InstanceType.ScaledInput), df_info),
        "adv": get_dummy_version(get_type_instance(result_df, InstanceType.ScaledAdv), df_info),
        "not_dummy_input": get_type_instance(result_df, InstanceType.ScaledInput).drop(labels=df_info.target_name, axis=1), # .drop(df_info.target_name, axis=1)
        "not_dummy_adv": get_type_instance(result_df, InstanceType.ScaledAdv), #.drop(df_info.target_name, axis=1)
        "df_info": df_info,
        # "groundtruth": get_dummy_version(get_type_instance(result_df, InstanceType.ScaledInput)[df_info.target_name], df_info),
    }


def get_evaluations(result_df: pd.DataFrame, df_info: DfInfo, matrix: List[EvaluationMatrix], models=None, model_name=None):
    '''
    Perform evaluation on the result dataframe according to the matrix given.

    [result_df] -> data frame containing input query and its counterfactaul.
    [df_info] -> DfInfo instance containing all data information.
    [matrix] -> The evaluation matrix to perform on `result_df`.
    '''

    evaluation_df = result_df.copy(deep=True)

    ## Only perform evaluation on the row with found adv.
    # found_idx = evaluation_df[evaluation_df['Found']=="Y"].index
    adv_found_eaval_df = evaluation_df.copy(deep=True)

    if len(adv_found_eaval_df) < 1:
        raise Exception("No adversarial example found, can't provide any evaluation.")

    input_and_adv = prepare_evaluation_dict(adv_found_eaval_df, df_info)

    # pred_attack_success, groundtruth_attack_success, original_accuracy, robust_accuracy =  get_attack_success_accuracy(models, model_name, **input_and_adv)

    metric = {}

    # adv_found_eaval_df[f'eval_pred_attack_success'] = pred_attack_success
    # adv_found_eaval_df[f'groundtruth_attack_success'] = groundtruth_attack_success
    # adv_found_eaval_df[f'original_accuracy'] = original_accuracy
    # adv_found_eaval_df[f'robust_accuracy'] = robust_accuracy

    # metric[f'eval_pred_attack_success']=np.array(pred_attack_success).mean().astype(np.float32)
    # metric[f'groundtruth_attack_success']=np.array(groundtruth_attack_success).mean().astype(np.float32)
    # metric[f'original_accuracy']=np.array(original_accuracy).mean().astype(np.float32)
    # metric[f'robust_accuracy']=np.array(robust_accuracy).mean().astype(np.float32)

    for m in matrix:
        adv_metric = evaluation_name_to_func[m](**input_and_adv)
        adv_found_eaval_df[m.value] = adv_metric
        metric[m.value]=np.array(adv_metric).mean().astype(np.float32)

    evaluation_df.loc[:, adv_found_eaval_df.columns] = adv_found_eaval_df

    return evaluation_df, metric


# def compare_ndarrays(arr1, arr2):
#     if arr1.shape != arr2.shape:
#         raise ValueError("Input arrays have different shapes")
#     return np.where(arr1 == arr2, 0, 1)

# def get_attack_success_accuracy(models, model, **kwargs, ):

#     input_array = np.array(kwargs['input'])
#     adv_array = np.array(kwargs['adv'])
#     groundtruth = np.array(kwargs['groundtruth'])

#     if model == 'dt':
#         predictions = models['dt'].predict(input_array)
#         adv_predictions = models['dt'].predict(adv_array)
#     if model == 'rfc':
#         predictions = models['rfc'].predict(input_array)
#         adv_predictions = models['rfc'].predict(adv_array)
#     if model == 'svc':
#         predictions = models['svc'].predict(input_array)
#         adv_predictions = models['svc'].predict(adv_array)
#     if model == 'lr':
#         predictions = models['lr'].predict(input_array)
#         adv_predictions = models['lr'].predict(adv_array)
#     if model == 'gbc':
#         predictions = models['gbc'].predict(input_array)
#         adv_predictions = models['gbc'].predict(adv_array)
#     if model == 'nn':
#         predictions = (models['nn'].predict(input_array) > 0.5).flatten().astype(int)
#         adv_predictions = (models['nn'].predict(adv_array) > 0.5).flatten().astype(int)
#     if model == 'nn_2':
#         predictions = models['nn_2'].predict(input_array).argmax(axis=1).flatten().astype(int)
#         adv_predictions = models['nn_2'].predict(adv_array).argmax(axis=1).flatten().astype(int)

#     pred_attack_success = compare_ndarrays(predictions, adv_predictions)
#     groundtruth_attack_success = compare_ndarrays(groundtruth, adv_predictions)
#     original_accuracy = accuracy_score(groundtruth, predictions)
#     robust_accuracy = accuracy_score(groundtruth, adv_predictions)


#     return pred_attack_success, groundtruth_attack_success, original_accuracy, robust_accuracy


# def get_performance(result_df: pd.DataFrame, df_info: DfInfo, models, model_name):
#     '''
#     Perform evaluation on the result dataframe according to the matrix given.

#     [result_df] -> data frame containing input query and its counterfactaul.
#     [df_info] -> DfInfo instance containing all data information.
#     [matrix] -> The evaluation matrix to perform on `result_df`.
#     '''

#     evaluation_df = result_df.copy(deep=True)

#     ## Only perform evaluation on the row with found adv.
#     # found_idx = evaluation_df[evaluation_df['Found']=="Y"].index
#     adv_found_eaval_df = evaluation_df.copy(deep=True)

#     if len(adv_found_eaval_df) < 1:
#         raise Exception("No adversarial example found, can't provide any evaluation.")

#     input_and_adv = prepare_evaluation_dict(adv_found_eaval_df, df_info)

#     pred_attack_success, groundtruth_attack_success, original_accuracy, robust_accuracy =  get_attack_success_accuracy(models, model_name, **input_and_adv)


#     # metric = {}

#     adv_found_eaval_df[f'eval_pred_attack_success'] = pred_attack_success
#     adv_found_eaval_df[f'groundtruth_attack_success'] = groundtruth_attack_success
#     adv_found_eaval_df[f'original_accuracy'] = original_accuracy
#     adv_found_eaval_df[f'robust_accuracy'] = robust_accuracy
#     # metric[m.value]=np.array(adv_metric).mean().astype(np.float32)

#     evaluation_df.loc[:, adv_found_eaval_df.columns] = adv_found_eaval_df

#     return evaluation_df