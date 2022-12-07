import numpy as np
import pandas as pd

from time import time
from utils.preprocessing import DfInfo
from utils.preprocessing import inverse_dummy

from art.attacks.evasion import DeepFool
from art.estimators.classification import SklearnClassifier, KerasClassifier
# from art.estimators.classification.scikitlearn import ScikitlearnDecisionTreeClassifier
# from art.estimators.classification.scikitlearn import ScikitlearnRandomForestClassifier
from art.estimators.classification.scikitlearn import ScikitlearnLogisticRegression
from art.estimators.classification.scikitlearn import ScikitlearnSVC

'''
Acronym:

dt -> Decision Tree
rfc -> Random Forest Classifier
nn -> Nueral Network
ohe -> One-hot encoding format
'''

'''

CLASSIFIER_CLASS_LOSS_GRADIENTS_TYPE

art.estimators.classification.SklearnClassifier()
art.estimators.classification.KerasClassifier(
    model, # Keras model
    use_logits, 
    clip_values, # Tuple of the form (min, max) representing the minimum and maximum values allowed for features
    )

'''


def art_wrap_models(models, feature_range):
    '''
    Wrap the model to meet the requirements to art.attacks.evasion.DeepFool
    '''

    return {
        'lr': ScikitlearnLogisticRegression(models['lr'], clip_values=feature_range),
        'svc': ScikitlearnSVC(models['svc'], clip_values=feature_range),
        'nn_2': KerasClassifier(models['nn_2'], clip_values=feature_range),
    }


def get_deepfool_ae(wrapped_models, max_iters):
    '''
    '''
    deepfool_ae = {}

    for k in wrapped_models.keys():
        deepfool_ae[k] = DeepFool(
            classifier=wrapped_models[k],
            max_iter=max_iters,
            verbose=True,
            batch_size=64
        )
    
    return deepfool_ae


def generate_deepfool_result(
        df_info: DfInfo,
        train_df,
        models,
        num_instances,
        num_ae_per_instance,
        X_train, y_train, X_test, y_test,
        max_iters=1000,
        models_to_run=['svc', 'lr', 'nn_2'],
        output_int=True
):

    # Since we use min-max scaler and one-hot encoding, we can contraint the range in [0, 1]
    # feature_range = (np.zeros((1, len(df_info.feature_names))), 
    #                  np.ones((1, len(df_info.feature_names))))
    
    feature_range=(0,1)

    print("Feature range:" )
    print(feature_range)

    wrapped_models = art_wrap_models(models, feature_range)

    # Get adversarial examples generator instance.
    deepfool_ae = get_deepfool_ae(wrapped_models, max_iters)

    # Initialise the result dictionary.(It will be the return value.)
    results = {}

    X_test_re=X_test[0:num_instances]
    y_test_re=y_test[0:num_instances]

    # Loop through every models (svc, lr, nn_2)
    for k in models_to_run:
        # Intialise the result for the classifier (predicting model).
        results[k] = []

        print(f"Finding adversarial examples for {k}")

        start_t = time()
        ae = deepfool_ae[k].generate(X_test_re,y_test_re)
        end_t = time()

        # Calculate the running time.
        running_time = end_t - start_t

        # Get the prediction from original predictive model in a human-understandable format.
        if k == 'nn_2':
            # nn return float [0, 1], so we need to define a threshold for it. (It's usually 0.5 for most of the classifier).
            prediction = np.argmax(models[k].predict(X_test_re), axis=1).astype(int)
            
        else:
            # dt and rfc return int {1, 0}, so we don't need to define a threshold to get the final prediction.
            prediction = models[k].predict(X_test_re)

        # Looping throguh first `num_instances` in the test set.
        for idx, instance in enumerate(X_test_re):
            example = instance.reshape(1, -1)

            ae_instance = inverse_dummy(pd.DataFrame(ae[idx].reshape(1,-1), columns=df_info.ohe_feature_names), df_info.cat_to_ohe_cat)

            # Change the found input from ohe format to original format.
            input_df = inverse_dummy(pd.DataFrame(example, columns=df_info.ohe_feature_names), df_info.cat_to_ohe_cat)
            input_df.loc[0, df_info.target_name] = df_info.target_label_encoder.inverse_transform([prediction[idx]])[0]

            results[k].append({
                "input": input_df,
                "ae": ae_instance,
                "running_time": running_time,
                "ground_truth": df_info.target_label_encoder.inverse_transform([y_test[idx]])[0],
                "prediction": df_info.target_label_encoder.inverse_transform([prediction[idx]])[0],
            })
    
    return results
                


def process_result(results, df_info):
    '''
    Process the result dictionary to construct data frames for each (dt, rfc, nn).
    '''

    results_df = {}

    # Loop through ['dt', 'rfc', 'nn']
    for k in results.keys():

        all_data = []
        for i in range(len(results[k])):

            final_df = pd.DataFrame([{}])

            # Inverse the scaling process to get the original data for input.
            scaled_input_df = results[k][i]['input'].copy(deep=True)
            origin_columns = [
                f"origin_input_{col}" for col in scaled_input_df.columns]
            origin_input_df = scaled_input_df.copy(deep=True)
            scaled_input_df.columns = [
                f"scaled_input_{col}" for col in scaled_input_df.columns]

            origin_input_df[df_info.numerical_cols] = df_info.scaler.inverse_transform(
                origin_input_df[df_info.numerical_cols])
            origin_input_df.columns = origin_columns

            final_df = final_df.join([scaled_input_df, origin_input_df])

            # If counterfactaul found, inverse the scaling process to get the original data for cf.
            if not results[k][i]['ae'] is None:
                scaled_ae_df = results[k][i]['ae'].copy(deep=True)
                # Comment this
                # scaled_cf_df.loc[0, target_name] = target_label_encoder.inverse_transform([scaled_cf_df.loc[0, target_name]])[0]
                origin_ae_columns = [
                    f"origin_ae_{col}" for col in scaled_ae_df.columns]
                origin_ae_df = scaled_ae_df.copy(deep=True)
                scaled_ae_df.columns = [
                    f"scaled_ae_{col}" for col in scaled_ae_df.columns]

                origin_ae_df[df_info.numerical_cols] = df_info.scaler.inverse_transform(
                    origin_ae_df[df_info.numerical_cols])
                origin_ae_df.columns = origin_ae_columns

                final_df = final_df.join([scaled_ae_df, origin_ae_df])

            # Record additional information.
            final_df['running_time'] = results[k][i]['running_time']
            final_df['Success?'] = "Y" if not results[k][i]['ground_truth'] == results[k][i]['prediction'] else "N"
            final_df['ground_truth'] = results[k][i]['ground_truth']
            final_df['prediction'] = results[k][i]['prediction']

            all_data.append(final_df)

        results_df[k] = pd.concat(all_data)

    return results_df