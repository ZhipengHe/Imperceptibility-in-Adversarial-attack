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

# Set the desired parameters for the attack
deepfool_params = {
    # 'targeted': False,
    # 'confidence': 0.0, 
    'max_iter': 1000, 
    # 'learning_rate': 1e-2, 
    # 'binary_search_steps': 1, 
    # 'initial_const': 1e-2,
    'batch_size': 64,
    'verbose': True,
    }



def art_wrap_models(models, feature_range):
    '''
    Wrap the model to meet the requirements to art.attacks.evasion.DeepFool
    '''

    return {
        'lr': ScikitlearnLogisticRegression(models['lr'], clip_values=feature_range),
        'svc': ScikitlearnSVC(models['svc'], clip_values=feature_range),
        'nn_2': KerasClassifier(models['nn_2'], clip_values=feature_range),
    }


def get_deepfool_instance(wrapped_models):
    '''
    '''
    adv_instance = {}

    for k in wrapped_models.keys():
        adv_instance[k] = DeepFool(classifier=wrapped_models[k], **deepfool_params)
    
    return adv_instance


def generate_deepfool_result(
        df_info: DfInfo,
        models,
        num_instances,
        X_test, y_test,
        models_to_run=['svc', 'lr', 'nn_2'],
):

    # Since we use min-max scaler and one-hot encoding, we can contraint the range in [0, 1]
    # feature_range = (np.zeros((1, len(df_info.feature_names))), 
    #                  np.ones((1, len(df_info.feature_names))))
    
    feature_range=(0,1)

    print("Feature range:" )
    print(feature_range)

    wrapped_models = art_wrap_models(models, feature_range)

    # Get adversarial examples generator instance.
    adv_instance = get_deepfool_instance(wrapped_models)

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
        adv = adv_instance[k].generate(X_test_re,y_test_re)
        end_t = time()

        # Calculate the running time.
        running_time = end_t - start_t

        # Get the prediction from original predictive model in a human-understandable format.
        if k == 'nn_2':
            # nn return float [0, 1], so we need to define a threshold for it. (It's usually 0.5 for most of the classifier).
            prediction = np.argmax(models[k].predict(X_test_re), axis=1).astype(int)
            adv_prediction = np.argmax(models[k].predict(adv), axis=1).astype(int)
            
        else:
            # dt and rfc return int {1, 0}, so we don't need to define a threshold to get the final prediction.
            prediction = models[k].predict(X_test_re)
            adv_prediction = models[k].predict(adv)

        # Looping throguh first `num_instances` in the test set.
        for idx, instance in enumerate(X_test_re):
            example = instance.reshape(1, -1)
            adv_example = adv[idx].reshape(1,-1)

            adv_example_df = inverse_dummy(pd.DataFrame(adv_example, columns=df_info.ohe_feature_names), df_info.cat_to_ohe_cat)

            # Change the found input from ohe format to original format.
            input_df = inverse_dummy(pd.DataFrame(example, columns=df_info.ohe_feature_names), df_info.cat_to_ohe_cat)
            input_df.loc[0, df_info.target_name] = df_info.target_label_encoder.inverse_transform([prediction[idx]])[0]

            results[k].append({
                "input": instance,
                "input_df": input_df,
                "adv_example": adv[idx],
                "adv_example_df": adv_example_df,
                "running_time": running_time,
                "ground_truth": df_info.target_label_encoder.inverse_transform([y_test[idx]])[0],
                "prediction": df_info.target_label_encoder.inverse_transform([prediction[idx]])[0],
                "adv_prediction": df_info.target_label_encoder.inverse_transform([adv_prediction[idx]])[0],
            })
    
    return results
