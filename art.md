# Essential information for attack packages

## Adversarial Robustness Toolbox (ART) v1.12.2

- [Documentation](https://adversarial-robustness-toolbox.readthedocs.io/)
- [Github](https://github.com/Trusted-AI/adversarial-robustness-toolbox/tree/1.12.2)


## Which predictive models can be used in ART?

|                     	| **Type**          	|  **Decision Tree** 	|  **Random Forest** 	|   **Linear SVC**   	| **Logistic Regression** 	| **Neural Networks** 	|
|---------------------	|-------------------	|:------------------:	|:------------------:	|:------------------:	|:-----------------------:	|:-------------------:	|
| **DeepFool**        	| Gradients         	|         :x:        	|         :x:        	| :heavy_check_mark: 	|    :heavy_check_mark:   	|  :heavy_check_mark: 	|
| **LowProFool**      	| Gradients         	|         :x:        	|         :x:        	| :heavy_check_mark: 	|    :heavy_check_mark:   	|  :heavy_check_mark: 	|
| **C&W Attack**      	| Gradients         	|         :x:        	|         :x:        	| :heavy_check_mark: 	|    :heavy_check_mark:   	|  :heavy_check_mark: 	|
| **Boundary Attack** 	| Black Box Attack   	| :heavy_check_mark: 	| :heavy_check_mark: 	| :heavy_check_mark: 	|    :heavy_check_mark:   	|  :heavy_check_mark: 	|
| **HopSkipJump Attack**| Black Box Attack   	| :heavy_check_mark:    | :heavy_check_mark: 	| :heavy_check_mark: 	|    :heavy_check_mark:   	|  :heavy_check_mark: 	|
| **Sign-OPT Attack**   | Black Box Attack   	| :heavy_check_mark:    | :heavy_check_mark: 	| :heavy_check_mark: 	|    :heavy_check_mark:   	|  :heavy_check_mark: 	|

<details><summary>Classifier types in ART</summary>
<p>

```python
CLASSIFIER_CLASS_LOSS_GRADIENTS_TYPE = Union[  # pylint: disable=C0103
    ClassifierClassLossGradients,
    EnsembleClassifier,
    GPyGaussianProcessClassifier,
    KerasClassifier,
    MXClassifier,
    PyTorchClassifier,
    ScikitlearnLogisticRegression,
    ScikitlearnSVC,
    TensorFlowClassifier,
    TensorFlowV2Classifier,
]

CLASSIFIER_NEURALNETWORK_TYPE = Union[  # pylint: disable=C0103
    ClassifierNeuralNetwork,
    DetectorClassifier,
    EnsembleClassifier,
    KerasClassifier,
    MXClassifier,
    PyTorchClassifier,
    TensorFlowClassifier,
    TensorFlowV2Classifier,
]

CLASSIFIER_DECISION_TREE_TYPE = Union[  # pylint: disable=C0103
    ClassifierDecisionTree,
    LightGBMClassifier,
    ScikitlearnDecisionTreeClassifier,
    ScikitlearnExtraTreesClassifier,
    ScikitlearnGradientBoostingClassifier,
    ScikitlearnRandomForestClassifier,
    XGBoostClassifier,
]

CLASSIFIER_TYPE = Union[  # pylint: disable=C0103
    Classifier,
    BlackBoxClassifier,
    CatBoostARTClassifier,
    DetectorClassifier,
    EnsembleClassifier,
    GPyGaussianProcessClassifier,
    KerasClassifier,
    JaxClassifier,
    LightGBMClassifier,
    MXClassifier,
    PyTorchClassifier,
    ScikitlearnClassifier,
    ScikitlearnDecisionTreeClassifier,
    ScikitlearnExtraTreeClassifier,
    ScikitlearnAdaBoostClassifier,
    ScikitlearnBaggingClassifier,
    ScikitlearnExtraTreesClassifier,
    ScikitlearnGradientBoostingClassifier,
    ScikitlearnRandomForestClassifier,
    ScikitlearnLogisticRegression,
    ScikitlearnSVC,
    TensorFlowClassifier,
    TensorFlowV2Classifier,
    XGBoostClassifier,
    CLASSIFIER_NEURALNETWORK_TYPE,
]
```

</p>
</details>

## Current problems in ART

### DeepFool

When the neural network only have one output node, it has the following error:

```
ValueError: This attack has not yet been tested for binary classification with a single output classifier.
```

The error is from ([source code](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/art/attacks/evasion/deepfool.py#L104-L107)):

```python
if self.estimator.nb_classes == 2 and preds.shape[1] == 1:
    raise ValueError(  # pragma: no cover
        "This attack has not yet been tested for binary classification with a single output classifier."
    )
```

https://stats.stackexchange.com/questions/207049/neural-network-for-binary-classification-use-1-or-2-output-neurons