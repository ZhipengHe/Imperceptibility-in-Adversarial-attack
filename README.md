# Evaluating Imperceptibility of White-Box Adversarial Attacks on Tabular Data

> Submission for IJCAI-23 main track

## Abstract

Adversarial attacks are a potential threat to machine learning models, as they can cause the model to make incorrect predictions by introducing imperceptible perturbations to the input data. Adversarial attacks are extensively explored in the literature for image data but not for tabular data, which is low-dimensional and heterogeneous. In this work, we propose a set of standard metrics for evaluating imperceptibility of adversarial attacks, which encompass three perspectives: adversary perspective, observer perspective and detector perspective. We evaluate the performance and imperceptibility of three white-box adversarial attack methods and their variants using different machine learning models and with a focus on tabular data. As an insightful finding from our evaluation, it is challenging to craft adversarial examples that are both the most effective and least perceptible due to a trade-off between imperceptibility and performance.


## Data Profiling

| Dataset       	| Data Type 	| Total Inst. 	| Train/Test<br>(80%:20%) 	| Batch/Adv Inst.<br>(batch_size=64) 	| Total Feat. 	| Categorical Feat. 	| Numerical Feat. 	| Total Categorical Feat.<br>after One Hot Enc. 	|
|---------------	|:---------:	|:-----------:	|:-----------------------:	|:----------------------------------:	|:-----------:	|:-----------------:	|:---------------:	|:---------------------------------------------:	|
| Adult/Income  	|   Mixed   	|    32651    	|        26048/6513       	|              101/6464              	|     12      	|         8         	|        4        	|                       98                      	|
| Breast Cancer 	|    Num    	|     569     	|         455/114         	|                1/64                	|      30     	|         0         	|        30       	|                       0                       	|
| COMPAS        	|   Mixed   	|     7214    	|        5771/1443        	|               22/1408              	|      11     	|         7         	|        4        	|                       19                      	|
| Diabetes      	|    Num    	|     768     	|         614/154         	|                2/128               	|      8      	|         0         	|        8        	|                       0                       	|
| German Credit 	|   Mixed   	|     1000    	|         800/200         	|                3/192               	|      20     	|         15        	|        5        	|                       58                      	|
