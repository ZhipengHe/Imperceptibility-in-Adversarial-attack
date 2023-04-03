# Imperceptibility in Adversarial attack

## Data Profiling

| Dataset       	| Data Type 	| Total Inst. 	| Train/Test<br>(80%:20%) 	| Batch/Adv Inst.<br>(batch_size=64) 	| Total Feat. 	| Categorical Feat. 	| Numerical Feat. 	| Total Categorical Feat.<br>after One Hot Enc. 	|
|---------------	|:---------:	|:-----------:	|:-----------------------:	|:----------------------------------:	|:-----------:	|:-----------------:	|:---------------:	|:---------------------------------------------:	|
| Adult/Income  	|   Mixed   	|    32651    	|        26048/6513       	|              101/6464              	|     12      	|         8         	|        4        	|                       98                      	|
| Breast Cancer 	|    Num    	|     569     	|         455/114         	|                1/64                	|      30     	|         0         	|        30       	|                       0                       	|
| COMPAS        	|   Mixed   	|     7214    	|        5771/1443        	|               22/1408              	|      11     	|         7         	|        4        	|                       19                      	|
| Diabetes      	|    Num    	|     768     	|         614/154         	|                2/128               	|      8      	|         0         	|        8        	|                       0                       	|
| German Credit 	|   Mixed   	|     1000    	|         800/200         	|                3/192               	|      20     	|         15        	|        5        	|                       58                      	|
