# Categorical and Numeric Relations Dataset

**Ref:** [Dataset Link](https://phys-techsciences.datastations.nl/dataset.xhtml?persistentId=doi:10.17026/dans-zxp-t7tf)

Collaboratively constructed knowledge bases play an important role in information systems, but are essentially always incomplete. Thus, a large number of models has been developed for Knowledge Base Completion, the task of predicting new attributes of entities given partial descriptions of these entities. Virtually all of these models either concentrate on numeric attributes (e.g., "What is Italy's GDP?") or they concentrate on categorical attributes (e.g., "Tim Cook is the chairman of Apple"). This dataset was created as part of a research experiment to develop a model for the joint prediction of numeric and categorical attributes based on embeddings learned from textual occurrences of the entities in question. This dataset consists of numeric and categorical relation tuples spanning seven different domains such as 'animal,' 'country,' 'people,' etc. The tuples presented in this dataset have been used to train and test a neural network framework to perform the above-mentioned task. All data presented in this dataset has been scraped from FreeBase.

**Categorical and Numeric Files:** These files are divided into 'test,' 'train,' and 'validation' sets for different domains like 'animal,' 'book,' etc. Each set contains both 'categorical' and 'numeric' data files:
- **Test Files:** Used to test the model's performance after training.
- **Train Files:** Used to train our model.
- **Validation Files:** Used to validate the model during the training process.
Dataset in txt format, converted into CSV files.

The reason for categorizing the Numeric Value column ("0", "1") in our dataset is that classification algorithms need to work with categorical values rather than continuous values. Classification algorithms make predictions between a certain number of defined classes, so continuous values need to be assigned to these classes.

Since our Numeric Value column contains continuous numeric values, it cannot be used directly with a classification algorithm. I have converted this column into a form that classification algorithms can use by dividing it into specific ranges and marking each range as a separate category.

**Report:**
## KNN with K=3 Classification Report:
|           | Precision |  Recall  | F1-Score | Support |
|-----------|-----------|----------|----------|---------|
|      0    |   1.00    |   0.97   |   0.98   |  45431  |
|      1    |   0.91    |   0.99   |   0.95   |  15926  |
|-----------|-----------|----------|----------|---------|
| Accuracy  |           |          |   0.97   |  61357  |
| Macro Avg |   0.95    |   0.98   |   0.97   |  61357  |
| Weighted Avg | 0.97   |   0.97   |   0.97   |  61357  |

 
## KNN with K=7 Classification Report:
|           | Precision |  Recall  | F1-Score | Support |
|-----------|-----------|----------|----------|---------|
|      0    |   0.99    |   0.97   |   0.98   |  45431  |
|      1    |   0.91    |   0.97   |   0.94   |  15926  |
|-----------|-----------|----------|----------|---------|
| Accuracy  |           |          |   0.97   |  61357  |
| Macro Avg |   0.95    |   0.97   |   0.96   |  61357  |
| Weighted Avg | 0.97   |   0.97   |   0.97   |  61357  |

 
## KNN with K=11 Classification Report:
|           | Precision |  Recall  | F1-Score | Support |
|-----------|-----------|----------|----------|---------|
|      0    |   0.99    |   0.96   |   0.98   |  45431  |
|      1    |   0.90    |   0.96   |   0.93   |  15926  |
|-----------|-----------|----------|----------|---------|
| Accuracy  |           |          |   0.96   |  61357  |
| Macro Avg |   0.94    |   0.96   |   0.95   |  61357  |
| Weighted Avg | 0.96   |   0.96   |   0.96   |  61357  |

 
## MLP with layers (32,) Classification Report:
|           | Precision |  Recall  | F1-Score | Support |
|-----------|-----------|----------|----------|---------|
|      0    |   0.97    |   0.98   |   0.97   |  45431  |
|      1    |   0.94    |   0.91   |   0.93   |  15926  |
|-----------|-----------|----------|----------|---------|
| Accuracy  |           |          |   0.97   |  61357  |
| Macro Avg |   0.95    |   0.97   |   0.96   |  61357  |
| Weighted Avg | 0.97   |   0.97   |   0.97   |  61357  |

 
## MLP with layers (32, 32, 32) Classification Report:
|           | Precision |  Recall  | F1-Score | Support |
|-----------|-----------|----------|----------|---------|
|      0    |   0.99    |   0.96   |   0.98   |  45431  |
|      1    |   0.90    |   0.97   |   0.94   |  15926  |
|-----------|-----------|----------|----------|---------|
| Accuracy  |           |          |   0.97   |  61357  |
| Macro Avg |   0.95    |   0.97   |   0.96   |  61357  |
| Weighted Avg | 0.97   |   0.97   |   0.97   |  61357  |


## Naive Bayes Classification Report:
|           | Precision |  Recall  | F1-Score | Support |
|-----------|-----------|----------|----------|---------|
|      0    |   0.77    |   0.92   |   0.84   |  45431  |
|      1    |   0.50    |   0.23   |   0.31   |  15926  |
|-----------|-----------|----------|----------|---------|
| Accuracy  |           |          |   0.74   |  61357  |
| Macro Avg |   0.63    |   0.57   |   0.58   |  61357  |
| Weighted Avg | 0.70   |   0.74   |   0.70   |  61357  |
