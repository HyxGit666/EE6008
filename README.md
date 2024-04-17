# EE6008
Group work

AA-STL-MIA.csv is the dataset we use, contain an airline delay data with different reasons.


Below is our experimental setting, you can configure your environment like this:
![image](https://github.com/HyxGit666/EE6008/assets/167158033/748e1f5c-32ba-4777-8fc4-d2090b01bf3f)



To run the file preprocessing.py, please change the file path to read the dataset.
![image](https://github.com/HyxGit666/EE6008/assets/167158033/9a931196-509c-4234-bb49-ec0931d191f4)

model_repo.py have all deeplearning models we used, to use the fuction, just using create_xxx(inputshape1,inputshape2)

processing.py process the raw dataset and convert it into data that can be used for training and evaluation.

training_HEYU.py can train and evalute different models one by one and see the results.

training_figure.py train all models at the same time and give the figure the loss, mae and mse decrease during the training process.

 RF and KNN.ipynb train and evaluate the RF and KNN models one by one and see the results.
