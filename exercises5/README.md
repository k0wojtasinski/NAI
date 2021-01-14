# Classification API

### Goal
Goal of this project is to prepare simple API to perform classification experiments.  
It supports float/int-based datasets (except for y-column) and y-column needs to be the last one.  
It provides "MachineLearningExperimentResults" class to save the output of model validation (which can be saved into json files).  
It supports SVM and AdaBoost classifiers.  
I tested it with three datasets (data/iris.csv, data/diabetes.csv, data/absenteeism.csv).  


### Datasets

Both iris and diabetes datasets are from (https://machinelearningmastery.com/standard-machine-learning-datasets/)
Absenteeism dataset is from https://archive.ics.uci.edu/ml/datasets/Absenteeism+at+work
It tries to determine how different factors can impact absenteeism among workers  

### How to run it
First, you have to install all the dependencies (``pip install -r requirements ``)
Then you can use CLI mode, which supports two modes:

a) validate (`python classifier_api.py validate`)  

it takes input to gather all the arguments (including csv file path) and runs validation (validates model with test data)

b) predict (`python classifier_api.py predict`)  

it takes input to gather all the arguments (including csv file path) and runs prediction (based on comma separated user's input)

### Examples

a) output for validation of diabetes dataset  
![diabetes](https://i.imgur.com/qvaARNa.png)

b) output for predict of iris dataset  
![iris](https://i.imgur.com/Zyp3R1e.png)

c) output for validation of absenteeism dataset  
![absenteeism](https://i.imgur.com/BS3qFGl.png)

