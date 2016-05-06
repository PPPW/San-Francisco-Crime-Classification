# San-Francisco-Crime-Classification

This Kaggle competition asks people to predict the crime type by using features such as date, time, address and coordinates. 
A detailed description can be accessed from [here](https://www.kaggle.com/c/sf-crime).

The code in this repo used various types of machine learning models, and combined the best models via stacking, to improve the performance. 

# Dependencies:
* Pandas
* NumPy
* SciKit-Learn
* Lasagne
* nolearn

# Usage:

Download the raw data from [here](https://www.kaggle.com/c/sf-crime/data), placed it in the same folder as `SF.py`. 

Run the code for different models: 
```
SF.py modelName isFinal toSave
```

`modelName`: the name of the model to use, it can be:   

* lr: logistic regression using one-vs-all   
* softmax: multinomial regression   
* knn   
* sgd: stochastic gradient descent   
* rf: random forest   
* lasagne: neural networks   
* stackLog: stack the models using logistic regression on the models' results. Need to run those models first. Can change what models to combine by changing the variable `models`.

`isFinal`: If true, then run the model and save the result. If false, then run the model for testing only, don't save the result. 

`toSave`: If true and `isFinal` is true as well, save the prediction result in the format for submission. 




