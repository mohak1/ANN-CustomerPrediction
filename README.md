# ANN-CustomerPrediction
Predicting whether a customer stays or leaves the bank based on its previous data.
Makes use of numpy and pandas libraries
Using 'Lable Encoder' and 'One Hot Encoder' for data preprocessing
The data is split into 'Training' and 'Testing' using sklearn's train_test_split
Feature scaling is done using 'Standard Scalar'
'Dropout' layer is used to reduce overfitting on the training set
Parameter Tuning is done by making the use of 'GridSearchCV'. This is done to obtain efficient values for the Hyper Parameters.
Grid Search creates a keras wrapper for the model which let's us test different optimizers, batch_size and epochs for the model and obtain the best possible combination of optimizer, batch_size and epoch which leads to highest accuracy.
