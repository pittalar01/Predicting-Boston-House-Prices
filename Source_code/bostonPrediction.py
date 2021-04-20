import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score

boston = load_boston()

data = boston.data
target = boston.target

X_learning, X_evaluation, Y_learning, Y_evaluation = train_test_split(data, target, test_size = 0.10)

X_train, X_test, Y_train, Y_test = train_test_split(X_learning, Y_learning, test_size = 0.20)

en = ElasticNet(alpha=0.01, l1_ratio=0.5)
en.fit(X_train,Y_train)
Y_predict_train = en.predict(X_train) #predictions on training data
Y_predict_test = en.predict(X_test) #predictions on testing data
Y_predict_evaluation = en.predict(X_evaluation) #predictions on evaluation data

print('On Training Data in Learning:')
print('RMSE: ',np.sqrt(mean_squared_error(Y_train,Y_predict_train)))
print('R2: ',r2_score(Y_train,Y_predict_train))
print('\n')
print('On Testing Data in Learning:')
print('RMSE: ',np.sqrt(mean_squared_error(Y_test,Y_predict_test)))
print('R2: ',r2_score(Y_test,Y_predict_test))
print('\n')
print('On Evaluation Data:')
print('RMSE: ',np.sqrt(mean_squared_error(Y_evaluation,Y_predict_evaluation)))
print('R2: ',r2_score(Y_evaluation,Y_predict_evaluation))
