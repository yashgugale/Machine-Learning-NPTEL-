import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

#1. Prepare a synthetic dataset using the equation:
#  y = x/2 + sin(x) + E, where E~N(0,1) is the Gausian noise to add randomness to the data
number_of_samples = 200
x = np.linspace(-np.pi, np.pi, number_of_samples)
y = 0.5 * np.sin(x) + np.random.random(x.shape)
plt.scatter(x, y, edgecolors='black')
plt.xlabel('x-input feature')
plt.ylabel('y-input feature')
plt.title('Fig 1 : Data for linear regression')
plt.show()

#2. Split the data into training, validation and test sets:
#Here we do a 70-15-15 random split between the training, validation and test sets respectively
random_indices = np.random.permutation(number_of_samples)
#Training set:
x_train = x[random_indices[:140]]
y_train = y[random_indices[:140]]
#Validation set:
x_val = x[random_indices[140:175]]
y_val = y[random_indices[140:175]]
#Test set:
x_test = x[random_indices[175:]]
y_test = y[random_indices[175:]]

#3. Fit a line to the data:
model = linear_model.LinearRegression()     #Create a least squared error linear regression object
#sklearn takes the input as matrices. Hence, we reshape the array into column matrices
x_train_for_line_fitting = np.matrix(x_train.reshape(len(x_train), 1))
y_train_for_line_fitting = np.matrix(y_train.reshape(len(y_train), 1))

#fit the line to the training data
model.fit(x_train_for_line_fitting, y_train_for_line_fitting)

#plot the line
plt.scatter(x_train, y_train, edgecolors='black')
plt.plot(x.reshape((len(x),1)), model.predict(x.reshape((len(x), 1))), color = 'blue')
plt.xlabel('x-input features')
plt.ylabel('y-target values')
plt.title('Fig 2 : Line fitted to the training data')
plt.show()

#4. Evaluate the model:
mean_val_error = np.mean((y_val - model.predict(x_val.reshape(len(x_val), 1)))**2)
mean_test_error = np.mean((y_test - model.predict(x_test.reshape(len(x_test), 1)))**2)
print('Validation MSE : ', mean_val_error, '\nTest MSE : ', mean_test_error)

"""
Sample output:
Validation MSE :  0.350500756656
Test MSE :  0.308628606527
"""
