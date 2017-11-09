import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt

#We solve the same problem (from linear regression) using decision tree
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
maximum_depth_of_tree = np.arange(10)+1
train_err_arr = []
val_err_arr = []
test_err_arr = []

for depth in maximum_depth_of_tree:
    model = tree.DecisionTreeRegressor(max_depth=depth)
    #sklearn takes the input as matrices. Hence, we reshape the array into column matrices
    x_train_for_line_fitting = np.matrix(x_train.reshape(len(x_train), 1))
    y_train_for_line_fitting = np.matrix(y_train.reshape(len(y_train), 1))

    #fit the line to the training data
    model.fit(x_train_for_line_fitting, y_train_for_line_fitting)

    #plot the line
    plt.figure()
    plt.scatter(x_train, y_train, edgecolors='black')
    plt.plot(x.reshape((len(x),1)), model.predict(x.reshape((len(x), 1))), color = 'blue')
    plt.xlabel('x-input features')
    plt.ylabel('y-target values')
    plt.title('Fig 1 : Line fitted to the training data with max_depth='+str(depth))
    plt.show()

    #4. Evaluate the model:
    mean_train_error = np.mean((y_train - model.predict(x_train.reshape(len(x_train), 1)))**2)
    mean_val_error = np.mean((y_val - model.predict(x_val.reshape(len(x_val), 1)))**2)
    mean_test_error = np.mean((y_test - model.predict(x_test.reshape(len(x_test), 1)))**2)

    train_err_arr.append(mean_train_error)
    val_err_arr.append(mean_val_error)
    test_err_arr.append(mean_test_error)

    print('Training MSE : ', mean_train_error, '\nValidation MSE : ', mean_val_error, '\nTest MSE : ', mean_test_error)

plt.figure()
plt.plot(train_err_arr, c='red')
plt.plot(val_err_arr, c='blue')
plt.plot(test_err_arr, c='green')
plt.legend(['Training error', 'Validation error', 'Test error'])
plt.title('Variation of error with maximum depth of tree')
plt.show()

"""
Sample output:
Training MSE :  0.101661766036 
Validation MSE :  0.113132772356 
Test MSE :  0.0843867822835
Training MSE :  0.0909332059427 
Validation MSE :  0.0891472010995 
Test MSE :  0.0825891039603
Training MSE :  0.0789544757112 
Validation MSE :  0.104266378311 
Test MSE :  0.0738895354471
Training MSE :  0.0706595693462 
Validation MSE :  0.115178966824 
Test MSE :  0.0613256637749
Training MSE :  0.0583884289574 
Validation MSE :  0.149286474984 
Test MSE :  0.0762389943854
Training MSE :  0.052168543683 
Validation MSE :  0.170172447253 
Test MSE :  0.0858981954201
Training MSE :  0.0412500382918 
Validation MSE :  0.176938569038 
Test MSE :  0.0945658522321
Training MSE :  0.0320747824683 
Validation MSE :  0.182167592662 
Test MSE :  0.103506863832
Training MSE :  0.0252679526569 
Validation MSE :  0.172878797909 
Test MSE :  0.121785621488
Training MSE :  0.0186383627477 
Validation MSE :  0.184024098585 
Test MSE :  0.104052286692
"""
