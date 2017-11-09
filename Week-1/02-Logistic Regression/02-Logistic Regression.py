import numpy as np
from sklearn import linear_model, datasets
import matplotlib.pyplot as plt

#1. Prepare data (use iris dataset)
iris = datasets.load_iris()
X = iris.data[:, :2]  #choosing only the first 2 input features
Y = iris.target

#The first 50 samples are class 0 and the next 50 samples are class 1
X = X[:100]
Y = Y[:100]
number_of_samples = len(Y)

#Splitting into training, validation and test sets:
random_indices = np.random.permutation(number_of_samples)
#Training set:
num_training_samples = int(number_of_samples*0.7)
x_train = X[random_indices[:num_training_samples]]
y_train = Y[random_indices[:num_training_samples]]
#Validation set:
num_validation_samples = int(number_of_samples*0.15)
x_val = X[random_indices[num_training_samples:num_training_samples+num_validation_samples]]
y_val = Y[random_indices[num_training_samples:num_training_samples+num_validation_samples]]
#Test set:
num_test_samples = int(number_of_samples*0.15)
x_test = X[random_indices[-num_test_samples:]]
y_test = Y[random_indices[-num_test_samples:]]

#Visualizing the training data:
X_class_0 = np.asmatrix([x_train[i] for i in range(len(x_train)) if y_train[i] == 0])
Y_class_0 = np.zeros((X_class_0.shape[0]), dtype=np.int)
X_class_1 = np.asmatrix([x_train[i] for i in range(len(x_train)) if y_train[i] == 1])
Y_class_1 = np.ones((X_class_1.shape[0]), dtype=np.int)

plt.scatter([X_class_0[:, 0]],[X_class_0[:, 1]], edgecolors='red')
plt.scatter([X_class_1[:, 0]],[X_class_1[:, 1]], edgecolors='blue')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend(['class 0', 'class 1'])
plt.title('Fig 1 : Visualization of the training data')
plt.show()

#2. Fit the logistic regression model:
model = linear_model.LogisticRegression(C=1e5)  #C is the inverse of the regularization factor
full_X = np.concatenate((X_class_0, X_class_1), axis=0)
full_Y = np.concatenate((Y_class_0, Y_class_1), axis=0)
model.fit(full_X, full_Y)

#Display the decision boundary
#Visualization code taken from : http://scikit-learn.org
#For plotting the decision boundary, we will assign a color to each point in the mesh:

h = 0.2     #step size in the mesh
x_min, x_max = full_X[:, 0].min() - .5, full_X[:, 0].max() + .5
y_min, y_max = full_X[:, 1].min() - .5, full_X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
#predict for the entire mesh to find the regions for each class
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

#Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap = plt.cm.Paired)

#Plot also the training points:
plt.scatter([X_class_0[:, 0]],[X_class_0[:, 1]], c='red', edgecolors='k', cmap=plt.cm.Paired)
plt.scatter([X_class_1[:, 0]],[X_class_1[:, 1]], c='blue', edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.title('Fig 1 : Visualization of the decision boundary')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.show()

#4. Evaluate the model:
validation_set_predictions = [model.predict(x_val[i].reshape((1,2)))[0] for i in range(x_val.shape[0])]
validation_misclassification_percentage = 0
for i in range(len(validation_set_predictions)):
    if validation_set_predictions[i] != y_val[i]:
        validation_misclassification_percentage += 1
validation_misclassification_percentage *= 100/len(y_val)
print("Validation misclassification precentage = ", validation_misclassification_percentage, "%")

test_set_predictions = [model.predict(x_test[i].reshape((1,2)))[0] for i in range(x_test.shape[0])]
test_misclassification_percentage = 0
for i in range(len(test_set_predictions)):
    if test_set_predictions[i] != y_test[i]:
        test_misclassification_percentage += 1
test_misclassification_percentage *= 100/len(y_test)
print("Test misclassification precentage = ", test_misclassification_percentage, "%")

"""
Sample output:
Validation misclassification precentage =  0.0 %
Test misclassification precentage =  0.0 %

Note: Zero misclassification error was possible only because the two classes were lineary seperable in the chosen
feature space. However, this is seldom the case in most real world classification problems.
"""
