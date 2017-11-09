import numpy as np
from sklearn import tree, datasets
from sklearn.externals.six import StringIO
import pydotplus
from IPython.display import Image
import graphviz
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


#2. Fit the decision tree model:
model = tree.DecisionTreeClassifier()
model.fit(x_train, y_train)
print(model.fit(x_train, y_train)) #prints the output on running this line

#3. Visualize the model:
dot = tree.export_graphviz(model, out_file=None)
graph = graphviz.Source(dot)
graph.render("04-Decision Tree(output tree)")

dot_data = StringIO()
tree.export_graphviz(model, out_file=dot_data,
                     feature_names=iris.feature_names,
                     class_names=iris.target_names,
                     filled = True, rounded = True,
                     special_characters = True)
graph = pydotplus.graph_from_dot_data((dot_data.getvalue()))
Image(graph.create_png())

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
DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=None, splitter='best')
Validation misclassification precentage =  6.666666666666667 %
Test misclassification precentage =  0.0 %
"""
