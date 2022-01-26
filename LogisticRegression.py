Logistic Regression:
Load packages
Define functions Sigmoid
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
 def sigmoid(z):
   """
   Compute the sigmoid of z
   Arguments:
   x -- A scalar or numpy array of any size.
   Return:
   s -- sigmoid(z)
   """
   s=1/(1+np.exp(-z))
   return s
sigmoid(np.array([2,3,4]))
 Initialize weights
def initialize_weights(dim):
   """
   This function creates a vector of zeros of shape (dim, 1) for w and
initializes b to 0.
Argument:
   dim -- size of the w vector we want (or number of parameters in this
case)
   Returns:
   w -- initialized vector of shape (dim, 1)
   b -- initialized scalar (corresponds to the bias)
   """
   w=np.zeros([dim,1])
   b=0
    assert(w.shape == (dim, 1))
   assert(isinstance(b, float) or isinstance(b, int))
return w, b
 Forward and backward propagation
def propagate(w, b, X, Y):
   """
   Implement the cost function and its gradient for the propagation
explained in the assignment
   Arguments:
   w -- weights, a numpy array of size (num_px * num_px, 1)
   b -- bias, a scalar
   X -- data of size (number of examples, num_px * num_px)
   Y -- true "label" vector of size (1, number of examples)
   Return:
   cost -- negative log-likelihood cost for logistic regression
   dw -- gradient of the loss with respect to w, thus same shape as w
   db -- gradient of the loss with respect to b, thus same shape as b
 
  """
m = X.shape[0]
# FORWARD PROPAGATION (FROM X TO COST)
y_pred=sigmoid(np.dot(w.T, X.T) + b)
cost=-1/m * np.sum( np.dot(np.log(y_pred), Y.T) + np.dot(np.log(1-
y_pred), (1-Y.T)))
# BACKWARD PROPAGATION (TO FIND GRAD)
dw=np.dot(X.T,(y_pred-Y).T)/m
db=np.sum((y_pred-Y))/m
assert(dw.shape == w.shape)
assert(db.dtype == float)
cost = np.squeeze(cost)
assert(cost.shape == ())
grads = {"dw": dw,
         "db": db}
return grads, cost
Gradient descent
def gradient_descent(w, b, X, Y, num_iterations, learning_rate):
   """
   This function optimizes w and b by running a gradient descent algorithm
   Arguments:
   w -- weights, a numpy array of size (num_px * num_px, 1)
   b -- bias, a scalar
   X -- data of shape (num_px * num_px, number of examples)
   Y -- true "label" vector of shape (1, number of examples)
   num_iterations -- number of iterations of the optimization loop
   learning_rate -- learning rate of the gradient descent update rule
 
    Returns:
   params -- dictionary containing the weights w and bias b
   grads -- dictionary containing the gradients of the weights and bias
with respect to the cost function
   costs -- list of all the costs computed during the optimization, this
will be used to plot the learning curve.
   Tips:
   You basically need to write down two steps and iterate through them:
       1) Calculate the cost and the gradient for the current parameters.
Use propagate().
       2) Update the parameters using gradient descent rule for w and b.
   """
   costs = []
   for i in range(num_iterations):
       # Cost and gradient calculation
       grads, cost = propagate(w, b, X, Y)
       # Retrieve derivatives from grads
       dw = grads["dw"]
       db = grads["db"]
# update rule
       w=w-(dw*learning_rate)
       b=b-(db*learning_rate)
       # Record the costs
       if i % 100 == 0:
           costs.append(cost)
           # Print the cost every 100 training examples
           print ("Cost after iteration %i: %f" % (i, cost))
   params = {"w": w,
             "b": b}
 
 grads = {"dw": dw,
         "db": db}
return params, grads, costs
 Make predictions
def predict(w, b, X):
   '''
   Predict whether the label is 0 or 1 using learned logistic regression
parameters (w, b)
   Arguments:
   w -- weights, a numpy array of size (num_px * num_px, 1)
   b -- bias, a scalar
   X -- data of size (num_px * num_px, number of examples)
Returns:
   Y_prediction -- a numpy array (vector) containing all predictions (0/1)
for the examples in X
'''
   m = X.shape[0]
   Y_prediction = np.zeros((1, m))
   w = w.reshape(X.shape[1], 1)
   # Compute vector "A" predicting the probabilities of the picture
containing a 1
   A = sigmoid(np.dot(w.T, X.T) + b)
   for i in range(A.shape[1]):
       # Convert probabilities A[0,i] to actual predictions p[0,i]
     if A[0][i]<0.5:
       Y_prediction[0][i]=0
     else:
       Y_prediction[0][i]=1
 
  assert(Y_prediction.shape == (1, m))
return Y_prediction
Merge functions and run your model
# LOAD DATA
class0 = 5
class1 = 6
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train[np.isin(y_train,[class0,class1]),:,:]
y_train = 1*(y_train[np.isin(y_train,[class0,class1])]>class0)
x_test = x_test[np.isin(y_test,[class0,class1]),:,:]
y_test = 1*(y_test[np.isin(y_test,[class0,class1])]>class0)
 x_train.shape[0]
plt.imshow(x_train[6],cmap="gray")
  # RESHAPE
x_train_flat = x_train.reshape(x_train.shape[0],-1)
print(x_train_flat.shape)
print('Train: '+str(x_train_flat.shape[0])+' images and
'+str(x_train_flat.shape[1])+' neurons \n')
x_test_flat = x_test.reshape(x_test.shape[0],-1)
print(x_test_flat.shape)
print('Test: '+str(x_test_flat.shape[0])+' images and
'+str(x_test_flat.shape[1])+' neurons \n')
# STRANDARIZE
x_train_flat = x_train_flat / 255
x_test_flat = x_test_flat / 255
 Train the model (in training set)
# Initialize parameters with zeros (≈ 1 line of code)
 
 w, b = initialize_weights(x_train_flat.shape[1])
# Gradient descent (≈ 1 line of code)
learning_rate = 0.005
num_iterations = 2000
parameters, grads, costs = gradient_descent(w, b, x_train_flat, y_train,
2000, 0.005)
 Test the model (in testing set)
# Retrieve parameters w and b from dictionary "parameters"
w = parameters["w"]
b = parameters["b"]
# Predict test/train set examples (≈ 2 lines of code)
y_prediction_test = predict(w, b, x_test_flat)
y_prediction_train = predict(w, b, x_train_flat)
# Print train/test Errors
print('')
print("train accuracy: {} %".format(100 -
np.mean(np.abs(y_prediction_train - y_train)) * 100))
print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test
- y_test)) * 100))
print('')
plt.figure(figsize=(13,5))
plt.plot(range(0,2000,100),costs)
plt.title('Cost training vs iteration')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.xticks(range(0,2000,100))
plt.figure(figsize=(13,5))
plt.imshow(w.reshape(28,28))
plt.title('Template')
 x_test_flat.shape
 
 w = parameters["w"]
b = parameters["b"]
 def TP(y,y_pred):
 count=0
 for i in range(0,len(y)):
   if y[i]==1 and y_pred[i]==1:
     count=count+1
 return count
def TN(y,y_pred):
 count=0
 for i in range(0,len(y)):
   if y[i]==0 and y_pred[i]==0:
     count=count+1
 return count
def FN(y,y_pred):
 count=0
 for i in range(0,len(y)):
   if y[i]==1 and y_pred[i]==0:
     count=count+1
 return count
def FP(y,y_pred):
 count=0
 for i in range(0,len(y)):
   if y[i]==0 and y_pred[i]==1:
     count=count+1
return count
 y_prediction_test.shape
y_test.shape
y_prediction_test=y_prediction_test.reshape(y_test.shape[0],)
y_prediction_test=np.array(list(map(int,y_prediction_test)))
y_prediction_test
tp=TP(y_test,y_prediction_test)
tn=TN(y_test,y_prediction_test)
       
fn=FN(y_test,y_prediction_test)
fp=FP(y_test,y_prediction_test)
balanced_accuracy=((tp/(tp+fn))+(tn/(tn+fp)))/2
balanced_accuracy
 
