Load and visualize MNIST da t a
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from skimage import measure
import warnings
warnings.filterwarnings("ignore") # Added this at the end to show a clean
output with no warnings but not necessary
Choose number to visualize (from 0 to 9):
(x_train, y_train), (x_test, y_test) = mnist.load_data()
number = 0
x = x_train[y_train==number,:,:]
print('The shape of x is:')
print(x.shape)
print('which means:')
print('Number '+str(number)+' has '+str(x.shape[0])+' images of size
'+str(x.shape[1])+'x'+str(x.shape[2]))
Plot average image:
m = np.mean(x, axis=0) # IMPORTANT: indexes in python start at "0", not
"1", so the first element of array "a" would be a[0]
plt.figure()
plt.subplot(1,2,1)
plt.imshow(m)
plt.title('Average image')
mt = 1*(m > 60) # Thresholding
plt.subplot(1,2,2)
 plt.imshow(mt)
plt.title('Thresholded image')
 From a thresholded image, we can use the regionprops function from skimage.measure
mt_props = measure.regionprops(mt)
num_regions = len(mt_props)
print(str(num_regions)+' region/s were found')
print('')
print('Area (in pixels):')
area = mt_props[0].area # Remember, index 0 is the first region found
print(area)
print('')
print('Perimeter (in pixels):')
perimeter = mt_props[0].perimeter
print(perimeter)
print('')
print('Centroid (pixel coordinates):')
centroid = mt_props[0].centroid
print(centroid)
print('Eccentricity:')
eccentricity = mt_props[0].eccentricity
print(eccentricity)
print('')
print('Minor axis length:')
minor_axis = mt_props[0].minor_axis_length
print(minor_axis)
print('')
 Example: Scatter plot of Area vs Perimeter for all images of numbers "number" and "number+1"

 Are "Area" and "Perimeter" good features to classify "number" and "number+1"?
x0 = x_train[y_train==number,:,:]
x1 = x_train[y_train==number+1,:,:]
buf0 = "Number %d" % number
buf1 = "Number %d" % (number+1)
# Threshold images
t0 = 1*(x0 > 60)
t1 = 1*(x1 > 60)
# Region properties
area0 = np.zeros(t0.shape[0])
perimeter0 = np.zeros(t0.shape[0])
for i in range(0,t0.shape[0]):
 props = measure.regionprops(t0[i,:,:])
 area0[i] = props[0].area
 perimeter0[i] = props[0].perimeter
area1 = np.zeros(t1.shape[0])
perimeter1 = np.zeros(t1.shape[0])
for i in range(0,t1.shape[0]):
 props = measure.regionprops(t1[i,:,:])
 area1[i] = props[0].area
 perimeter1[i] = props[0].perimeter
plt.figure(figsize=(20,5))
plt.subplot(1,2,1)
plt.scatter(area0,perimeter0, label=buf0)
plt.scatter(area1,perimeter1, label=buf1)
plt.title('All images for both classes')
plt.legend()
plt.subplot(1,2,2)
plt.scatter(area0[0:100],perimeter0[0:100], label=buf0)
plt.scatter(area1[0:100],perimeter1[0:100], label=buf1)
plt.title('100 images of each class')
plt.legend()
  
 # Function to calculate mean vector of features for different classes
def mean_vect(area_x, perimeter_x):
 mean = np.array([np.mean(area_x), np.mean(perimeter_x)])
 return mean
 # Mean Vector Array
mean0 = mean_vect(area0, perimeter0)
print("Mean of Class 0:\n", mean0)
mean1 = mean_vect(area1, perimeter1)
print("Mean of Class 1:\n", mean1)
#Mean Array of Class 0 and Class 1 features
mean_vector = np.array([mean0, mean1])
print("Total Mean Vector: \n", mean_vector)
shape_matrix = np.array([t0.shape[0], t1.shape[0]])
area_matrix = np.array([area0,area1])
perimeter_matrix = np.array([perimeter0,perimeter1])
matrix_0 = np.array([area0,perimeter0]).T
matrix_1 = np.array([area1, perimeter1]).T
overall_mean = (mean0+mean1)/2
Sw = np.zeros((2,2))
Sw_mat_0 = np.zeros((2,2))
Sw_mat_1= np.zeros((2,2))
Sb = np.zeros((2,2))
for i in range(t0.shape[0]):
 row, mv =
np.array([matrix_0[i][:2]]).reshape(2,1),mean_vector[0].reshape(2,1)
 Sw_mat_0 += (row-mv).dot((row-mv).T)
for i in range(t1.shape[0]):
 row, mv =
np.array([matrix_1[i][:2]]).reshape(2,1),mean_vector[1].reshape(2,1)
 Sw_mat_1 += (row-mv).dot((row-mv).T)
Sw = Sw_mat_0 + Sw_mat_1
print('Within class Matrix:\n SW', Sw)
for i in range (2):
 
  n=shape_matrix[i]
 mv=mean_vector[i].reshape(2,1)
 overall_mean = overall_mean.reshape(2,1)
 Sb += n*(overall_mean-mv).dot((overall_mean-mv).T)
print('\n Between class Matrix:\n SB', Sb)
e_vals, e_vecs = np.linalg.eig(np.linalg.inv(Sw).dot(Sb))
print('Eigenvectors \n%s' %e_vecs)
print('\nEigenvalues \n%s' %e_vals)
e_pairs=[(np.abs(e_vals[i]),e_vecs[:,i]) for i in range(len(e_vals))]
e_pairs=sorted(e_pairs,key=lambda k: k[0], reverse=True)
print('\nEigenvalues in decreasing order:\n')
for i in e_pairs:
print(i[0])
W = np.hstack(e_pairs[0][1].reshape(2,1))
W= W.reshape(2,1)
print('Mat W:\n', W.real)
X_0 =np.dot(np.array([area0 ,perimeter0]).T,W)
X_1 =np.dot(np.array([area1 ,perimeter1]).T,W)
lin0=np.zeros((1,shape_matrix[0]))
lin1=np.ones((1,shape_matrix[1]))
plt.figure()
plt.hist(X_0,label = 'Number 0')
plt.hist(X_1, label = 'Number 1')
plt.legend()
threshold = -60
values0 = 0
a0 = 0
failed0 = {}
for i,x in enumerate(X_0) :
 if x < threshold :
   values0 = values0 + 1
else :
 
    failed0[a0] = i
   a0 = a0 + 1
print ("The number of 0s in training sample: ", (X_0.shape[0]))
print ("True Negative TN : ", values0)
print ("False Positive FP : ", a0)
values1 = 0
a1 = 0
failed1 = {}
for i,x in enumerate(X_1) :
 if x > threshold :
   values1 = values1 + 1
 else :
   failed1[a1] = i
   a1 = a1 + 1
print ("\nThe number of 1s in training sample: ", (X_1.shape[0]))
print ("True Positive TP : ", values1)
print ("False Negative FN : ", a1)
print("\n Accuracy : ",((values0+values1)/((values0+values1+a1+a0))*100))
Sensitivity = values1/(a1+values1)
Specificity = values0/(a0+values0)
balanced_accuracy = (Sensitivity + Specificity)/2
print ("\n The Balanced accuracy: ",(balanced_accuracy*100))
x_test_0 = x_test[y_test==0,:,:]
x_test_1 = x_test[y_test==1,:,:]
t0 = 1*(x_test_0 > 60)
t1 = 1*(x_test_1 > 60)
area_test0 = np.zeros(t0.shape[0])
perimeter_test0 = np.zeros(t0.shape[0])
for i in range(0,t0.shape[0]):
 props = measure.regionprops(t0[i,:,:])
 area_test0[i] = props[0].area
 perimeter_test0[i] = props[0].perimeter
area_test1 = np.zeros(t1.shape[0])
perimeter_test1 = np.zeros(t1.shape[0])
for i in range(0,t1.shape[0]):
 
  props = measure.regionprops(t1[i,:,:])
 area_test1[i] = props[0].area
 perimeter_test1[i] = props[0].perimeter
test_matrix_shape = np.array([t0.shape[0],t1.shape[0]])
X_test_lda_0 = np.dot(np.array([area_test0,perimeter_test0]).T,W)
X_test_lda_1 = np.dot(np.array([area_test1,perimeter_test1]).T,W)
lin0=np.zeros((1,test_matrix_shape[0]))
lin1=np.ones((1,test_matrix_shape[1]))
plt.figure()
plt.hist(X_test_lda_0,label = 'Number 0')
plt.hist(X_test_lda_1, label = 'Number 1')
plt.legend()
plt.show()
print ("The number of 0s in test sample: ", (x_test_0.shape[0]))
threshold = -60
count0 = 0
k0 = 0
failed0 = {}
for i,x in enumerate(X_test_lda_0) :
   if x < threshold :
     count0 = count0 + 1
   else :
     failed0[k0] = i
     k0 = k0 + 1
print ("True Negative TN : ", count0)
print ("False Positive FP: ", k0)
print ("\nThe number of 1s in test sample: " , (x_test_1.shape[0]))
threshold = -60
count1 = 0
k1 = 0
failed1 = {}
for i,x in enumerate(X_test_lda_1) :
 if x > threshold :
   count1 = count1 + 1
 else :
   failed1[k1] = i
 
k1 = k1 + 1
print ("True Positive TP : ", count1)
print ("False Negative FN : ", k1)
Sensitivity = count1/(k1+count1)
Specificity = count0/(k0+count0)
balanced_accuracy = (Sensitivity + Specificity)/2
print ("\n The Balanced accuracy: ", (balanced_accuracy*100))
