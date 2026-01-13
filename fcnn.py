# crucial steps Loading ds, splitting ds , eda of ds

# Defining Model Architecture of FCNN
# We will be using tensor flow's api known as keras to build our neural network

model = keras.Sequential([
    keras.layers.Flatten(input_shape=info.features['image'].shape),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(info.features['label'].num_classes, activation='softmax')
])

# in above code, keras.Sequential is building our 3 layered neural network sequentially
#
# Input Layer :  Flatten Layer
#
keras.layers.Flatten(input_shape = info.features['image'].shape)
''' where first layer is input layer which is known as flatten layer because our images ds is 3d 
but our connected layer known as Dense in neural network only works with 1d ds
so flatten layer is just unrolling the 3d images and converting all the images into 1d vector so that we can pass our
input features to the dense layer'''

#
# Hidden Layer : Dense layer + Actn Fn[ReLU - rectified linear unit] 
#
keras.layers.Dense(512, activation = 'relu' )

''' 512 = numbers of neurons in a particular layer, we can choose no of neurons randomly but it should be in the power of 2^n 

activation fn. = relu 

#Function of Dense Layer 
whenever any input feature passes through dense layer along with its weight we calculate its pre-activation or logit
then we pass the pre-activation to relu
fn of relu.: f(x) = max(0,x)
    if pre-activation = -ve:
        activation = 0
    else 
        activation = pre-activation


with the help of relu we calculate activation and pass that activation to outputlayer or next dense layer as per your model achit...

Achitecture of H.Layer


i/p feature ---> linear fn. (calc. pre activation z = (i/p * weight)+bias ----> relu (calc f(z) = max(0,z)) ---> o/p layer'''


#
# Output Layer
#
keras.layers.Dense(n, activation = 'softmax')
'''n = no of neurons you want in your output layer which is nothing but total no of categories in your ds

fn of softmax activation function

it converts the floating value output into probablity so that we can understand the confidense level = 1-alpha of a particular output
