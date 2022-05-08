# Neural_Network_Charity_Analysis

## Overview of the analysis

This analysis is to use the target and features in the Alphabet Soup dataset provided to predict whether applicants will be successful in funding Alphabet Soup. 
The Alphabet Soup dataset contains more than 34,000 organizations that have received funding from Alphabet Soup over the years. We will be using machine learning and nerual netwroks for this analysis. 

## Results 

### Data Preprocessing

- The 'IS_SUCCESSFUL' column is the variable we are targeting for this model. This column identifies whether the donation was used effectively. 

```
# Split our preprocessed data into our features and target arrays
y = application_df.IS_SUCCESSFUL
X = application_df.drop("IS_SUCCESSFUL",axis=1)

# Split the preprocessed data into a training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=78)
```

- The "APPLICATION_TYPE","AFFILIATION","CLASSIFICATION","USE_CASE","ORGANIZATION","INCOME_AMT", and "SPECIAL_CONSIDERATIONS" columns are considered to be features for this model. These are all object data types. 

![image](https://user-images.githubusercontent.com/91445591/167315905-173f614e-fb8a-4fa7-9d47-1001fd8d0437.png)


- the EIN and NAME columns have been removed from the input data since they are identification columns and are neither targets nor feratures. 

```
application_df = application_df.drop(columns=['EIN', 'NAME'])
application_df.head()
```

### Compiling, Training, and Evaluating the Model

The neural network model has two hidden layers. The first hidden layer has 80 neurons with a rulu function.
The second hidden layer consists of 30 neurons and a relu function. The output uses the sigmoid function with 1 neuron.

```
# Define the model - deep neural net, i.e., the number of input features and hidden nodes for each layer.
input_features = len(X_train_scaled[0])
hidden_layer1 =  80
hidden_layer2 = 30

nn = tf.keras.models.Sequential()

# First hidden layer
nn.add(tf.keras.layers.Dense(units=hidden_layer1, input_dim=input_features, activation="relu"))

# Second hidden layer
nn.add(tf.keras.layers.Dense(units=hidden_layer2, activation="relu"))

# Output layer
nn.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

# Check the structure of the model
nn.summary()
```

I was unable to achieve the target model performance of over 75%. In order to try and increase the model performance, I took the following steps:

- Dropped the STATUS cloumn
- Modified the APPLICATION_TYPE and CLASSIFICATION bins from grouping based on valued less than 500 and 1800 to values less than 100 and 300.
- Added two hidden layers and modified the activation function to include Linear and Tanh. 
- I also increased the number of epochs to 120

These steps did not have a significant enough impact to surpass the target 75% model performance.

### Summary: 

The overall results of the deep learning model were unsuccessful. The model accuracy returned was about 72% with a significant model loss of 56%.
An alternative model could be used to solve this classification problem such as using Rain Forest Deep Learning and comparing it to the Deep Learning Neural Network model used. 
