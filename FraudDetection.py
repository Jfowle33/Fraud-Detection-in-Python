import tensorflow as tf
import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

credit_card_data = pd.read_csv('creditcard.csv')
# print(credit_card_data)

# 1. shuffle/randomize data
# 2. one-hot encoding
# 3. normalize the data
# 4. splitting up x and y values
# 5. convert dataframes to numpy arrays(float32)
# 6. splitting the final data into x/y train/test

# shuffle and randomizing the data
shuffled_data = credit_card_data.sample(frac=1)
# print(shuffled_data)
# change class column into Class_0 ((0 1) for legit data) and Class_1((1 0) for fraudulent data)
one_hot_data = pd.get_dummies(shuffled_data, columns=['Class'])
# print(one_hot_data)
# change all values into numbers between 0 and 1
normalized_data = (one_hot_data - one_hot_data.min()) / (one_hot_data.max() - one_hot_data.min())
# print(normalized_data)
# store just columns V1 through V28 in df_X columns and columns Class_0 and Class_1  in df_Y
df_X = normalized_data.drop(['Class_0', 'Class_1'], axis=1)
# print(df_X)
df_Y = normalized_data[['Class_0', 'Class_1']]
# print(df_Y)
# convert both dataframes in np arrays of float32
ar_x, ar_Y = np.asarray(df_X.values, dtype='float32'), np.asarray(df_Y.values, dtype='float32')
# print(ar_x, ar_Y)
# allocate the first 80% of the data into training data, the remaining 20% into testing data
train_size = int(0.8 * len(ar_x))
# print(train_size)
(raw_X_train, raw_Y_train) = (ar_x[:train_size], ar_Y[:train_size])
# print(raw_X_train, raw_Y_train)
(raw_X_test, raw_Y_test) = (ar_x[train_size:], ar_Y[train_size:])
# print(raw_X_test,raw_Y_test)

count_legit, count_fraud = np.unique(credit_card_data['Class'], return_counts=True)[1]
fraud_ratio = float(count_fraud/(count_legit + count_fraud))
print('Percent of fraudulent transactions', fraud_ratio)

# applies a Logit weighing of 578 (1/.0017) to fraudulent transactions to cause model to pay more attention to them
weighing = 1 / fraud_ratio
raw_Y_train[:, 1] = raw_Y_train[:, 1] * weighing
# 30 cells for the input
input_dimensions = ar_x.shape[1]
# 2 cells for the output
output_dimensions = ar_Y.shape[1]
# 100 cells for the first layer
num_layer1_cells = 100
# 150 cells for the second layer
num_layer2_cells = 150

# Use this as inputs to the model when it comes time to train the model
x_train_node = tf.placeholder(tf.float32, [None, input_dimensions], name = "X_train")
y_train_node = tf.placeholder(tf.float32, [None, output_dimensions], name = "Y_train")

# These are going to be used for testing the model
X_test_node = tf.constant(raw_X_test, name="X_test")
Y_test_node = tf.constant(raw_Y_test, name="Y_test")

# first layer takes input and passes it to the 2nd layer
weight1_node = tf.Variable(tf.zeros([input_dimensions, num_layer1_cells]), name='Weight1')
biases1_node = tf.Variable(tf.zeros([num_layer1_cells]), name='Biases_1')

# 2nd layer takes first layer output and passes it to the 3rd layer
weight2_node = tf.Variable(tf.zeros([num_layer1_cells, num_layer2_cells]), name='Weight2')
biases2_node = tf.Variable(tf.zeros([num_layer2_cells]), name='Biases_2')

# 3rd layer takes 2nd layer output as input and outputs [0 1] or [1 0] whether fraud or legit
weight3_node = tf.Variable(tf.zeros([num_layer2_cells, output_dimensions]), name='Weight3')
biases3_node = tf.Variable(tf.zeros([output_dimensions]), name='Biases_3')

# Function to run an input tensor through the 3 layers and output a tensor that will give us a fraud/legit result
# each layer uses a different function to fit lines through the data and predict whether a given input tensor will
# result in a fraudulent or legitimate transaction
def network(input_tensor):
    # sigmoid fits modeled data well
    layer1 = tf.nn.sigmoid(tf.matmul(input_tensor, weight1_node) + biases1_node)
    # dropout function prevents model from becoming lazy and over confident
    layer2 = tf.nn.dropout(tf.nn.sigmoid(tf.matmul(layer1, weight2_node) + biases2_node), 0.85)
    # softmax works very well with hot encoding which is how results are outputted
    layer3 = tf.nn.softmax(tf.matmul(layer2, weight3_node) + biases3_node)
    return layer3

# used to predict what results will be given for training or testing input data
# remember x_train_node is just a placeholder for now. will enter values at runtime
y_train_predict = network(x_train_node)
y_test_predict = network(X_test_node)

# cross entropy loss function measures differences between actual and predicted outputs
cross_entropy = tf.losses.softmax_cross_entropy(y_train_node, y_train_predict)

# AdamOptimizer will try to minimize the loss (cross_entropy) but changing the 3 layers' variable values
# at a learning rate of .005
optimizer = tf.train.AdamOptimizer(0.005).minimize(cross_entropy)

# function to calculate the accuracy of the actual result vs the predicted result
def calculate_accuracy(actual, predicted):
    actual = np.argmax(actual, 1)
    predicted = np.argmax(predicted, 1)
    return 100 * np.sum(np.equal(predicted, actual)) / predicted.shape[0]

num_epochs = 100

import time

with tf.Session() as session:
    tf.global_variables_initializer().run()
    for epoch in range(num_epochs):
        start_time = time.time()

        _, cross_entropy_score = session.run([optimizer, cross_entropy],
                                             feed_dict={x_train_node: raw_X_train, y_train_node: raw_Y_train})
        if epoch % 10 == 0:
            timer = time.time() - start_time
            print('Epoch: {}'.format(epoch), 'Current Loss {0:.4f}'.format(cross_entropy_score),
                  'Elapsed {0:.2f} seconds'.format(timer))

            final_y_test = Y_test_node.eval()
            final_y_test_prediction = y_test_predict.eval()
            final_accuracy = calculate_accuracy(final_y_test, final_y_test_prediction)
            print("Current accuracy: {:.4f}%".format(final_accuracy))
    final_y_test = Y_test_node.eval()
    final_y_test_prediction = y_test_predict.eval()
    final_accuracy = calculate_accuracy(final_y_test, final_y_test_prediction)
    print("Final accuracy: {:.4f}%".format(final_accuracy))

final_fraud_y_test = final_y_test[final_y_test[:, 1] == 1]
final_fraud_y_test_prediction = final_y_test_prediction[final_y_test[:, 1] == 1]
final_fraud_accuracy = calculate_accuracy(final_fraud_y_test, final_fraud_y_test_prediction)
print("Final fraud specific accuracy: {:.2f}%".format(final_fraud_accuracy))
