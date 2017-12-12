import pickle
import tensorflow as tf
from format_data import *

# getting the entire pickled formatted data
d = open('data.p', 'rb')
Data = pickle.load(d) 
X_all = Data['X']
Y_all = Data['Y']

# Leaving 100 datapoints for testing
X = X_all[0:-100]
Y = Y_all[0:-100]

# Taking the last 100 points for testing
X_test = X_all[-100:]
Y_test = Y_all[-100:]
nof_data_test = X_test.shape[0]

nof_data = X.shape[0]

learning_rate= 0.003
lstm_layer_units = 128
nof_iterations = 500
batch_size = 20

# Should match with the formatted values
input_size = 10
num_steps = 4

tf.reset_default_graph()
rnn_lstm = tf.Graph()

Costs = []

with rnn_lstm.as_default():

    lr = tf.constant(learning_rate)
    lstm_inputs = tf.placeholder(tf.float32, [None, num_steps, input_size])
    labels = tf.placeholder(tf.float32, [None, input_size])
    
    lstm_cell = tf.contrib.rnn.LSTMCell(lstm_layer_units, state_is_tuple = True)
    lstm_outputs, final_rnn_state = tf.nn.dynamic_rnn(lstm_cell, lstm_inputs, dtype = tf.float32)
    
    lstm_outputs = tf.transpose(lstm_outputs, [1, 0, 2])
    last_output = tf.gather(lstm_outputs, int(lstm_outputs.get_shape()[0]) - 1)

    W = tf.Variable(tf.truncated_normal([lstm_layer_units, input_size]))
    b = tf.Variable(tf.constant(0.1, shape = [1, input_size]))  
    Z = tf.matmul(last_output, W) + b
    
    cost = tf.reduce_mean(tf.square(Z - labels))
    optimizer = tf.train.RMSPropOptimizer(lr)
    minimize = optimizer.minimize(cost)

    # To test the test cost
    """
    lstm_inputs_test = tf.placeholder(tf.float32, [nof_data_test, num_steps, input_size])
    lstm_labels_test = tf.placeholder(tf.float32, [nof_data_test, input_size])

    lstm_outputs_test, final_rnn_state_test = tf.nn.dynamic_rnn(lstm_cell, lstm_inputs_test, dtype = tf.float32)
    lstm_outputs_test = tf.transpose(lstm_outputs_test, [1, 0, 2])
    last_output_test = tf.gather(lstm_outputs_test, int(lstm_outputs_test.get_shape()[0]) - 1)

    Z_test = tf.matmul(last_output_test, W) + b
    """


with tf.Session(graph = rnn_lstm) as sess:
    
    tf.global_variables_initializer().run()
    X_batches, Y_batches = data_batches(X, Y, batch_size) 

    for i in range(nof_iterations):
        tc = 0

        for j in range(batch_size):
            p_values = {
                lstm_inputs : X_batches[j],
                labels : Y_batches[j],
            }
            train_cost, c = sess.run([cost, minimize], p_values)
            tc += train_cost

        Costs.append(tc)
        print('Cost after iteration ' + str(i) +' : ' + str(tc))   



    """
    if i == nof_iterations - 1:
        test_p_values = {
            lstm_inputs_test : X_test,
            lstm_labels_test : Y_test,
        }   
        print(sess.run(Z_test))  
    """    

    #saver = tf.train.Saver()
    #saver.save(sess, 'model_2.ckpt', global_step = nof_iterations)      
    
# Storing the decreaing cost values during training    
cost_graph = {}
cost_graph['Costs'] = Costs   
fileObject = open('cost_graph.p','wb')
#pickle.dump(cost_graph, fileObject)    