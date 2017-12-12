import pandas as pd
import numpy as np
import pickle

# Formatting the closing prices
def format_data(closing_prices, input_size, num_steps):

    nof_slots = closing_prices.shape[0] // input_size
    slots = []
    for i in range(nof_slots):
        slots.append(closing_prices[input_size*i:input_size*(i+1)])
    
    last_oflast = slots[0][0]
    for i in range(len(slots)):
        last_oflast1 = slots[i][-1]
        slots[i] = slots[i] / last_oflast
        last_oflast = last_oflast1
        
    nof_dpoints = nof_slots - num_steps
    X1 = []
    Y1 = []
    for i in range(nof_dpoints):
        X2 = []
        for j in range(num_steps):
            X2.append(slots[i+j])
        X1.append(X2)
        
        Y1.append(slots[i+num_steps])

    X = np.array(X1)
    Y = np.array(Y1)
 
    Data = {}
    Data['X'] = X
    Data['Y'] = Y
    
    return Data


df = pd.read_csv("S&P500.csv")
closing_prices = df['Close'].values
input_size = 10
num_steps = 4

Data = format_data(closing_prices, input_size, num_steps)

fileObject = open('data.p','wb')
pickle.dump(Data, fileObject)


def data_batches(X, Y, batch_size):
    
    nof_batches = len(X) // batch_size
    if batch_size * nof_batches < len(X):
        nof_batches += 1
    
    X_batches = []
    Y_batches = []
    for i in range(nof_batches):
        X_batches.append(X[i*batch_size:(i+1)*batch_size])
        Y_batches.append(Y[i*batch_size:(i+1)*batch_size])
    
    X_batches = np.array(X_batches)
    Y_batches = np.array(Y_batches)
    
    return X_batches, Y_batches
        

        