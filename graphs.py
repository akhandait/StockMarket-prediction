import pickle
import matplotlib.pyplot as plt

# getting the entire pickled formatted data
d = open('data.p', 'rb')
Data = pickle.load(d) 
X_all = Data['X']
Y_all = Data['Y']

# Taking the last 100 points for testing
X_test = X_all[-100:]
Y_test = Y_all[-100:]

d2 = open('cost_graph.p', 'rb')
cost_graph = pickle.load(d2)

Costs = cost_graph['Costs']

Costs = Costs[7::50]
plt.plot(Costs)
plt.title('Cost Variation')
plt.xlabel('Number of Iterations / 50')
plt.ylabel('Cost')
plt.show()