import numpy as np
import random
import matplotlib.pyplot as plt
import pyswarms as ps
from deap import algorithms, base, creator, tools

creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

class GA(object):
    def __init__(self, inputX, actual_y, seed = 50):
        self.randomseed = seed
        self.X = inputX
        self.y = actual_y
        self.bias = 1

        random.seed(self.randomseed)

    # Activation function which is sigmoid
    def sigmoid(self, z):
        return 1 / (1 + np.exp((-1) * z))

    def hidden(self, IP1, IP2, wt1, wt2, wt0):
        inducedfield = IP1*wt1 + IP2*wt2 + self.bias*wt0
        return self.sigmoid(inducedfield)

    def finaloutput(self, IP1, IP2, IP3, IP4, wt1, wt2, wt3, wt4, wt0):
        inducedfield = IP1*wt1 + IP2*wt2 + IP3*wt3 + IP4*wt4 + self.bias*wt0
        return self.sigmoid(inducedfield)
    
    
    # Evaluation function.
    def Evaluator(self, individual, XOR = True):
        
        differenceTotalXor = 0
        if XOR:
            for i in range(len(self.X)):
                actualValueXor = self.y[i]
                valueFromHid1 = self.hidden(self.X[i][0], self.X[i][1], individual[0], individual[1], individual[2])
                valueFromHid2 = self.hidden(self.X[i][0], self.X[i][1], individual[3], individual[4], individual[5])
                valueFromHid3 = self.hidden(self.X[i][0], self.X[i][1], individual[6], individual[7], individual[8])
                valueFromHid4 = self.hidden(self.X[i][0], self.X[i][1], individual[9], individual[10], individual[11])
                valueFromfinal = self.finaloutput(valueFromHid1, valueFromHid2, valueFromHid3, valueFromHid4,
                                individual[12], individual[13], individual[14], individual[15], individual[16])
                differenceXor = (actualValueXor - valueFromfinal)**2
                
                differenceTotalXor +=  differenceXor
               
            
            return (differenceTotalXor,)
        else:
            for i in range(len(self.X)):
                actualValueXNor = self.y[i]
                valueFromHid1 = self.hidden(self.X[i][0], self.X[i][1], individual[0], individual[1], individual[2])
                valueFromHid2 = self.hidden(self.X[i][0], self.X[i][1], individual[3], individual[4], individual[5])
                valueFromHid3 = self.hidden(self.X[i][0], self.X[i][1], individual[6], individual[7], individual[8])
                valueFromHid4 = self.hidden(self.X[i][0], self.X[i][1], individual[9], individual[10], individual[11])
                valueFromfinal = self.finaloutput(valueFromHid1, valueFromHid2, valueFromHid3, valueFromHid4,
                                individual[12], individual[13], individual[14], individual[15], individual[16])
                differenceXNor = (actualValueXNor - valueFromfinal)**2
                differenceTotalXor +=  differenceXNor
            return (differenceTotalXor,)


# XOR Problem 

inputXOR = [[0,0],[0,1],[1,0],[1,1]]
outputXOR = [[0],[1],[1],[0]]


XOR_ev = GA(inputXOR, outputXOR, seed = 50)
# 17 random floating numbers between -10 and 10. 3 for each of the 4 nodes in the hidden layer including the bias
# (12 in total for this layer), 5 for the output node including its bias.
toolbox = base.Toolbox()
toolbox.register("attr_bool", random.uniform, -10, 10)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_bool, n = 17)
toolbox.register("population", tools.initRepeat, list, 
                 toolbox.individual)
# blend of crossover and gaussian mutation
# Selection is tournament selection with 3 individuals
toolbox.register("evaluate", XOR_ev.Evaluator)
toolbox.register("mate", tools.cxBlend,alpha = 0.5)
toolbox.register("mutate", tools.mutGaussian, mu = 0.0,sigma= 0.5, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize = 3)
# 500 individual population and 100 generations
pop = toolbox.population(n = 300)
result = algorithms.eaSimple(pop, toolbox, cxpb = 0.5, mutpb = 0.15, ngen = 80, verbose = False)

bestIndividual = tools.selBest(pop, k=1)[0]
# Printing out the results
print ('The XOR minimum loss value is', XOR_ev.Evaluator(bestIndividual)[0])
print ('The XOR weights for first Hidden node are', bestIndividual[1],'and', bestIndividual[2], 'The bias weight is ', bestIndividual[0])
print ('The XOR weights for second Hidden node are',bestIndividual[3],'and',bestIndividual[4],'The bias weight is ',bestIndividual[5])
print ('The XOR weights for third Hidden node are',bestIndividual[6],'and',bestIndividual[7],'The bias weight is ',bestIndividual[8])
print ('The  XOR weights for fourth Hidden node are',bestIndividual[9],'and',bestIndividual[10],'The bias weight is ',bestIndividual[11])
print ('The XOR weights for the output node are',bestIndividual[12], bestIndividual[13], bestIndividual[14],'and', bestIndividual[15],'The bias weight is ',bestIndividual[16])
print
for i in range(4):
    print ('with inputs (',XOR_ev.X[i][0],",",XOR_ev.X[i][1],"),")
    valueFromHid1 = XOR_ev.hidden(XOR_ev.X[i][0],XOR_ev.X[i][1],bestIndividual[1],bestIndividual[2],bestIndividual[0])
    valueFromHid2 = XOR_ev.hidden(XOR_ev.X[i][0],XOR_ev.X[i][1],bestIndividual[3],bestIndividual[4],bestIndividual[5])
    valueFromHid3 = XOR_ev.hidden(XOR_ev.X[i][0],XOR_ev.X[i][1],bestIndividual[6],bestIndividual[7],bestIndividual[8])
    valueFromHid4 = XOR_ev.hidden(XOR_ev.X[i][0],XOR_ev.X[i][1],bestIndividual[9],bestIndividual[10],bestIndividual[11])
    final_OP = XOR_ev.finaloutput(valueFromHid1, valueFromHid2, valueFromHid3, valueFromHid4, bestIndividual[12],bestIndividual[13],bestIndividual[14], bestIndividual[15], bestIndividual[16])
    print ('output of optimized XOR was ', final_OP)


print("\n")


# XNOR

inputXNOR = [[0,0],[0,1],[1,0],[1,1]]
outputXNOR = [[1],[0],[0],[1]]


XNOR_ev = GA(inputXNOR, outputXNOR, seed = 50)
# 17 random floating numbers between -10 and 10. 3 for each of the 4 nodes in the hidden layer including the bias
# (12 in total for this layer), 5 for the output node including its bias.
toolbox = base.Toolbox()
toolbox.register("attr_bool", random.uniform, -10, 10)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_bool, n = 17)
toolbox.register("population", tools.initRepeat, list, 
                 toolbox.individual)
# blend of crossover and gaussian mutation
# Selection is tournament selection with 3 individuals
toolbox.register("evaluate", XNOR_ev.Evaluator)
toolbox.register("mate", tools.cxBlend, alpha = 0.5)
toolbox.register("mutate", tools.mutGaussian, mu = 0.0,sigma = 0.5, indpb = 0.05)
toolbox.register("select", tools.selTournament, tournsize = 3)
# 500 individual population and 100 generations
pop = toolbox.population(n= 300)
result = algorithms.eaSimple(pop, toolbox, cxpb=0.6, mutpb=0.15, ngen= 80, verbose=False)

bestIndividual = tools.selBest(pop, k=1)[0]
# Printing out the results
print ('The XNOR minimum loss value is', XNOR_ev.Evaluator(bestIndividual, XOR = False)[0])
print ('The XNOR weights for first Hidden node are', bestIndividual[1],'and', bestIndividual[2], 'The bias weight is ', bestIndividual[0])
print ('The XNOR weights for second Hidden node are',bestIndividual[3],'and',bestIndividual[4],'The bias weight is ',bestIndividual[5])
print ('The XNOR weights for third Hidden node are',bestIndividual[6],'and',bestIndividual[7],'The bias weight is ',bestIndividual[8])
print ('The XNOR weights for fourth Hidden node are',bestIndividual[9],'and',bestIndividual[10],'The bias weight is ',bestIndividual[11])
print ('The XNOR weights for the output node are',bestIndividual[12], bestIndividual[13], bestIndividual[14],'and', bestIndividual[15],'The bias weight is ',bestIndividual[16])
print
for i in range(4):
    print ('with inputs (',XNOR_ev.X[i][0],",",XNOR_ev.X[i][1],"),")
    valueFromHid1 = XNOR_ev.hidden(XNOR_ev.X[i][0],XNOR_ev.X[i][1],bestIndividual[1],bestIndividual[2],bestIndividual[0])
    valueFromHid2 = XNOR_ev.hidden(XNOR_ev.X[i][0],XNOR_ev.X[i][1],bestIndividual[3],bestIndividual[4],bestIndividual[5])
    valueFromHid3 = XNOR_ev.hidden(XNOR_ev.X[i][0],XNOR_ev.X[i][1],bestIndividual[6],bestIndividual[7],bestIndividual[8])
    valueFromHid4 = XNOR_ev.hidden(XNOR_ev.X[i][0],XNOR_ev.X[i][1],bestIndividual[9],bestIndividual[10],bestIndividual[11])
    print ('output of optimized XNOR was ', XNOR_ev.finaloutput(valueFromHid1, valueFromHid2, valueFromHid3, valueFromHid4, bestIndividual[12],bestIndividual[13],bestIndividual[14], bestIndividual[15], bestIndividual[16]))



class PSO(object):
    def __init__(self, X, y, num_inputs, num_hidden, num_classes, num_samples):
        self.X = X
        self.y = y
        self.n_inputs = num_inputs
        self.n_hidden = num_hidden
        self.n_classes = num_classes
        self.n_samples = num_samples
    
    def forwardPass(self, p):
        # Roll-back the weights and biases
        W1 = p[0:8].reshape((self.n_inputs,self.n_hidden))
        b1 = p[8:12].reshape((self.n_hidden,))
        W2 = p[12:16].reshape((self.n_hidden,self.n_classes))
        b2 = p[16:17].reshape((self.n_classes,))

        # Pre-activation in Layer 1
        v1 = self.X.dot(W1) + b1 

        # Activation in Layer 1 
        a1 = self.sigmoid(v1)  

        # Pre-activation in Layer 2   
        v2 = a1.dot(W2) + b2 

        # Activation in Layer 2
        a2 = self.sigmoid(v2)

        return a2         


    def sigmoid(self, Z):
        return 1 / (1 + np.exp((-1) * Z))

    # # Forward propagation
    def error_calc(self,params):

        network_scores = self.forwardPass(params)
        loss = np.sum((self.y - network_scores)**2)

        return loss

    def f(self, x):

        n_particles = x.shape[0]
        j = [self.error_calc(x[i]) for i in range(n_particles)]
        
        return np.array(j)

    def predict(self, pos):
        
        prediction = self.forwardPass(pos)
        return prediction


X = [[0,0],[0,1],[1,0],[1,1]]
X = np.array(X)
options = {'c1': 0.6, 'c2': 0.3, 'w':0.9}

# XOR
XOR_y = [[0], [1], [1], [0]]
XOR_y = np.array(XOR_y)

# instance of XOR PSO
XOR_pso = PSO(X, XOR_y, 2, 4, 1, 4)

dimensions = (XOR_pso.n_inputs * XOR_pso.n_hidden) + (XOR_pso.n_hidden * XOR_pso.n_classes) + XOR_pso.n_hidden + XOR_pso.n_classes
optimizer = ps.single.GlobalBestPSO(n_particles=100, dimensions=dimensions, options=options)

# Perform optimization
cost, pos = optimizer.optimize(XOR_pso.f, iters=5000)

print("XOR prediction \n", XOR_pso.predict(pos))

# XNOR

XNOR_y = [[1],[0],[0],[1]]
XNOR_y = np.array(XNOR_y )


# instance of XNOR PSO
XNOR_pso = PSO(X, XNOR_y, 2, 4, 1, 4)

dimensions = (XNOR_pso.n_inputs * XNOR_pso.n_hidden) + (XNOR_pso.n_hidden * XNOR_pso.n_classes) + XNOR_pso.n_hidden + XNOR_pso.n_classes
optimizer = ps.single.GlobalBestPSO(n_particles=80, dimensions=dimensions, options=options)

# Perform optimization
cost, pos = optimizer.optimize(XNOR_pso.f, iters=5000)

print("XNOR prediction \n", XNOR_pso.predict(pos))