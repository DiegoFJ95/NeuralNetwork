import math
import random

# Clase para un nodo de un grafo que describe operaciones matemáticas y que es capaz de hacer derivadas parciales.

class Node:
    def __init__(self, data, _children=(), _op='', label=''):

        # Valor numérico del nodo
        self.data= data
        # Operador usado para crear este nodo
        self._op = _op
        # Nombre del nodo (opcional)
        self.label = label
        # Gradiente de la función respecto a este nodo
        self.grad = 0.0
        # Nodos que fueron operados para crear el nodo actual
        self._prev = _children


    # Sobrecarga de operadores para poder hacer operaciones entre nodos.

    def __repr__(self):
        return f"Node(data={self.data})" #, children={self._prev})"
    
    def __add__(self, other):
        out = Node(self.data + other.data, (self, other), '+')
        return out
    
    def __mul__(self, other):
        out = Node(self.data * other.data, (self, other), '*')
        return out
    
    def __sub__(self, other):
        out = Node(self.data - other.data, (self, other), '-')
        return out
    
    def __pow__(self, other):
        out = Node(self.data**other.data, (self, other), '**')
        return out

    def __iadd__(self, other):
        total = self.data + other.data
        out = Node(total, (self, other), '+')
        return out
    

    # Funciones de activación que pueden aplicarse a un nodo.

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Node(t, (self, ), 'TanH')
        return out
    
    def RelU(self):
        x = self.data
        r = max(0, x)
        out = Node(r, (self, ), 'ReLU')
        return out
    

    # Derivadas para cada operación, con recursión para recorrer el grafo hacia atrás

    def backward(self):
        self.grad = 1
        def recurse(node):
            if not node._prev:
                return
            
            if node._op == '+':
                for n in node._prev:
                    n.grad += 1.0 * node.grad
                    recurse(n)

            if node._op == '-':
                node._prev[0].grad += 1.0 * node.grad
                recurse(node._prev[0])
                node._prev[1].grad += -1.0 * node.grad
                recurse(node._prev[1])

            if node._op == '*':
                node._prev[0].grad += node._prev[1].data * node.grad
                recurse(node._prev[0])
                node._prev[1].grad += node._prev[0].data * node.grad
                recurse(node._prev[1])
            
            if node._op == '**':
                exp = node._prev[1].data
                node._prev[0].grad += exp * (node._prev[0].data ** (exp - 1)) * node.grad
                recurse(node._prev[0])
                # recurse(node._prev[1])

            if node._op == 'TanH':
                node._prev[0].grad += (1 - node.data**2) * node.grad
                recurse(node._prev[0])

            if node._op == 'RelU':
                if node.data > 0:
                    node._prev[0].grad += 1 * node.grad
                else:
                    node._prev[0].grad += 0
                recurse(node._prev[0])
                      
        recurse(self)



# Clase para una neurona de la red neuronal. Usa nodos para representar las operaciones matemáticas que describen la neurona (pesos, bias, suma ponderada, función de activación).

class Neuron:
    def __init__(self, inputs, activation='TanH'):
        # Lista de pesos (uno por cada entrada)
        self.w = []
        self.activation = activation
        for _ in range(inputs):
            self.w.append(Node(random.uniform(-1, 1)))
        # Bias
        self.b = Node(random.uniform(-1,1))

        # print(self.w, self.b)

    # Regresa los parámetros de la neurona (pesos y bias) como nodos.
    def parameters(self):
        return self.w + [self.b]

    # Al llamar a la neurona con una lista de entradas regresa la salida de la neurona.
    def __call__(self, x):
        # Inicializar el total de la activación sumandole el bias.
        total = Node(self.b.data)

        # Crea pares de peso y entrada.
        for wi, xi in zip(self.w, x):
            # Si la entrada no es un nodo, lo convierte en uno para poderla operar.
            if type(xi) != Node:
                xi = Node(xi)
            # print('weight/input: ', wi, xi)
            # Multiplica el peso por la entrada y lo suma al total.
            activation = wi * xi
            total = total + activation
        # Normaliza la activación de la neurona
        if self.activation == 'TanH':
            output = total.tanh()
        elif self.activation == 'ReLU':
            output = total.RelU()
        else:
            output = total

        # out = activation.tanh()
        return output
    


# Clase para una capa de la red neuronal compuesta de varias neuronas. Está completamente conectada a la capa anterior.

class Layer:
    def __init__(self, inputs, outputs, activation):
        # Crea una lista de neuronas correspondiendo con el número de salidas de esa capa. Cada neurona está conectada a todas las entradas.
        self.neurons = []
        for _ in range(outputs):
            self.neurons.append(Neuron(inputs, activation=activation))
    
    # Al llamar la capa con una lista de entradas regresa la salida de cada neurona.
    def __call__(self, x):
        output = []
        for neuron in self.neurons:
            # print('neuron: ', neuron.w, neuron.b)
            output.append(neuron(x))
        return output[0] if len(output) == 1 else output
    
    # Regresa los parámetros de cada neurona de la capa como nodos. No necesita estar ordenada por que cada uno guarda su propio gradiente.
    def parameters(self):
        parameters = []
        for neuron in self.neurons:
            ps = neuron.parameters()
            parameters.extend(ps)
        return parameters
    


# Clase para una red neuronal densa. 

class NN:
    def __init__(self, inputs, layers, activations):
        # Crear una lista con el número de valores por capa incluyendo la entrada para poder crear las capas con el número de entradas y de neuronas correcto. 
        inout = [inputs] + layers 
        self.layers = []

        # Crea capas que tienen el número de entradas = al número de salidas de la capa anterior.
        for i in range(len(layers)):
            self.layers.append(Layer(inout[i], inout[i+1], activations[i]))
    
    # Al llamar la red neuronal con una lista de entradas regresa la salida de la última capa. Se llama de forma iterativa para ir actualizando los valores desde la primera capa.
    def __call__(self, x):
        for layer in self.layers:
            # print('layer: ', layer)
            x = layer(x)
        return x

    # Regresa lois parámetros de cada neurona de cada capa como nodos.
    def parameters(self):
        parameters = []
        for layer in self.layers:
            ps = layer.parameters()
            parameters.extend(ps)
        return parameters
    



# Red neuronal con 3 entradas, 2 capas de 4 neuronas y 1 neurona de salida.
n = NN(3, [4,4,1], ['TanH', 'TanH', 'TanH'])

# Entradas de ejemplo
xs = [[2.0, 3.0, -1.0],
      [3.0, -1.0, 0.5],
      [0.5, 1.0, 1.0],
      [1.0, 1.0, -1.0]]

# salidas de ejemplo
ys = [1.0, -1.0, -1.0, 1.0]

def predict(network, xs):
    prediction = []
    for x in xs:
        prediction.append(network(x))
    return prediction

ypred = predict(n, xs)

print( 'predicciones iniciales: ', ypred )

def train(network, xs, ys, epochs=1000, learning_rate=0.01):
    for i in range(2000):

        # Fordward
        ypred = predict(network, xs)
        # Calculo de la pérdida con el error cuadrático.
        loss = Node(0.0)
        for yt, yout in zip(ys, ypred):
            loss += (yout - Node(yt))**Node(2)

        # Backward 
        # Backpropagation calculando el gradiente de la función de pérdida respecto a cada parámetro.
        loss.backward()

        # Update
        for parameter in network.parameters():
            parameter.data += -learning_rate * parameter.grad
            parameter.grad = 0.0
        
        if i % 100 == 0:
            print(f"Epoch {i}, loss: {loss.data}")

    

train(n, xs, ys, epochs=2000, learning_rate=0.05)

print('predicciones finales: ', predict(n, xs))


import sys
sys.setrecursionlimit(100000)

import pandas as pd
df = pd.read_csv('StressLevelDataset.csv')
df.head()

y = df[['stress_level']]
x = df[['self_esteem', 'sleep_quality', 'depression']]


print(y)