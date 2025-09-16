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
    
    def __truediv__(self, other):
        out = Node(self.data / other.data, (self, other), '/')
        return out

    # Funciones de activación que pueden aplicarse a un nodo.

    def TanH(self):
        x = self.data
        if x > 100:
            x = 100
        elif x < -100:
            x = -100
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Node(t, (self, ), 'TanH')
        return out
    
    def RelU(self):
        x = self.data
        alpha = 0.01 # Leaky RelU
        r = max(alpha * x, x)
        out = Node(r, (self, ), 'RelU')
        return out
    

    def Softmax(self, others):
        # max_val = max(n.data for n in [self] + others)
        exps = [math.exp(n.data) for n in [self] + others]
        sum_exp = sum(exps)
        softmax_val = exps[0] / sum_exp
        out = Node(softmax_val, ([self] + others), 'Softmax')
        return out


    def backward(self):
        self.grad = 1

            # Crear lista topológicamente ordenada
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        # Construir orden topológico
        build_topo(self)

        for node in reversed(topo):
            if node._op == '+':
                for n in node._prev:
                    n.grad += 1.0 * node.grad

            elif node._op == '-':
                node._prev[0].grad += 1.0 * node.grad
                node._prev[1].grad += -1.0 * node.grad

            elif node._op == '*':
                node._prev[0].grad += node._prev[1].data * node.grad
                node._prev[1].grad += node._prev[0].data * node.grad
            
            elif node._op == '**':
                exp = node._prev[1].data
                node._prev[0].grad += exp * (node._prev[0].data ** (exp - 1)) * node.grad
                # recurse(node._prev[1])

            elif node._op == 'TanH':
                node._prev[0].grad += (1 - node.data**2) * node.grad
                # self._prev[0].recurse()

            elif node._op == 'RelU':
                if node.data > 0:
                    node._prev[0].grad += 1 * node.grad
                else:
                    node._prev[0].grad += 0.01 * node.grad
                # node._prev[0].recurse()
            
            elif node._op == '/':
                node._prev[0].grad += (1/ node._prev[1].data) * node.grad
                node._prev[1].grad += ((-node._prev[0].data) / (node._prev[1].data**2))* node.grad 

            elif node._op == 'Softmax':
                softmax_out = node.data
                for i, n in enumerate(node._prev):
                    if i == 0:
                        n.grad += softmax_out * (1 - softmax_out) * node.grad
                    else:
                        n.grad += -softmax_out * n.data * node.grad



# Clase para una neurona de la red neuronal. Usa nodos para representar las operaciones matemáticas que describen la neurona (pesos, bias, suma ponderada, función de activación).

class Neuron:
    def __init__(self, inputs, activation):
        # Lista de pesos (uno por cada entrada)
        self.w = []
            # Inicialización He para ReLU
        if activation == 'RelU':
            scale = math.sqrt(2.0/inputs)
            for _ in range(inputs):
                self.w.append(Node(random.uniform(-scale, scale)))
        elif activation == 'TanH':
            for _ in range(inputs):
                self.w.append(Node(random.uniform(-1, 1)))
        # Bias
        self.b = Node(random.uniform(-1,1))

        self.activation = activation

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
            output = total.TanH()
        elif self.activation == 'RelU':
            output = total.RelU()

        # out = activation.tanh()
        return output
    


# Clase para una capa de la red neuronal compuesta de varias neuronas. Está completamente conectada a la capa anterior.

class Layer:
    def __init__(self, inputs, outputs, activation):
        # Crea una lista de neuronas correspondiendo con el número de salidas de esa capa. Cada neurona está conectada a todas las entradas.
        self.neurons = []
        for _ in range(outputs):
            self.neurons.append(Neuron(inputs, activation))
    
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
    def __init__(self, inputs, layers, activation):
        # Crear una lista con el número de valores por capa incluyendo la entrada para poder crear las capas con el número de entradas y de neuronas correcto. 
        inout = [inputs] + layers 
        self.layers = []

        # Crea capas que tienen el número de entradas = al número de salidas de la capa anterior.
        for i in range(len(layers)):
            self.layers.append(Layer(inout[i], inout[i+1], activation))
    
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
n = NN(3, [4,4,1], 'RelU')

# Entradas de ejemplo
xs = [[2.0, 3.0, -1.0],
      [3.0, -1.0, 0.5],
      [0.5, 1.0, 1.0],
      [1.0, 1.0, -1.0]]

# salidas de ejemplo
ys = [1.0, 0.0, 0.0, 1.0]

def predict(network, xs):
    prediction = []
    for x in xs:
        prediction.append(network(x))
    return prediction

ypred = predict(n, xs)

print('Iniciando entrenamiento de prueba para comprobar que el modelo se puede entrenar. Los valores finales deben ser 1, -1, -1, 1')
print( 'predicciones iniciales: ', ypred )

def train(network, xs, ys, epochs=1000, learning_rate=0.001, printability=500):
    for i in range(epochs):

        # Fordward
        ypred = predict(network, xs)
        # Calculo de la pérdida con el error cuadrático.
        loss = Node(0.0)
        for yt, yout in zip(ys, ypred):
            
            # Convert single values to list
            if isinstance(yt, (int, float)):
                yt = [yt]
            
            # Ensure prediction is in correct format
            if not isinstance(yout, list):
                yout = [yout]

            for target, pred in zip(yt, yout):
                loss += (pred - Node(target))**Node(2)
                
        loss = loss * Node(1/len(ys))
        # Backward 
        # Backpropagation calculando el gradiente de la función de pérdida respecto a cada parámetro.
        loss.backward()

        # Update
        for parameter in network.parameters():
            parameter.data += -learning_rate * parameter.grad
            parameter.grad = 0.0
        
        if i % printability == 0:
            print(f"Epoch {i}, loss: {loss.data}")

    

train(n, xs, ys, epochs=2000, learning_rate=0.001)

print('predicciones finales: ', predict(n, xs))




import pandas as pd
df = pd.read_csv('StressLevelDataset.csv')
df.head()


y = df[['stress_level']]
x = df[['self_esteem', 'sleep_quality', 'depression']]

# x = df[['anxiety_level', 'self_esteem', 'depression', 'headache', 'blood_pressure', 'sleep_quality', 'breathing_problem', 'noise_level', 'living_conditions', 'safety', 'basic_needs', 'academic_performance', 'study_load', 'teacher_student_relationship', 'future_career_concerns', 'social_support', 'peer_pressure', 'extracurricular_activities', 'bullying']]


y = y/2 # Normalizar el estres de -1 a 1


import numpy as np


def normalize_data(data):
    # Calcular min y max para cada columna
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    
    # Evitar división por cero y agregar pequeño epsilon
    range_vals = max_vals - min_vals
    epsilon = 1e-8
    range_vals = np.where(range_vals < epsilon, epsilon, range_vals)
    
    # Normalizar a rango [-0.9, 0.9] para evitar saturación
    normalized = -0.9 + 1.8 * (data - min_vals) / range_vals
    return normalized

x_normalized = normalize_data(np.array(x))


# Xtrain = np.array(x[:-100])
# Xtest = np.array(x[-100:])

Xtrain = x_normalized[:-100]
Xtest = x_normalized[-100:]

Ytrain = np.array(y[:-100])
Ytest = np.array(y[-100:])

X_train_list = Xtrain.tolist()
Y_train_list = Ytrain.tolist()

Y_test_list = Ytest.tolist()
X_test_list = Xtest.tolist()


import sys
sys.setrecursionlimit(100000) # Aumentar el límete de recursión para evitar errores al hacer backpropagation en redes grandes.




stress_predictor = NN(3, [12, 8, 4, 1], 'RelU')
# stress_predictor(X_test_list[7])





print('\n\n\nIniciando entrenamiento real sobre el dataset de estrés hasta 200 iteraciones, puede tomar un tiempo... \n')

train(stress_predictor, X_train_list, Y_train_list, epochs=200, learning_rate=0.001, printability=10) # Printability es cada cuantas iteraciones se imprime el error actual.


import matplotlib.pyplot as plt



# Obtener resultados de train

y_pred_train = []
for x in X_train_list:
    pred = stress_predictor(x)
    # Convertir Node a valor float
    if isinstance(pred, list):
        y_pred_train.append([p.data for p in pred])
    else:
        y_pred_train.append(pred.data)

# Convertir predicciones y valores reales a arrays numpy
y_pred_train = np.array(y_pred_train)
y_true = np.array([yt[0] for yt in Y_train_list])  # Extraer valores de las listas


# Crear índices ordenados basados en los valores reales
indices_ordenados = np.argsort(y_true)

# Reordenar tanto los valores reales como las predicciones
y_true_ordenado = y_true[indices_ordenados]
y_pred_train_ordenado = y_pred_train[indices_ordenados]




# Obtener resultados de test

y_pred_test = []
for x in X_test_list:
    pred = stress_predictor(x)
    # Convertir Node a valor float
    if isinstance(pred, list):
        y_pred_test.append([p.data for p in pred])
    else:
        y_pred_test.append(pred.data)

# Convertir predicciones y valores reales a arrays numpy
y_pred_test = np.array(y_pred_test)
y_true_test = np.array([yt[0] for yt in Y_test_list])  # Extraer valores de las listas

# Crear índices ordenados basados en los valores reales
indices_ordenados = np.argsort(y_true_test)

# Reordenar tanto los valores reales como las predicciones
y_true__test_ordenado = y_true_test[indices_ordenados]
y_pred_test_ordenado = y_pred_test[indices_ordenados]




# Crear una figura con dos subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

# Primera gráfica (datos de entrenamiento)
ax1.scatter(range(len(y_true_ordenado)), y_true_ordenado, c='blue', label='Valor Real', alpha=0.5)
ax1.scatter(range(len(y_pred_train_ordenado)), y_pred_train_ordenado, c='red', label='Predicción', alpha=0.5)
ax1.set_xlabel('Índice de Muestra')
ax1.set_ylabel('Nivel de Estrés')
ax1.set_title('Comparación con datos de entrenamiento')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Segunda gráfica (datos de prueba)
ax2.scatter(range(len(y_true__test_ordenado)), y_true__test_ordenado, c='blue', label='Valor Real', alpha=0.5)
ax2.scatter(range(len(y_pred_test_ordenado)), y_pred_test_ordenado, c='red', label='Predicción', alpha=0.5)
ax2.set_xlabel('Índice de Muestra')
ax2.set_ylabel('Nivel de Estrés')
ax2.set_title('Comparación con datos de prueba')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Ajustar el espacio entre subplots
plt.tight_layout()

# Mostrar la figura
plt.show()

# Imprimir métricas de error
mse_train = np.mean((y_true - y_pred_train)**2)
mse_test = np.mean((y_true_test - y_pred_test)**2)
print(f"Error cuadrático medio (entrenamiento): {mse_train:.4f}")
print(f"Error cuadrático medio (prueba): {mse_test:.4f}")