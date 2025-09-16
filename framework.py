import torch
from torch import nn
import pandas as pd

# Definir la arquitectura de la red
class StressPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(5, 8),    
            nn.Tanh(),      
            nn.Linear(8, 6),    
            nn.Tanh(), 
            nn.Linear(6, 4),    
            nn.Tanh(), 
            nn.Linear(4, 1),   
            nn.Tanh()          
        )

    def forward(self, x):
        return self.network(x)
    



    # Preparar los datos
def prepare_data(x, y):
    x_tensor = torch.FloatTensor(x)
    y_tensor = torch.FloatTensor(y)
    return torch.utils.data.TensorDataset(x_tensor, y_tensor)

# Cargar y normalizar datos
df = pd.read_csv('StressLevelDataset.csv')
x = df[['self_esteem', 'sleep_quality', 'depression', 'anxiety_level', 'headache']]
y = df[['stress_level']]
y = (y - 1)  # Normalizar estrés a [-1,1]

# Normalizar features
# x = (x - x.min()) / (x.max() - x.min()) * 2 - 1

# Separar train/test
X_train = x[:-100].values
X_test = x[-100:].values
y_train = y[:-100].values
y_test = y[-100:].values

# Crear datasets
train_dataset = prepare_data(X_train, y_train)
test_dataset = prepare_data(X_test, y_test)

# Crear dataloaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

# Inicializar modelo, pérdida y optimizador
model = StressPredictor()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)



# Entrenamiento
num_epochs = 200
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # Backward pass y optimización
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}')


import matplotlib.pyplot as plt
import numpy as np

# Evaluación
model.eval()
with torch.no_grad():
    # Predicciones de entrenamiento
    X_train_tensor = torch.FloatTensor(X_train)
    y_pred_train = model(X_train_tensor).numpy()
    
    # Predicciones de prueba
    X_test_tensor = torch.FloatTensor(X_test)
    y_pred_test = model(X_test_tensor).numpy()

# Ordenar por valores reales
train_indices = np.argsort(y_train.flatten())
test_indices = np.argsort(y_test.flatten())

# Crear una figura con dos subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

# Gráfica de entrenamiento
ax1.scatter(range(len(y_train)), y_train[train_indices], c='blue', label='Valor Real', alpha=0.5)
ax1.scatter(range(len(y_pred_train)), y_pred_train[train_indices], c='red', label='Predicción', alpha=0.5)
ax1.set_xlabel('Índice de Muestra')
ax1.set_ylabel('Nivel de Estrés')
ax1.set_title('Comparación con datos de entrenamiento')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Gráfica de prueba
ax2.scatter(range(len(y_test)), y_test[test_indices], c='blue', label='Valor Real', alpha=0.5)
ax2.scatter(range(len(y_pred_test)), y_pred_test[test_indices], c='red', label='Predicción', alpha=0.5)
ax2.set_xlabel('Índice de Muestra')
ax2.set_ylabel('Nivel de Estrés')
ax2.set_title('Comparación con datos de prueba')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Métricas de error
mse_train = np.mean((y_train - y_pred_train)**2)
mse_test = np.mean((y_test - y_pred_test)**2)
print(f"Error cuadrático medio (entrenamiento): {mse_train:.4f}")
print(f"Error cuadrático medio (prueba): {mse_test:.4f}")