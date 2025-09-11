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
