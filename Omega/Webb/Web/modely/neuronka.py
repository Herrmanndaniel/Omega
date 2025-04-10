import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt

# Načtení a příprava dat
df = pd.read_csv('../cleaner/doopravdy_hotove_auta.csv')

# Úprava cílové proměnné
df['Kombinovaná'] = df['Kombinovaná'].str.replace(' l/100km', '').astype(float)
df['Výkon'] = df['Výkon'].str.replace('kW', '').astype(float)

# Feature engineering
df['Stáří'] = 2025 - df['Rok uvedení do provozu']
X = df.drop(['Kombinovaná', 'Rok uvedení do provozu'], axis=1)
y = df['Kombinovaná'].values

# Kódování kategoriálních proměnných
categorical_cols = ['Karoserie', 'Palivo', 'Motor']
encoder = OneHotEncoder(sparse_output=False)
X_cat = encoder.fit_transform(X[categorical_cols])

# Standardizace numerických vlastností
numerical_cols = ['Výkon', 'Stáří']
scaler = StandardScaler()
X_num = scaler.fit_transform(X[numerical_cols])

# Spojení vlastností
X_processed = np.hstack([X_cat, X_num])

# Rozdělení dat
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

# Konverze na PyTorch tensory
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)


# Definice neuronové sítě
class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Inicializace modelu
model = Net(X_train.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Trénování
train_losses = []
val_losses = []
best_loss = np.inf
patience = 10
trigger_times = 0

for epoch in range(200):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    # Backward pass
    loss.backward()
    optimizer.step()

    # Validace
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_test_tensor)
        val_loss = criterion(val_outputs, y_test_tensor)

    train_losses.append(loss.item())
    val_losses.append(val_loss.item())

    # Early stopping
    if val_loss < best_loss:
        best_loss = val_loss
        trigger_times = 0
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print(f'Early stopping at epoch {epoch}')
            break

# Predikce
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor).numpy().flatten()

# Výpočet MAE
mae = np.mean(np.abs(y_test - y_pred))
print(f'Test MAE: {mae:.2f}')

# Vizualizace
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Loss Evolution')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel('Skutečné hodnoty')
plt.ylabel('Predikce')
plt.title('Neuronová síť: Skutečné vs. Predikované hodnoty')
plt.tight_layout()
plt.show()

"""
Test MAE: 0.45
"""
