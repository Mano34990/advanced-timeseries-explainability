import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from lstm_model import LSTMForecastModel

# ----------------------------
# Load Dataset
# ----------------------------
df = pd.read_csv("multivariate_series.csv")

# Detect feature columns (all except date & target)
feature_cols = [c for c in df.columns if c not in ["date", "target"]]
target_col = "target"

X = df[feature_cols].values
y = df[target_col].shift(-1).fillna(method='ffill').values

# Convert to tensors
X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (batch, seq=1, features)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# ----------------------------
# Initialize Model
# ----------------------------
input_size = len(feature_cols)
model = LSTMForecastModel(input_size=input_size, hidden_size=64, num_layers=2)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ----------------------------
# Training Loop
# ----------------------------
epochs = 5
for epoch in range(epochs):
    total_loss = 0
    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

# ----------------------------
# Save Model
# ----------------------------
torch.save(model.state_dict(), "best_lstm_model.pth")
print("Model saved as best_lstm_model.pth")
