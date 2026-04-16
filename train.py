import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load the Dataset
# Replace with your actual dataset path
df = pd.read_csv("amc_dataset.csv")

# Define features and labels
features = [
    "Tx_Power_dBm",
    "Target_SNR_dB",
    "Mean_Channel_Gain",
    "Noise_Power_Watts",
    "Phase_Noise_Var",
]
X = df[features].values

# Map string labels to integers (0 to 5)
scheme_map = {"BPSK": 0, "QPSK": 1, "8PSK": 2, "16QAM": 3, "64QAM": 4, "256QAM": 5}
y = df["Label_Best_Scheme"].map(scheme_map).values

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.LongTensor(y_train)


# 2. Define the Neural Network
class AdaptiveModNN(nn.Module):
    def __init__(self):
        super(AdaptiveModNN, self).__init__()
        self.fc1 = nn.Linear(5, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 6)  # 6 output classes

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


model = AdaptiveModNN()

# 3. Train the Model
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

print("Training Network...")
for epoch in range(200):
    optimizer.zero_grad()
    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)
    loss.backward()
    optimizer.step()

# 4. Export the Model Weights to MATLAB format
with open("matlab_weights.txt", "w") as f:
    f.write("%% --- NEURAL NETWORK WEIGHTS ---\n")
    for name, param in model.named_parameters():
        var_name = name.replace(".", "_")
        val = param.detach().numpy()
        if val.ndim == 2:
            # Matrix (Weights)
            rows = ["  " + " ".join([f"{x:.8f}" for x in row]) for row in val]
            matrix_str = "; ...\n".join(rows)
            f.write(f"{var_name} = [ ...\n{matrix_str} ...\n];\n\n")
        else:
            # Vector (Biases)
            vec_str = " ".join([f"{x:.8f}" for x in val])
            f.write(f"{var_name} = [{vec_str}]';\n\n")

print(
    "Model trained! Open 'matlab_weights.txt' and copy its contents into your MATLAB script."
)
print("Scaler Means:", scaler.mean_)
print("Scaler Variances:", scaler.var_)
