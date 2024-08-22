import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Load node embedding vector from .npy file
node_embeddings_array = np.load('final_embeddings.npy')
node_ids = node_embeddings_array[:, 0]
node_embeddings = node_embeddings_array[:, 1:]

node_embeddings = node_embeddings.astype(float)

# Convert to tensor
node_embeddings_tensor = torch.tensor(node_embeddings, dtype=torch.float32)  # Convert to tensor

# Load node feature matrix from CSV file
df = pd.read_csv('feature_vectors.csv')  # Load CSV into DataFrame

# Extract features and node IDs
node_ids_features = df.iloc[:, 0].values  # Get node IDs from CSV
node_features_df = df.iloc[:, 1:]  # Get feature columns
node_features = node_features_df.values  # Convert to NumPy array

# Ensure node_features is numeric
node_features = node_features.astype(float)

# Convert to tensor
node_features_tensor = torch.tensor(node_features, dtype=torch.float32)  # Convert to tensor

# Create a mapping from node IDs to features
node_id_to_features = dict(zip(node_ids_features, node_features_tensor))

# Filter node_embeddings_tensor and align it with node_features_tensor
matching_node_ids = [node_id for node_id in node_ids if node_id in node_id_to_features]
matching_node_embeddings = [node_embeddings_tensor[node_ids == node_id][0] for node_id in matching_node_ids]
matching_node_features = torch.stack([node_id_to_features[node_id] for node_id in matching_node_ids])

# Convert lists to tensors
matching_node_embeddings_tensor = torch.stack(matching_node_embeddings)

print(matching_node_embeddings_tensor.shape)
print(matching_node_features.shape)

# Define Linear Transformation Model
class LinearTransform(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearTransform, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)  # No bias

    def forward(self, x):
        return self.linear(x)

# Initialize model, loss function, and optimizer
model = LinearTransform(input_dim=matching_node_embeddings_tensor.shape[1], output_dim=matching_node_features.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = model(matching_node_embeddings_tensor)
    loss = criterion(output, matching_node_features)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Get the feature embedding vector
feature_embedding = model(matching_node_embeddings_tensor).detach().numpy()

# Optionally, save the feature embedding vector
np.save('feature_embedding.npy', feature_embedding)
