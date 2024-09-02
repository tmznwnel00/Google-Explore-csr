import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

node_embeddings_array = np.load('final_embeddings.npy')
node_ids = node_embeddings_array[:, 0]
node_embeddings = node_embeddings_array[:, 1:]

node_embeddings = node_embeddings.astype(float)

node_embeddings_tensor = torch.tensor(node_embeddings, dtype=torch.float32)  

df = pd.read_csv('feature_vectors.csv') 

node_ids_features = df.iloc[:, 0].values  
node_features_df = df.iloc[:, 1:]  
node_features = node_features_df.values 

node_features = node_features.astype(float)

node_features_tensor = torch.tensor(node_features, dtype=torch.float32)

node_id_to_features = dict(zip(node_ids_features, node_features_tensor))

matching_node_ids = [node_id for node_id in node_ids if node_id in node_id_to_features]
matching_node_embeddings = [node_embeddings_tensor[node_ids == node_id][0] for node_id in matching_node_ids]
matching_node_features = torch.stack([node_id_to_features[node_id] for node_id in matching_node_ids])

matching_node_embeddings_tensor = torch.stack(matching_node_embeddings)

class LinearTransform(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearTransform, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False) 

    def forward(self, x):
        return self.linear(x)

model = LinearTransform(input_dim=matching_node_embeddings_tensor.shape[1], output_dim=matching_node_features.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

num_epochs = 1000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = model(matching_node_embeddings_tensor)
    loss = criterion(output, matching_node_features)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

feature_embedding = model(matching_node_embeddings_tensor).detach().numpy()

np.save('feature_embedding.npy', feature_embedding)
