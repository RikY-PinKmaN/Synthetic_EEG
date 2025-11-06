import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class AutoencoderTrainer:
    def __init__(self, input_dim, hidden_dim, learning_rate=0.001):
        self.model = Autoencoder(input_dim, hidden_dim)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def train(self, train_data, epochs, batch_size, save_path):
        # Prepare DataLoader
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

        # Training loop
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch in train_loader:
                self.optimizer.zero_grad()
                inputs = batch[0]
                outputs = self.model(inputs)
                loss = self.criterion(outputs, inputs)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}")

        # Save the trained model
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    def load_model(self, load_path):
        self.model.load_state_dict(torch.load(load_path))
        print(f"Model loaded from {load_path}")
