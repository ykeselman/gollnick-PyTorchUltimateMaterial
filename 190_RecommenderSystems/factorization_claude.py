# %%

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

USER_ID = 'userId'
MOVIE_ID = 'movieId'

class MatrixFactorization(nn.Module):
    def __init__(self, n_users, n_movies, n_factors=50, bias=False):
        super(MatrixFactorization, self).__init__()
        
        # User and movie embedding layers
        self.user_factors = nn.Embedding(n_users, n_factors)
        self.movie_factors = nn.Embedding(n_movies, n_factors)
        
        # Bias terms (optional but usually helpful)
        self.use_bias = bias
        if bias:
            self.user_bias = nn.Embedding(n_users, 1)
            self.movie_bias = nn.Embedding(n_movies, 1)
            self.global_bias = nn.Parameter(torch.zeros(1))
        
        # Initialize embeddings
        self._init_weights()
    
    def _init_weights(self):
        # Initialize with small random values
        nn.init.normal_(self.user_factors.weight, std=0.1)
        nn.init.normal_(self.movie_factors.weight, std=0.1)
        
        if self.use_bias:
            nn.init.normal_(self.user_bias.weight, std=0.01)
            nn.init.normal_(self.movie_bias.weight, std=0.01)
    
    def forward(self, user_ids, movie_ids):
        # Get embeddings
        user_embedding = self.user_factors(user_ids)
        movie_embedding = self.movie_factors(movie_ids)
        
        # Compute dot product (element-wise multiplication then sum)
        dot_product = (user_embedding * movie_embedding).sum(dim=1)
        
        if self.use_bias:
            # Add bias terms
            user_b = self.user_bias(user_ids).squeeze()
            movie_b = self.movie_bias(movie_ids).squeeze()
            prediction = dot_product + user_b + movie_b + self.global_bias
        else:
            prediction = dot_product
            
        return prediction

class MovieRecommender:
    def __init__(self, n_factors=50, learning_rate=0.01, reg_lambda=0.001, 
                 n_epochs=100, batch_size=32, use_bias=True):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.use_bias = use_bias
        
        self.user_encoder = {}
        self.movie_encoder = {}
        self.user_decoder = {}
        self.movie_decoder = {}
        
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def _encode_ids(self, df):
        """Encode user and movie IDs to continuous integers starting from 0"""
        # Create mappings for users
        unique_users = df[USER_ID].unique()
        self.user_encoder = {user_id: idx for idx, user_id in enumerate(unique_users)}
        self.user_decoder = {idx: user_id for user_id, idx in self.user_encoder.items()}
        
        # Create mappings for movies
        unique_movies = df[MOVIE_ID].unique()
        self.movie_encoder = {movie_id: idx for idx, movie_id in enumerate(unique_movies)}
        self.movie_decoder = {idx: movie_id for movie_id, idx in self.movie_encoder.items()}
        
        # Apply encoding
        df['user_idx'] = df[USER_ID].map(self.user_encoder)
        df['movie_idx'] = df[MOVIE_ID].map(self.movie_encoder)
        
        return df
    
    def prepare_data(self, df, test_size=0.2, random_state=42):
        """Prepare and split the data"""
        # Encode IDs
        df = self._encode_ids(df.copy())
        
        # Split data
        train_df, test_df = train_test_split(df, test_size=test_size, 
                                           random_state=random_state, 
                                           stratify=df['rating'])
        
        self.n_users = df['user_idx'].nunique()
        self.n_movies = df['movie_idx'].nunique()
        
        print(f"Dataset info:")
        print(f"Users: {self.n_users}, Movies: {self.n_movies}")
        print(f"Training samples: {len(train_df)}, Test samples: {len(test_df)}")
        
        return train_df, test_df
    
    def _create_dataloader(self, df, shuffle=True):
        """Create PyTorch DataLoader"""
        dataset = torch.utils.data.TensorDataset(
            torch.LongTensor(df['user_idx'].values),
            torch.LongTensor(df['movie_idx'].values),
            torch.FloatTensor(df['rating'].values)
        )
        
        return torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=shuffle
        )
    
    def train(self, train_df, val_df=None):
        """Train the matrix factorization model"""
        # Initialize model
        self.model = MatrixFactorization(
            self.n_users, self.n_movies, 
            self.n_factors, self.use_bias
        ).to(self.device)
        
        # Setup optimizer and loss
        optimizer = optim.Adam(self.model.parameters(), 
                              lr=self.learning_rate, 
                              weight_decay=self.reg_lambda)
        criterion = nn.MSELoss()
        
        # Create data loaders
        train_loader = self._create_dataloader(train_df, shuffle=True)
        val_loader = self._create_dataloader(val_df, shuffle=False) if val_df is not None else None
        
        # Training loop
        train_losses = []
        val_losses = []
        
        for epoch in range(self.n_epochs):
            # Training
            self.model.train()
            train_loss = 0
            for user_ids, movie_ids, ratings in train_loader:
                user_ids = user_ids.to(self.device)
                movie_ids = movie_ids.to(self.device)
                ratings = ratings.to(self.device)
                
                optimizer.zero_grad()
                predictions = self.model(user_ids, movie_ids)
                loss = criterion(predictions, ratings)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Validation
            val_loss = 0
            if val_loader:
                self.model.eval()
                with torch.no_grad():
                    for user_ids, movie_ids, ratings in val_loader:
                        user_ids = user_ids.to(self.device)
                        movie_ids = movie_ids.to(self.device)
                        ratings = ratings.to(self.device)
                        
                        predictions = self.model(user_ids, movie_ids)
                        loss = criterion(predictions, ratings)
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                val_losses.append(val_loss)
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                if val_loader:
                    print(f'Epoch [{epoch+1}/{self.n_epochs}], '
                          f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
                else:
                    print(f'Epoch [{epoch+1}/{self.n_epochs}], Train Loss: {train_loss:.4f}')
        
        return train_losses, val_losses
    
    def predict(self, user_ids, movie_ids):
        """Make predictions for given user-movie pairs"""
        if not isinstance(user_ids, (list, np.ndarray)):
            user_ids = [user_ids]
        if not isinstance(movie_ids, (list, np.ndarray)):
            movie_ids = [movie_ids]
            
        # Encode IDs
        try:
            user_indices = [self.user_encoder[uid] for uid in user_ids]
            movie_indices = [self.movie_encoder[mid] for mid in movie_ids]
        except KeyError as e:
            raise ValueError(f"Unknown ID encountered: {e}")
        
        # Convert to tensors
        user_tensor = torch.LongTensor(user_indices).to(self.device)
        movie_tensor = torch.LongTensor(movie_indices).to(self.device)
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(user_tensor, movie_tensor)
            
        return predictions.cpu().numpy()
    
    def recommend_movies(self, user_id, n_recommendations=10, exclude_rated=True):
        """Recommend top N movies for a user"""
        if user_id not in self.user_encoder:
            raise ValueError(f"Unknown user ID: {user_id}")
        
        user_idx = self.user_encoder[user_id]
        
        # Get all movie indices
        all_movie_indices = list(range(self.n_movies))
        
        # Predict ratings for all movies
        user_indices = [user_idx] * self.n_movies
        
        predictions = []
        batch_size = 1000  # Process in batches to avoid memory issues
        
        for i in range(0, len(all_movie_indices), batch_size):
            batch_users = user_indices[i:i + batch_size]
            batch_movies = all_movie_indices[i:i + batch_size]
            
            batch_predictions = self.predict(
                [self.user_decoder[u] for u in batch_users],
                [self.movie_decoder[m] for m in batch_movies]
            )
            predictions.extend(batch_predictions)
        
        # Create recommendations dataframe
        recommendations = pd.DataFrame({
            MOVIE_ID: [self.movie_decoder[idx] for idx in all_movie_indices],
            'predicted_rating': predictions
        })
        
        # Sort by predicted rating
        recommendations = recommendations.sort_values('predicted_rating', ascending=False)
        
        return recommendations.head(n_recommendations)
    
    def evaluate(self, test_df):
        """Evaluate model performance on test set"""
        test_loader = self._create_dataloader(test_df, shuffle=False)
        
        predictions = []
        actuals = []
        
        self.model.eval()
        with torch.no_grad():
            for user_ids, movie_ids, ratings in test_loader:
                user_ids = user_ids.to(self.device)
                movie_ids = movie_ids.to(self.device)
                
                batch_predictions = self.model(user_ids, movie_ids)
                predictions.extend(batch_predictions.cpu().numpy())
                actuals.extend(ratings.numpy())
        
        mse = mean_squared_error(actuals, predictions)
        rmse = np.sqrt(mse)
        
        print(f"Test RMSE: {rmse:.4f}")
        return rmse, predictions, actuals

# Example usage and data loading
def load_and_preprocess_data(file_path):
    """
    Load movie ratings data from file
    Expected format: user_id, movie_id, rating
    """
    # Adjust column names based on your file format
    df = pd.read_csv(file_path)
    
    # Ensure correct column names
    expected_columns = [USER_ID, MOVIE_ID, 'rating']
    if not all(col in df.columns for col in expected_columns):
        print("Expected columns: user_id, movie_id, rating")
        print(f"Found columns: {df.columns.tolist()}")
        # You might need to rename columns here
        
    # Basic data cleaning
    df = df.dropna()
    print(f"Loaded {len(df)} ratings")
    print(f"Rating range: {df['rating'].min()} - {df['rating'].max()}")
    
    return df

# Main training pipeline
def train_movie_recommender(file_path, n_factors=50, n_epochs=100, learning_rate=0.01):
    """Complete training pipeline"""
    
    # Load data
    df = load_and_preprocess_data(file_path)
    
    # Initialize recommender
    recommender = MovieRecommender(
        n_factors=n_factors,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        reg_lambda=0.001,
        batch_size=1024
    )
    
    # Prepare data
    train_df, test_df = recommender.prepare_data(df, test_size=0.2)
    
    # Split training data for validation
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)
    
    # Train model
    print("Starting training...")
    train_losses, val_losses = recommender.train(train_df, val_df)
    
    # Evaluate
    print("\nEvaluating on test set...")
    rmse, predictions, actuals = recommender.evaluate(test_df)
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Curves')
    
    plt.subplot(1, 2, 2)
    plt.scatter(actuals[:1000], predictions[:1000], alpha=0.5)
    plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], 'r--')
    plt.xlabel('Actual Ratings')
    plt.ylabel('Predicted Ratings')
    plt.title('Predictions vs Actuals')
    plt.tight_layout()
    plt.show()
    
    return recommender

# %%
# Example usage:
import os
os.chdir('/home/yakov/Studies/gollnick-PyTorchUltimateMaterial/190_RecommenderSystems')

recommender = train_movie_recommender('ratings.csv')
recommendations = recommender.recommend_movies(user_id=1, n_recommendations=10)
