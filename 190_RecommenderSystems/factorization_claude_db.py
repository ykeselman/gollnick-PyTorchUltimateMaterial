# %%
# Main training pipeline with debugging
def train_movie_recommender(file_path, n_factors=20, n_epochs=100, learning_rate=0.001):
    """Complete training pipeline with debugging"""
    
    # Load and diagnose data
    print("=== LOADING AND DIAGNOSING DATA ===")
    df = load_and_preprocess_data(file_path, clean=True, min_user_ratings=10, min_movie_ratings=10)
    
    if len(df) < 1000:
        print("WARNING: Very small dataset after cleaning. Consider reducing min_ratings thresholds.")
    
    # Initialize recommender with conservative hyperparameters
    recommender = MovieRecommender(
        n_factors=n_factors,  # Start smaller
        n_epochs=n_epochs,
        learning_rate=learning_rate,  # Lower learning rate
        reg_lambda=0.01,  # Higher regularization
        batch_size=min(1024, len(df) // 10)  # Adaptive batch size
    )
    
    # Prepare data
    print("\n=== PREPARING DATA ===")
    train_df, test_df = recommender.prepare_data(df, test_size=0.2)
    
    # Further split training data for validation
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)
    
    print(f"Final split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Train model
    print("\n=== TRAINING MODEL ===")
    train_losses, val_losses = recommender.train(train_df, val_df)
    
    # Evaluate
    print("\n=== EVALUATING MODEL ===")
    rmse, predictions, actuals = recommender.evaluate(test_df)
    
    # Enhanced plotting with correlation analysis
    plt.figure(figsize=(15, 5))
    
    # Training curves
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Training Loss', alpha=0.8)
    plt.plot(val_losses, label='Validation Loss', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.title('Training Curves')
    plt.yscale('log')  # Log scale often reveals more detail
    
    # Predictions vs Actuals
    plt.subplot(1, 3, 2)
    plt.scatter(actuals[:2000], predictions[:2000], alpha=0.5, s=1)
    plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], 'r--', linewidth=2)
    plt.xlabel('Actual Ratings')
    plt.ylabel('Predicted Ratings')
    plt.title('Predictions vs Actuals')
    
    # Add correlation coefficient
    correlation = np.corrcoef(actuals, predictions)[0, 1]
    plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Residuals plot
    plt.subplot(1, 3, 3)
    residuals = predictions - actuals
    plt.scatter(actuals[:2000], residuals[:2000], alpha=0.5, s=1)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Actual Ratings')
    plt.ylabel('Residuals (Pred - Actual)')
    plt.title('Residuals Plot')
    
    plt.tight_layout()
    plt.show()
    
    # Final diagnostics
    print(f"\n=== FINAL DIAGNOSTICS ===")
    print(f"Correlation coefficient: {correlation:.4f}")
    print(f"Mean absolute error: {np.mean(np.abs(residuals)):.4f}")
    print(f"Standard deviation of residuals: {np.std(residuals):.4f}")
    
    # Check if model is learning anything useful
    if correlation < 0.1:
        print("\nWARNING: Very low correlation suggests the model isn't learning meaningful patterns.")
        print("Potential issues to check:")
        print("1. Data quality - too sparse or noisy")
        print("2. Hyperparameters - try different learning rates (0.01, 0.001, 0.0001)")
        print("3. Model complexity - try different n_factors (10, 20, 50, 100)")
        print("4. Data preprocessing - check rating normalization")
    elif correlation < 0.3:
        print("\nModel is learning some patterns but performance is still low.")
        print("Consider adjusting hyperparameters or data preprocessing.")
    else:
        print("\nModel appears to be learning successfully!")
    
    return recommender

# Quick hyperparameter tuning function
def quick_hyperparameter_search(file_path, n_trials=5):
    """Run a quick hyperparameter search to find better settings"""
    print("=== QUICK HYPERPARAMETER SEARCH ===")
    
    # Load data once
    df = load_and_preprocess_data(file_path, clean=True)
    
    # Parameter combinations to try
    param_combinations = [
        {'n_factors': 10, 'learning_rate': 0.01, 'reg_lambda': 0.001},
        {'n_factors': 20, 'learning_rate': 0.005, 'reg_lambda': 0.01},
        {'n_factors': 30, 'learning_rate': 0.001, 'reg_lambda': 0.01},
        {'n_factors': 50, 'learning_rate': 0.001, 'reg_lambda': 0.05},
        {'n_factors': 20, 'learning_rate': 0.0001, 'reg_lambda': 0.1},
    ]
    
    best_rmse = float('inf')
    best_params = None
    results = []
    
    for i, params in enumerate(param_combinations[:n_trials]):
        print(f"\nTrial {i+1}/{n_trials}: {params}")
        
        try:
            # Initialize recommender
            recommender = MovieRecommender(
                n_factors=params['n_factors'],
                learning_rate=params['learning_rate'],
                reg_lambda=params['reg_lambda'],
                n_epochs=50,  # Fewer epochs for speed
                batch_size=1024
            )
            
            # Prepare data
            train_df, test_df = recommender.prepare_data(df.copy(), test_size=0.2)
            train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)
            
            # Train
            train_losses, val_losses = recommender.train(train_df, val_df)
            
            # Evaluate
            rmse, predictions, actuals = recommender.evaluate(test_df)
            correlation = np.corrcoef(actuals, predictions)[0, 1]
            
            results.append({
                'params': params,
                'rmse': rmse,
                'correlation': correlation,
                'final_train_loss': train_losses[-1] if train_losses else float('inf'),
                'final_val_loss': val_losses[-1] if val_losses else float('inf')
            })
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_params = params
                
            print(f"  RMSE: {rmse:.4f}, Correlation: {correlation:.4f}")
            
        except Exception as e:
            print(f"  Failed with error: {e}")
            continue
    
    # Print results summary
    print(f"\n=== HYPERPARAMETER SEARCH RESULTS ===")
    print(f"Best RMSE: {best_rmse:.4f}")
    print(f"Best parameters: {best_params}")
    
    # Print all results sorted by RMSE
    results.sort(key=lambda x: x['rmse'])
    print("\nAll results (sorted by RMSE):")
    for result in results:
        print(f"  RMSE: {result['rmse']:.4f}, Corr: {result['correlation']:.4f}, "
              f"Params: {result['params']}")
    
    return best_params, results

# Troubleshooting function
def troubleshoot_training_issues(file_path):
    """Comprehensive troubleshooting for training issues"""
    print("=== COMPREHENSIVE TROUBLESHOOTING ===")
    
    # Load and analyze data
    df = load_and_preprocess_data(file_path, clean=False)  # Don't clean initially
    
    print(f"\n1. RAW DATA ANALYSIS:")
    diagnose_data_issues(df)
    
    print(f"\n2. TESTING DIFFERENT DATA CLEANING THRESHOLDS:")
    for min_ratings in [3, 5, 10, 20]:
        try:
            cleaned_df = clean_data_for_training(df.copy(), min_ratings, min_ratings)
            if len(cleaned_df) > 100:
                sparsity = (1 - len(cleaned_df) / (cleaned_df['user_id'].nunique() * cleaned_df['movie_id'].nunique())) * 100
                print(f"  Min {min_ratings} ratings: {len(cleaned_df)} samples, {sparsity:.1f}% sparse")
            else:
                print(f"  Min {min_ratings} ratings: Too few samples ({len(cleaned_df)})")
        except Exception as e:
            print(f"  Min {min_ratings} ratings: Failed ({e})")
    
    print(f"\n3. TESTING SIMPLE BASELINE:")
    # Test if we can predict the global mean
    cleaned_df = clean_data_for_training(df.copy(), 5, 5)
    if len(cleaned_df) > 100:
        train_df, test_df = train_test_split(cleaned_df, test_size=0.2, random_state=42)
        
        # Baseline: predict global mean
        global_mean = train_df['rating'].mean()
        baseline_predictions = np.full(len(test_df), global_mean)
        baseline_rmse = np.sqrt(mean_squared_error(test_df['rating'], baseline_predictions))
        print(f"  Baseline RMSE (global mean): {baseline_rmse:.4f}")
        
        # User mean baseline
        user_means = train_df.groupby('user_id')['rating'].mean()
        user_baseline_preds = []
        for _, row in test_df.iterrows():
            if row['user_id'] in user_means:
                user_baseline_preds.append(user_means[row['user_id']])
            else:
                user_baseline_preds.append(global_mean)
        user_baseline_rmse = np.sqrt(mean_squared_error(test_df['rating'], user_baseline_preds))
        print(f"  User mean baseline RMSE: {user_baseline_rmse:.4f}")
        
        print(f"  Your matrix factorization should beat these baselines!")

# Example usage with different approaches:
# 
# # Basic usage with improved defaults:
# recommender = train_movie_recommender('ratings.csv', n_factors=20, learning_rate=0.001)
#
# # If still having issues, run comprehensive troubleshooting:
# troubleshoot_training_issues('ratings.csv')
#
# # If you want to find better hyperparameters:
# best_params, results = quick_hyperparameter_search('ratings.csv', n_trials=5)
# recommender = train_movie_recommender('ratings.csv', **best_params)import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import torch

class MatrixFactorization(nn.Module):
    def __init__(self, n_users, n_movies, n_factors=50, bias=True):
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
        # Initialize with small random values - CRITICAL for convergence
        nn.init.normal_(self.user_factors.weight, mean=0.0, std=0.01)  # Smaller std
        nn.init.normal_(self.movie_factors.weight, mean=0.0, std=0.01)  # Smaller std
        
        if self.use_bias:
            nn.init.zeros_(self.user_bias.weight)  # Start biases at zero
            nn.init.zeros_(self.movie_bias.weight)  # Start biases at zero
    
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
    def __init__(self, n_factors=50, learning_rate=0.001, reg_lambda=0.01,  # Lower LR, higher reg
                 n_epochs=100, batch_size=1024, use_bias=True):
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
        self.rating_mean = None  # Store global mean for normalization
        
    def _encode_ids(self, df):
        """Encode user and movie IDs to continuous integers starting from 0"""
        # Create mappings for users
        unique_users = df['user_id'].unique()
        self.user_encoder = {user_id: idx for idx, user_id in enumerate(unique_users)}
        self.user_decoder = {idx: user_id for user_id, idx in self.user_encoder.items()}
        
        # Create mappings for movies
        unique_movies = df['movie_id'].unique()
        self.movie_encoder = {movie_id: idx for idx, movie_id in enumerate(unique_movies)}
        self.movie_decoder = {idx: movie_id for movie_id, idx in self.movie_encoder.items()}
        
        # Apply encoding
        df['user_idx'] = df['user_id'].map(self.user_encoder)
        df['movie_idx'] = df['movie_id'].map(self.movie_encoder)
        
        return df
    
    def prepare_data(self, df, test_size=0.2, random_state=42):
        """Prepare and split the data with normalization"""
        # Store global rating mean for normalization
        self.rating_mean = df['rating'].mean()
        print(f"Global rating mean: {self.rating_mean:.2f}")
        
        # Normalize ratings around 0 (helps with training stability)
        df = df.copy()
        df['rating'] = df['rating'] - self.rating_mean
        print(f"Normalized rating range: {df['rating'].min():.2f} to {df['rating'].max():.2f}")
        
        # Encode IDs
        df = self._encode_ids(df)
        
        # Split data ensuring stratification works with normalized ratings
        # Create rating bins for stratification
        df['rating_bin'] = pd.cut(df['rating'], bins=5, labels=False)
        
        train_df, test_df = train_test_split(df, test_size=test_size, 
                                           random_state=random_state, 
                                           stratify=df['rating_bin'])
        
        # Remove the rating_bin column
        train_df = train_df.drop('rating_bin', axis=1)
        test_df = test_df.drop('rating_bin', axis=1)
        
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
        """Train the matrix factorization model with improved training loop"""
        # Initialize model
        self.model = MatrixFactorization(
            self.n_users, self.n_movies, 
            self.n_factors, self.use_bias
        ).to(self.device)
        
        # Initialize global bias with training mean if using bias
        if self.use_bias:
            with torch.no_grad():
                self.model.global_bias.data.fill_(0.0)  # Start at 0 since we normalized
        
        # Setup optimizer and loss
        optimizer = optim.Adam(self.model.parameters(), 
                              lr=self.learning_rate, 
                              weight_decay=self.reg_lambda)
        
        # Add learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
        
        criterion = nn.MSELoss()
        
        # Create data loaders
        train_loader = self._create_dataloader(train_df, shuffle=True)
        val_loader = self._create_dataloader(val_df, shuffle=False) if val_df is not None else None
        
        # Training loop with early stopping
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        early_stop_patience = 20
        
        print("Starting training with improved diagnostics...")
        
        for epoch in range(self.n_epochs):
            # Training
            self.model.train()
            train_loss = 0
            batch_count = 0
            
            for user_ids, movie_ids, ratings in train_loader:
                user_ids = user_ids.to(self.device)
                movie_ids = movie_ids.to(self.device)
                ratings = ratings.to(self.device)
                
                optimizer.zero_grad()
                predictions = self.model(user_ids, movie_ids)
                loss = criterion(predictions, ratings)
                
                # Check for NaN
                if torch.isnan(loss):
                    print(f"NaN loss detected at epoch {epoch}, batch {batch_count}")
                    print(f"Predictions range: {predictions.min().item():.4f} to {predictions.max().item():.4f}")
                    break
                
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
                batch_count += 1
            
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
                
                # Learning rate scheduling
                scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= early_stop_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            # Enhanced progress reporting
            if (epoch + 1) % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                if val_loader:
                    print(f'Epoch [{epoch+1}/{self.n_epochs}], '
                          f'Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, '
                          f'LR: {current_lr:.6f}')
                else:
                    print(f'Epoch [{epoch+1}/{self.n_epochs}], '
                          f'Train Loss: {train_loss:.6f}, LR: {current_lr:.6f}')
                
                # Print some sample predictions to check training progress
                if epoch == 9:  # After 10 epochs
                    sample_size = min(10, len(train_df))
                    sample_df = train_df.sample(sample_size)
                    with torch.no_grad():
                        sample_preds = self.predict(
                            [self.user_decoder[idx] for idx in sample_df['user_idx'].values],
                            [self.movie_decoder[idx] for idx in sample_df['movie_idx'].values]
                        )
                    actual_denorm = sample_df['rating'].values + self.rating_mean
                    pred_denorm = sample_preds + self.rating_mean
                    print(f"Sample predictions (denormalized):")
                    for i in range(min(5, len(sample_preds))):
                        print(f"  Actual: {actual_denorm[i]:.2f}, Predicted: {pred_denorm[i]:.2f}")
        
        return train_losses, val_losses
    
    def predict(self, user_ids, movie_ids):
        """Make predictions for given user-movie pairs (returns normalized predictions)"""
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
    
    def predict_denormalized(self, user_ids, movie_ids):
        """Make predictions and denormalize to original rating scale"""
        normalized_preds = self.predict(user_ids, movie_ids)
        return normalized_preds + self.rating_mean
    
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
            'movie_id': [self.movie_decoder[idx] for idx in all_movie_indices],
            'predicted_rating': predictions
        })
        
        # Sort by predicted rating
        recommendations = recommendations.sort_values('predicted_rating', ascending=False)
        
        return recommendations.head(n_recommendations)
    
    def evaluate(self, test_df):
        """Evaluate model performance on test set with proper denormalization"""
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
        
        # Denormalize for evaluation
        predictions_denorm = np.array(predictions) + self.rating_mean
        actuals_denorm = np.array(actuals) + self.rating_mean
        
        mse = mean_squared_error(actuals_denorm, predictions_denorm)
        rmse = np.sqrt(mse)
        
        # Additional metrics
        mae = np.mean(np.abs(predictions_denorm - actuals_denorm))
        
        print(f"Test RMSE: {rmse:.4f}")
        print(f"Test MAE: {mae:.4f}")
        print(f"Prediction range: {predictions_denorm.min():.2f} to {predictions_denorm.max():.2f}")
        print(f"Actual range: {actuals_denorm.min():.2f} to {actuals_denorm.max():.2f}")
        
        return rmse, predictions_denorm, actuals_denorm

# Diagnostic functions
def diagnose_data_issues(df):
    """Diagnose common data issues that prevent convergence"""
    print("=== DATA DIAGNOSIS ===")
    
    # Basic stats
    print(f"Dataset shape: {df.shape}")
    print(f"Users: {df['user_id'].nunique()}, Movies: {df['movie_id'].nunique()}")
    print(f"Rating range: {df['rating'].min()} - {df['rating'].max()}")
    print(f"Rating distribution:\n{df['rating'].value_counts().sort_index()}")
    
    # Sparsity analysis
    total_possible = df['user_id'].nunique() * df['movie_id'].nunique()
    sparsity = (1 - len(df) / total_possible) * 100
    print(f"Matrix sparsity: {sparsity:.2f}%")
    
    # User/movie activity distribution
    user_counts = df['user_id'].value_counts()
    movie_counts = df['movie_id'].value_counts()
    
    print(f"\nUser activity stats:")
    print(f"  Min ratings per user: {user_counts.min()}")
    print(f"  Max ratings per user: {user_counts.max()}")
    print(f"  Median ratings per user: {user_counts.median()}")
    print(f"  Users with <5 ratings: {(user_counts < 5).sum()}")
    
    print(f"\nMovie popularity stats:")
    print(f"  Min ratings per movie: {movie_counts.min()}")
    print(f"  Max ratings per movie: {movie_counts.max()}")
    print(f"  Median ratings per movie: {movie_counts.median()}")
    print(f"  Movies with <5 ratings: {(movie_counts < 5).sum()}")
    
    # Check for extreme outliers
    if user_counts.min() == 1:
        single_rating_users = (user_counts == 1).sum()
        print(f"WARNING: {single_rating_users} users have only 1 rating")
    
    if movie_counts.min() == 1:
        single_rating_movies = (movie_counts == 1).sum()
        print(f"WARNING: {single_rating_movies} movies have only 1 rating")
    
    return user_counts, movie_counts

def clean_data_for_training(df, min_user_ratings=5, min_movie_ratings=5):
    """Clean data by removing users/movies with too few ratings"""
    print(f"\n=== CLEANING DATA ===")
    print(f"Original data: {len(df)} ratings, {df['user_id'].nunique()} users, {df['movie_id'].nunique()} movies")
    
    # Iteratively remove users and movies with few ratings
    prev_len = 0
    iteration = 0
    while len(df) != prev_len and iteration < 10:  # Max 10 iterations to prevent infinite loop
        prev_len = len(df)
        iteration += 1
        
        # Remove users with too few ratings
        user_counts = df['user_id'].value_counts()
        valid_users = user_counts[user_counts >= min_user_ratings].index
        df = df[df['user_id'].isin(valid_users)]
        
        # Remove movies with too few ratings
        movie_counts = df['movie_id'].value_counts()
        valid_movies = movie_counts[movie_counts >= min_movie_ratings].index
        df = df[df['movie_id'].isin(valid_movies)]
        
        print(f"Iteration {iteration}: {len(df)} ratings, {df['user_id'].nunique()} users, {df['movie_id'].nunique()} movies")
    
    return df

# Example usage and data loading
def load_and_preprocess_data(df: pd.DataFrame, clean=True, min_user_ratings=5, min_movie_ratings=5):
    """
    Load movie ratings data from file with comprehensive preprocessing
    Expected format: user_id, movie_id, rating
    """
    # Handle common IMDB dataset formats
    if 'userId' in df.columns:
        df = df.rename(columns={'userId': 'user_id', 'movieId': 'movie_id'})
    elif 'UserID' in df.columns:
        df = df.rename(columns={'UserID': 'user_id', 'MovieID': 'movie_id', 'Rating': 'rating'})
    
    # Ensure correct column names
    expected_columns = ['user_id', 'movie_id', 'rating']
    if not all(col in df.columns for col in expected_columns):
        print("Expected columns: user_id, movie_id, rating")
        print(f"Found columns: {df.columns.tolist()}")
        raise ValueError("Please ensure your data has columns: user_id, movie_id, rating")
        
    # Basic data cleaning
    df = df.dropna()
    
    # Convert to appropriate types
    df['user_id'] = df['user_id'].astype(int)
    df['movie_id'] = df['movie_id'].astype(int)
    df['rating'] = df['rating'].astype(float)
    
    # Diagnose issues
    user_counts, movie_counts = diagnose_data_issues(df)
    
    # Clean data if requested
    if clean:
        df = clean_data_for_training(df, min_user_ratings, min_movie_ratings)
        print(f"\nFinal cleaned data: {len(df)} ratings")
        
        # Re-diagnose after cleaning
        print("\n=== AFTER CLEANING ===")
        diagnose_data_issues(df)
    
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
# Load data

import os
os.chdir('/home/yakov/Studies/gollnick-PyTorchUltimateMaterial/190_RecommenderSystems')

df = pd.read_csv('ratings.csv')
df.rename(columns={'movieId': 'movie_id', 'userId': 'user_id'}, inplace=True)
recommender = train_movie_recommender(df)
recommendations = recommender.recommend_movies(user_id=1, n_recommendations=10)
