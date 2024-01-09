#_______________________________GradientDescentApproach________________________#

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class DataPreprocessor:
    @staticmethod
    def load_csv(file_path):
        return pd.read_csv(file_path)

class UserItemDataset(Dataset):
	def __init__(self, dataset):
		self.user_ids = torch.tensor(dataset['user-id'].values, dtype=torch.int64)
		self.movie_ids = torch.tensor(dataset['movie-id'].values, dtype=torch.int64)
		self.scores = torch.tensor(dataset['recommendation-score'].values, dtype=torch.float32)

	def __len__(self):
		return len(self.scores)

	def __getitem__(self, index):
		return self.user_ids[index], self.movie_ids[index], self.scores[index]

	class RecommenderNet(nn.Module):
		def __init__(self, total_users, total_movies, factors=10):
			super(RecommenderNet, self).__init__()
			self.user_embeddings = nn.Embedding(total_users, factors, sparse=True)
			self.movie_embeddings = nn.Embedding(total_movies, factors, sparse=True)
		
		def forward(self, user, movie):
			user_embed = self.user_embeddings(user)
			movie_embed = self.movie_embeddings(movie)
			return (user_embed * movie_embed).sum(1)

	class ModelTrainer:
		def __init__(self, model, data_loader):
			self.model = model
			self.data_loader = data_loader
			self.criterion = nn.L1Loss()
			self.optimizer = optim.SparseAdam(model.parameters(), lr=0.001)

	def train(self, epochs):
		training_losses = []
		for epoch in range(epochs):
			self.model.train()
			total_loss = 0.0
			for user, movie, score in self.data_loader:
				self.optimizer.zero_grad()
				predicted_score = self.model(user, movie)
				loss = self.criterion(predicted_score, score)
				loss.backward()
				self.optimizer.step()
				total_loss += loss.item()
			total_loss /= len(self.data_loader)
			training_losses.append(total_loss)
			print(f'Epoch {epoch + 1}/{epochs}, Average Loss: {total_loss:.4f}')
		return training_losses

class ModelEvaluator:
	def __init__(self, model, data_loader):
		self.model = model
		self.data_loader = data_loader
	def predict(self):
		self.model.eval()
		predictions = []
		with torch.no_grad():
			for user, movie, _ in self.data_loader: # Note the addition of '_'
				scores_predict = self.model(user, movie)
				predictions.extend(scores_predict.numpy())
		return predictions

class Plotter:
	@staticmethod
	def plot_loss(loss_values):
		plt.figure(figsize=(10, 6))
		plt.plot(range(1, len(loss_values) + 1), loss_values, label='Training Loss')
		plt.xlabel('Epoch')
		plt.ylabel('Average Loss')
		plt.title('Training Loss Across Epochs')
		plt.legend()
		plt.grid(True)
		plt.savefig('GD__PLOTS.png')
		plt.show()

def main():
	training_data = DataPreprocessor.load_csv('train.csv')
	testing_data = DataPreprocessor.load_csv('test.csv')
	training_dataset = UserItemDataset(training_data)
	training_loader = DataLoader(training_dataset, batch_size=64, shuffle=True)
	unique_users = training_data['user-id'].nunique()
	unique_movies = training_data['movie-id'].nunique()
	recommender_model = RecommenderNet(unique_users, unique_movies)
	trainer = ModelTrainer(recommender_model, training_loader)
	loss_values = trainer.train(100)
	Plotter.plot_loss(loss_values)
	testing_dataset = UserItemDataset(testing_data)
	testing_loader = DataLoader(testing_dataset, batch_size=64, shuffle=False)
	evaluator = ModelEvaluator(recommender_model, testing_loader)
	test_predictions = evaluator.predict()
	testing_data['predicted_rating'] = np.clip(test_predictions, 0, 5)
	testing_data.to_csv('predicted_gd_ratings.csv', index=False)
	print(testing_data)

if __name__ == '__main__':
	main()

#_______________________________ALS____________________________________________#
class DataProcessor:
	def __init__(self, path_training, path_testing):
		self.dataset_train = pd.read_csv(path_training)
		self.dataset_test = pd.read_csv(path_testing)
		self.total_users = self.dataset_train['user-id'].nunique()
		self.total_movies = self.dataset_train['movie-id'].nunique()

	def generate_user_movie_matrix(self):
		interaction_matrix = np.zeros((self.total_users, self.total_movies))
		for record in self.dataset_train.itertuples():
			user_index = record[1] - 1
			movie_index = record[2] - 1
			interaction_matrix[user_index, movie_index] = record[3]
		return interaction_matrix

class RecommenderSystem:
	def __init__(self, user_count, movie_count, factor_count=10):
		self.factors_users = np.random.rand(user_count, factor_count)
		self.factors_movies = np.random.rand(movie_count, factor_count)
		self.factor_count = factor_count

	def optimize_factors(self, user_movie_matrix, epochs=10):
		loss_tracking = []
		for epoch in range(epochs):
			self._update_user_factors(user_movie_matrix)
			self._update_movie_factors(user_movie_matrix)
			mae_loss = self._calculate_mae(user_movie_matrix)
			loss_tracking.append(mae_loss)
			print(f"Epoch {epoch + 1}/{epochs}, MAE Loss: {mae_loss:.4f}")
		return loss_tracking

	def _update_user_factors(self, user_movie_matrix):
		for i in range(user_movie_matrix.shape[0]):
			movie_indices = np.where(user_movie_matrix[i, :] > 0)[0]
			if len(movie_indices) == 0:
				continue
			A = self.factors_movies[movie_indices, :]
			b = user_movie_matrix[i, movie_indices]
			self.factors_users[i, :] = np.linalg.solve(A.T @ A + np.eye(self.factor_count), A.T @ b)

	def _update_movie_factors(self, user_movie_matrix):
		for j in range(user_movie_matrix.shape[1]):
			user_indices = np.where(user_movie_matrix[:, j] > 0)[0]
			if len(user_indices) == 0:
				continue
			A = self.factors_users[user_indices, :]
			b = user_movie_matrix[user_indices, j]
			self.factors_movies[j, :] = np.linalg.solve(A.T @ A + np.eye(self.factor_count), A.T @ b)

	def _calculate_mae(self, user_movie_matrix):
		predicted_ratings = self.factors_users @ self.factors_movies.T
		error = user_movie_matrix - predicted_ratings
		return np.abs(error[user_movie_matrix > 0]).mean()

class PredictionEvaluator:
	@staticmethod
	def make_predictions(dataset_test, factors_users, factors_movies):
		prediction_list = []
		for record in dataset_test.itertuples():
			index_user = record[1] - 1
			index_movie = record[2] - 1
			prediction_rating = np.dot(factors_users[index_user, :], factors_movies[index_movie, :])
			prediction_rating = np.clip(prediction_rating, 0, 5)
			prediction_list.append(prediction_rating)
		dataset_test['predicted_rating'] = prediction_list
		return dataset_test

class LossPlotter:
	@staticmethod
	def show_loss_graph(loss_values, file_path='loss_graph.png'):
		plt.plot(range(len(loss_values)), loss_values)
		plt.xlabel('Epoch')
		plt.ylabel('MAE Loss')
		plt.title('MAE Loss Progression')
		plt.savefig('ALS__PLOTS.png')
		plt.show()

# Implementing the classes
data_processor = DataProcessor('train.csv', 'test.csv')
user_movie_matrix = data_processor.generate_user_movie_matrix()

recommender = RecommenderSystem(data_processor.total_users,
data_processor.total_movies)
mae_loss_values = recommender.optimize_factors(user_movie_matrix)

test_data_predictions = PredictionEvaluator.make_predictions(data_processor.dataset_test, recommender.factors_users, recommender.factors_movies)
test_data_predictions.to_csv('predictions_als_rating.csv', index=False)

LossPlotter.show_loss_graph(mae_loss_values)
print(test_data_predictions)