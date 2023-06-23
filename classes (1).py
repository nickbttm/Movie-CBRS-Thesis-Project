
from PIL import Image
from wordcloud import WordCloud
import pandas as pd
import numpy as np
# import umap
import matplotlib.pyplot as plt
import pickle
import time

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import os

import wikipediaapi
from nltk.corpus import wordnet as wn

import requests
import spacy

import xgboost as xgb

import mwclient
import random

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn.decomposition import PCA
from sklearn.preprocessing import OrdinalEncoder

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from IPython.display import display

from joblib import Parallel, delayed
from libsvm.svmutil import svm_train, svm_predict

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

def timer_decorator(func):
  def wrapper(*args, **kwargs):
      start_time = time.time()
      result = func(*args, **kwargs)
      end_time = time.time()
      print(f"{func.__name__} took {end_time - start_time:.2f} seconds to execute.")
      return result
  return wrapper

class Master:
    def __init__(self, explanation=str, vector_size=100, epochs=50):

        self.data = Data()
        self.doc2vec = CustomDoc2Vec(vector_size=vector_size, epochs=epochs)
        self.xgb = CustomXGBoost(num_class=2)
        self.side_models = None
        self.user_vector_generator = None
        self.results = {}
        self.confusion_results = {}
        self.dataset_name = str()
        self.explanation = explanation
        self.results_eda = None
        self.tag_augmentor = None
    
    def save_instance(self):
        filename = self.explanation + "_" + self.dataset_name + ".pkl"
        with open(filename, "wb") as file:
            pickle.dump(self, file)

    def get_results(self):
      methods = ['knn', 'rf', 'dummy']
      
      for method in methods:
          self.results[f'{self.dataset_name}_{method}_train'], self.confusion_results[
            f'{self.dataset_name}_{method}_train'] = getattr(
              self.side_models, f'run_{method}')(train=True)
          self.results[f'{self.dataset_name}_{method}_test'], self.confusion_results[
            f'{self.dataset_name}_{method}_test'] = getattr(
              self.side_models, f'run_{method}')()

    @timer_decorator
    def load_and_preprocess_data(self):
        
        self.data.load_data(movies_path="movies.csv", 
          ratings_path="ratings.csv", tags_path="tags.csv", 
          genome_scores_path="genome-scores.csv", 
          genome_tags_path="genome-tags.csv")
        if self.dataset_name == 'comment_tag':
          self.data.denorme_comment_tag()
        elif self.dataset_name == 'genome_tag':
          self.data.denorme_genome()
        elif self.dataset_name == 'wordnet':
          self.data.denorme_comment_tag()
          self.tag_augmentor = TagAugmentor(self.data.denormed)
          self.tag_augmentor.augment_tags(method='wordnet', limit=3)
          self.data.denormed['tag'] = self.tag_augmentor.df.augmented_tags.astype('str')
          self.data.denormed.drop('augmented_tags', axis=1, inplace=True)
          # self.data.tokenize()
        elif self.dataset_name == 'wikidata':
          self.data.denorme_comment_tag()
          self.tag_augmentor = TagAugmentor(self.data.denormed)
          self.tag_augmentor.augment_tags(method='wikidata', limit=3)
          self.data.denormed['tag'] = self.tag_augmentor.df.augmented_tags.astype('str')
          self.data.denormed.drop('augmented_tags', axis=1, inplace=True)
        # Need to specify denorme_genome function too.
        self.data.tokenize()

    @timer_decorator
    def create_and_train_doc2vec(self):
        self.doc2vec.create_model(self.data.tokenized)
        self.doc2vec.train()
        self.doc2vec.get_vectors()

    def create_train_test_df(self, rating_low_end=100, rating_high_end=100):
        self.data.create_train_test_df(self.doc2vec.mv_tags_vectors, 
          rating_low_end=rating_low_end, rating_high_end=rating_high_end)

    def generate_user_vectors(self):
        self.user_vector_generator = UserVectorGenerator(
          self.data.train_test_df)
    
    def split_data(self):
        self.user_vector_generator.generate()
        self.user_vector_generator.generate_X_y_split()

    def train_and_evaluate_xgboost(self):
        self.xgb.fit(self.user_vector_generator.train_X, self.user_vector_generator.train_y)
        self.results[f'{self.dataset_name}_xgb_test'], self.confusion_results[
          f'{self.dataset_name}_xgb_test'] = self.xgb.evaluate(
            self.user_vector_generator.test_X, self.user_vector_generator.test_y)
        self.results[f'{self.dataset_name}_xgb_train'], self.confusion_results[
          f'{self.dataset_name}_xgb_train'] = self.xgb.evaluate(
            self.user_vector_generator.train_X, self.user_vector_generator.train_y)
    
    @timer_decorator
    def run_side_models(self):
        self.side_models = SideModels(
          self.user_vector_generator.train_X, self.user_vector_generator.test_X,
          self.user_vector_generator.train_y, self.user_vector_generator.test_y
          )
        self.get_results()

    def run_results_eda(self):
      results_eda = Results_EDA(results=self.results, 
                                confusion_results=self.confusion_results)
      
    def run(self, dataset_name=str):
      try:
        self.dataset_name = dataset_name
        self.load_and_preprocess_data()
        self.create_and_train_doc2vec()
        self.create_train_test_df()
        self.generate_user_vectors()
        self.split_data()
        # self.train_and_evaluate_xgboost()
        # self.run_side_models()
        # self.save_instance()
      except Exception as e:
        print(f"Error: {e}")
      finally:
        self.save_instance()
      # self.run_results_eda()

class Data:
  def __init__(self):
    self.mv_genres = None
    self.mv_ratings = None
    self.tags_df = None
    self.mv_tags = None
    self.mv_tags_desc = None
    self.denormed = None
    self.tokenized = None
    self.train_test_df = None

  def load_data(
    self, movies_path, ratings_path, genome_scores_path, 
    tags_path=False, genome_tags_path=False
    ):

    self.mv_genres = pd.read_csv(movies_path)
    self.mv_ratings = pd.read_csv(ratings_path)
    self.mv_tags = pd.read_csv(genome_scores_path)
    if tags_path:
      self.tags_df = pd.read_csv(tags_path)
    if genome_tags_path:
      self.mv_tags_desc = pd.read_csv(genome_tags_path)

  def denorme_genome(self):
    # join dataframes to get tag description and movie title name all in one table
    mv_tags_denorm = self.mv_tags.merge(self.mv_tags_desc, on = 'tagId').merge(
      self.mv_genres, on = 'movieId')

    # for each movie, compute the relevance rank of tags so we can eventually rank order tags for each movie
    mv_tags_denorm['relevance_rank'] = mv_tags_denorm.groupby("movieId")["relevance"].rank(
      method = "first", ascending = False).astype('int64')

    # flatten tags table to get a list of top 100 tags for each movie
    mv_tags_list = mv_tags_denorm[mv_tags_denorm.relevance_rank <= 50].groupby(
      ['movieId','title'])['tag'].apply(lambda x: ','.join(x)).reset_index()

    mv_tags_list['tag_list'] = mv_tags_list.tag.map(lambda x: x.split(','))
    
    self.denormed = mv_tags_list
  
  def denorme_comment_tag(self):
    unique_values = self.mv_tags['movieId'].unique()
    comment_tags = self.tags_df[self.tags_df['movieId'].isin(unique_values)]
    float_index_list = list()
    for i,j in comment_tags.iterrows():
      if type(j.tag) != str:
        # print(type(i))
        float_index_list.append(i)
    float_index_list = [i for i, j in comment_tags.iterrows() if not isinstance(j.tag, str)]
    comment_tags.drop(float_index_list, inplace=True)
    comment_tags_df = comment_tags.groupby(['movieId'])['tag'].apply(lambda x: ','.join(x)).reset_index()
    comment_tags_df['tag_list'] = comment_tags_df.tag.map(lambda x: x.split(','))
    self.denormed = comment_tags_df

  def tokenize(self):
    stop_words = stopwords.words('english')
    # tokenize document and clean
    def word_tokenize_clean(doc):
      
      # split into lower case word tokens
      tokens = word_tokenize(doc.lower())
      
      # remove tokens that are not alphabetic (including punctuation) and not a stop word
      tokens = [word for word in tokens if word.isalpha() and not word in stop_words]
      
      return tokens
    # corpus of movie tags
    mv_tags_corpus = self.denormed.tag.values 
    # preprocess corpus of movie tags before feeding it into Doc2Vec model 
    mv_tags_doc = [TaggedDocument(words=word_tokenize_clean(D), tags=[str(i)]) for i, D in enumerate(mv_tags_corpus)]
    self.tokenized = mv_tags_doc
  
  def create_train_test_df(self, mv_tags_vectors, rating_low_end=100, rating_high_end=32202):
    # Creating vector_df that has Doc2Vec vectors with movie ids of 13816 movies
    vector_df = self.denormed.join(pd.DataFrame(mv_tags_vectors))
    # print(vector_df.columns)
    columns_to_drop = ['title', 'tag', 'tag_list']
    vector_df.drop(axis=1, columns=[col for col in columns_to_drop if col in vector_df.columns], inplace=True)
  
    # ---taking only the movies that has ratings---
    mv_tag_ratings = self.mv_ratings[self.mv_ratings['movieId'].isin(self.mv_tags.movieId.unique())]

    # Only the users that have 100 ratings
    val_counts = pd.DataFrame(mv_tag_ratings['userId'].value_counts())
    val_counts = val_counts[((val_counts['userId'] >= rating_low_end) & (val_counts['userId'] <= rating_high_end ))].reset_index()
    # val_counts.drop('userId', axis=1, inplace=True)
    val_counts.columns = ['userId', 'count']
    val_counts.rename(columns={'index' : 'userId'}, inplace=True)
    # merging 
    mv_tag_ratings = mv_tag_ratings.merge(val_counts, on='userId')
    #  new_ratings column to transform the ratings to binary
    mv_tag_ratings['new_rating'] = [1 if i[1]['rating'] >= np.median(mv_tag_ratings.rating) else 0 for i in mv_tag_ratings.iterrows()]

    # dataframe consisting ratings and vectors on selected user 
    # --- for ML Stage

    # unique_users = mv_tag_ratings['userId'].unique() # Get unique user ids
    # selected_users = unique_users[:100] # Select first 100 unique user ids
    # mv_tag_ratings = mv_tag_ratings[mv_tag_ratings['userId'].isin(selected_users)] # Filter the dataframe to only include selected users
    
    self.train_test_df = mv_tag_ratings.merge(vector_df, on='movieId')

class CustomDoc2Vec:
  def __init__(self, vector_size=100, window=6, min_count=2, epochs=20, dm=0, alpha=0.025, min_alpha=0.00025):
      self.vector_size = vector_size
      self.window = window
      self.min_count = min_count
      self.epochs = epochs
      self.dm = dm
      self.alpha = alpha
      self.min_alpha = min_alpha
      self.model = None
      self.mv_tags_vectors = None 
      self.tokenized = None
  
  def create_model(self, tokenized):
    self.tokenized = tokenized
    self.model = Doc2Vec(vector_size=self.vector_size,
                alpha=self.alpha, 
                min_alpha=self.min_alpha,
                min_count=self.min_count,
                window = self.window,
                dm=self.dm) # paragraph vector distributed bag-of-words (PV-DBOW)
    self.model.build_vocab(tokenized)

  def train(self):
    # train Doc2Vec model
    # stochastic (random initialization), so each run will be different unless you specify seed

    print('Epoch', end = ': ')
    for epoch in range(self.epochs):
      print(epoch, end = ' ')
      self.model.train(self.tokenized,
                  total_examples=self.model.corpus_count,
                  epochs=self.model.epochs)
      print(self.model.alpha)
      # decrease the learning rate
      self.model.alpha -= 0.0002
      # fix the learning rate, no decay
      # model.min_alpha = model.alpha
  def get_vectors(self):
    self.mv_tags_vectors = self.model.dv.vectors

  def save_model(self, file_path):
    self.model.save(file_path)
  
  def load_model(self, file_path):
    self.model = Doc2Vec.load(file_path)
   
class UserVectorGenerator:
    def __init__(self, data, test_size=0.2, random_state=42):
        self.data = data
        self.test_size = test_size
        self.random_state = random_state
        self.train_data = None
        self.test_data = None
        self.user_vectors = None
        self.train_X = None
        self.test_X = None
        self.train_y = None
        self.test_y = None

    def split_data(self):
        self.train_data, self.test_data = train_test_split(self.data, test_size=self.test_size, random_state=self.random_state)

    def compute_user_vectors(self):
      if self.train_data is None:
          raise ValueError("Train data not initialized. Run split_data() first.")
    
      def weighted_average(x):
          weighted_vectors = x[[i for i in range(50)]] * x['rating'].values[:, None]
          sum_of_vectors = weighted_vectors.sum(axis=0)
          sum_of_ratings = x['rating'].sum()
          return sum_of_vectors / sum_of_ratings

      self.user_vectors = self.train_data.groupby('userId').apply(weighted_average)
      self.user_vectors.reset_index(inplace=True)

    def reduce_user_vectors(self, n_components=20):
        if self.user_vectors is None:
            raise ValueError("User vectors not initialized. Run compute_user_vectors() first.")
        
        pca = PCA(n_components=n_components)
        reduced_user_vectors = pca.fit_transform(self.user_vectors.iloc[:, 1:])
        
        reduced_user_vectors_df = pd.DataFrame(reduced_user_vectors, columns=[f'reduced_user_vec_{i}' for i in range(n_components)])
        reduced_user_vectors_df['userId'] = self.user_vectors['userId']
        self.user_vectors = reduced_user_vectors_df
    
    def merge_user_vectors(self):
        if self.train_data is None or self.test_data is None or self.user_vectors is None:
            raise ValueError("Test data or user vectors not initialized. Run split_data() and compute_user_vectors() first.")
        
        self.train_data = self.train_data.merge(self.user_vectors, on='userId', suffixes=('', '_user'))
        self.test_data = self.test_data.merge(self.user_vectors, on='userId', suffixes=('', '_user'))

    def generate(self):
        self.split_data()
        self.compute_user_vectors()
        # self.reduce_user_vectors()
        self.merge_user_vectors()
        return self.train_data, self.test_data

    
    def ordinal_encode(self):
      # Reshape train_y to be a 2D array
      train_y_2d = np.array(self.train_data.new_rating).reshape(-1, 1)

      # Initialize the OrdinalEncoder
      ordinal_encoder = OrdinalEncoder()

      # Fit and transform train_y_2d
      train_y_encoded = ordinal_encoder.fit_transform(train_y_2d)

      # Reshape train_y_encoded back to a 1D array
      self.train_y = train_y_encoded.ravel()

      # Reshape test_y to be a 2D array
      test_y_2d = np.array(self.test_data.new_rating).reshape(-1, 1)

      # Transform test_y_2d using the ordinal_encoder
      test_y_encoded = ordinal_encoder.transform(test_y_2d)

      # Reshape test_y_encoded back to a 1D array
      self.test_y = test_y_encoded.ravel()

    def generate_X_y_split(self):
      self.train_X = self.train_data.iloc[:,6:]
      self.test_X = self.test_data.iloc[:,6:]
      self.ordinal_encode()

      self.train_X.columns = self.train_X.columns.astype(str)
      self.test_X.columns = self.test_X.columns.astype(str)
      return self.train_X, self.test_X, self.train_y, self.test_y
      
class CustomXGBoost:
    def __init__(self, eta=0.3, max_depth=5, objective='multi:softprob',
      num_class=2, steps=20):
        self.eta = eta
        self.max_depth = max_depth
        self.objective = objective
        self.num_class = num_class
        self.steps = steps
  
        self.model = None
        self.grid = None

    param_grid = {
          'eta': [0.01, 0.3],
          'max_depth': [3, 5, 7],
          # 'gamma': [0, 0.1, 0.2],
          # 'subsample': [0.5, 0.8, 1],
          # 'colsample_bytree': [0.5, 0.8, 1],
          'lambda': [0.1, 1, 10]}

    def results(self,y_test, y_pred):
      results = dict()
      results['accuracy'] = accuracy_score(y_test, y_pred)
      results['precision'] = precision_score(y_test, y_pred, average='macro')
      results['recall'] = recall_score(y_test, y_pred, average='macro')
      # results['rmse'] = np.sqrt(mean_squared_error(y_test, y_pred))
      # results['mae'] = mean_absolute_error(y_test, y_pred)
      results['f1_macro'] = f1_score(y_test, y_pred, average='macro')
      results['f1_weighted'] = f1_score(y_test, y_pred, average='weighted')
      print(f"  Accuracy: {results['accuracy']}")
      print(f"  Precision: {results['precision']}")
      print(f"  Recall: {results['recall']}")
      print(f"  F1 Score (Macro): {results['f1_macro']}")
      print(f"  F1 Score (Weighted): {results['f1_weighted']}")
      # print(f"  RMSE: {results['rmse']}")
      # print(f"  MAE: {results['mae']}")
      return results

    # def show_confusion(self,y_test, y_pred):
    #   # Compute confusion matrix
    #   cm = confusion_matrix(y_test, y_pred)

    #   # Display confusion matrix
    #   disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    #   disp.plot()

    #   # Extract confusion matrix values
    #   tn, fp, fn, tp = cm.ravel()
      
    #   # Return confusion matrix values as a dictionary
    #   confusion_results = {
    #       "True Negative": tn,
    #       "False Positive": fp,
    #       "False Negative": fn,
    #       "True Positive": tp,
    #   }
    #   return confusion_results

    def show_confusion(self,y_test, y_pred):
      # Compute confusion matrix
      cm = confusion_matrix(y_test, y_pred)

      # Normalize the confusion matrix by row (i.e by the number of samples
      # in each class)
      cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

      # Display confusion matrix
      disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized)
      disp.plot()

      # Calculate confusion matrix percentages
      tn, fp, fn, tp = cm_normalized.ravel() * 100

      # Return confusion matrix values as a dictionary
      confusion_results = {
          "True Negative": f"{tn:.2f}%",
          "False Positive": f"{fp:.2f}%",
          "False Negative": f"{fn:.2f}%",
          "True Positive": f"{tp:.2f}%",
      }
      return confusion_results


    def fit(self, X, y):
        D_train = xgb.DMatrix(X, label=y)
        
        param = {
            'eta': self.eta,
            'max_depth': self.max_depth,
            'objective': self.objective,
            'num_class': self.num_class
        }
        
        self.model = xgb.train(param, D_train, self.steps)
        return self

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model is not trained yet. Call fit() first.")
        
        D_test = xgb.DMatrix(X)
        preds = self.model.predict(D_test)
        best_preds = np.asarray([np.argmax(line) for line in preds])
        # best_preds = (preds > 0.5).astype(int)
        return best_preds

    def evaluate(self, X_test, Y_test):
      D_test = xgb.DMatrix(X_test, label=Y_test)
      preds = self.model.predict(D_test)
      best_preds = np.asarray([np.argmax(line) for line in preds])

      return self.results(Y_test, best_preds), self.show_confusion(Y_test, best_preds)
    
    def cross_validate(self, X, y, nfold=5):
      dtrain = xgb.DMatrix(X, label=y)
      params = {
          'eta': self.eta,
          'max_depth': self.max_depth,
          'objective': self.objective,
          'num_class': self.num_class
      }
      cv_results = xgb.cv(
          params,
          dtrain,
          num_boost_round=self.steps,
          nfold=nfold,
          metrics=('f1_weighted'),
          early_stopping_rounds=10,
      )
      return cv_results

    def grid_search(self, X, y, param_grid=None):
      y = y.astype(int)
      if param_grid is None:
        param_grid = self.param_grid
      self.grid = GridSearchCV(xgb.XGBClassifier(
          eta=self.eta,
          max_depth=self.max_depth,
          objective='binary:logistic',
          # num_class=self.num_class,
          # n_estimators=self.steps
      ), param_grid, scoring='accuracy', error_score='raise')
      self.grid.fit(X, y)
      return self.grid.best_params_, self.grid.best_score_

class SideModels:
  def __init__(self,X_train,X_test,y_train,y_test):
    self.X_train = X_train
    self.X_test = X_test
    self.y_train = y_train
    self.y_test = y_test
    self.model = None

  def results(self, name, y_pred, train=False):
    if train:
      test = self.y_train
    else:
      test = self.y_test
    results = dict()
    print(name)
    results['accuracy'] = accuracy_score(test, y_pred)
    results['precision'] = precision_score(test, y_pred, average='macro')
    results['recall'] = recall_score(test, y_pred, average='macro')
    results['f1_macro'] = f1_score(test, y_pred, average='macro')
    results['f1_weighted'] = f1_score(test, y_pred, average='weighted')
    # results['rmse'] = np.sqrt(mean_squared_error(test, y_pred))
    # results['mae'] = mean_absolute_error(test, y_pred)
    print(f"  Accuracy: {results['accuracy']}")
    print(f"  Precision: {results['precision']}")
    print(f"  Recall: {results['recall']}")
    print(f"  F1 Score (Macro): {results['f1_macro']}")
    print(f"  F1 Score (Weighted): {results['f1_weighted']}")
    # print(f"  RMSE: {results['rmse']}")
    # print(f"  MAE: {results['mae']}")
    return results

  def show_confusion(self, y_pred, train=False):
    if train:
      cm = confusion_matrix(self.y_train, y_pred)
    else:
      # Compute confusion matrix
      cm = confusion_matrix(self.y_test, y_pred)

    # Display confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    # Extract confusion matrix values
    tn, fp, fn, tp = cm.ravel()
    
    # Return confusion matrix values as a dictionary
    confusion_results = {
        "True Negative": tn,
        "False Positive": fp,
        "False Negative": fn,
        "True Positive": tp,
        }
    return confusion_results

  def run_rf(self, name='run_rf', train=False):
    # Initialize the RandomForestClassifier
    rf_clf = RandomForestClassifier(random_state=42)

    # Train the model
    rf_clf.fit(self.X_train, self.y_train)
    if train:
      y_pred = rf_clf.predict(self.X_train)
      return self.results(y_pred, train=True), self.show_confusion(y_pred, train=True)
    else:
      # Make predictions
      y_pred = rf_clf.predict(self.X_test)
      return self.results(y_pred), self.show_confusion(y_pred)

  def run_knn(self, name='run_knn', train=False):
    # Initialize the KNeighborsClassifier
    knn_clf = KNeighborsClassifier()

    # Train the model
    knn_clf.fit(self.X_train, self.y_train)

    if train:
      y_pred = knn_clf.predict(self.X_train)
      return self.results(y_pred, train=True), self.show_confusion(y_pred, train=True)
    else:
      # Make predictions
      y_pred = knn_clf.predict(self.X_test)
      return self.results(y_pred), self.show_confusion(y_pred)

  @timer_decorator
  def run_svm(self, name='run_svm', train=False, n_jobs=None):

    if self.model is not None:
      svm_clf = self.model
    else:
      if n_jobs is not None:
        if isinstance(n_jobs, int) and n_jobs > 0:
          svm_clf = Parallel(n_jobs=n_jobs)(
            delayed(svm_train)(self.y_train.astype(int)[i], 
            self.X_train.to_numpy()[i], 
            '-q') for i in range(len(self.y_train)))
          # svm_clf = svm_train(self.y_train.tolist(), self.X_train.values.tolist(), '-q')
          self.model = svm_clf
        else:
          # Initialize the Support Vector Machines (SVM)
          print("Invalid value for n_jobs. Using default value of None.")
          return
      else:

        svm_clf = svm_train(
          self.y_train.astype(int), 
          self.X_train.to_numpy(), '-q')
        self.model = svm_clf

    if train:

      y_pred = svm_predict(
        self.y_train.astype(int), 
        self.X_train.to_numpy(), svm_clf)[0]

      return self.results(y_pred, train=True), self.show_confusion(y_pred, train=True)
    else:
      # Make predictions
      y_pred = svm_predict(
        self.y_test.astype(int), 
        self.X_test.to_numpy(), svm_clf)[0]
      return self.results(y_pred), self.show_confusion(y_pred)

  def run_dummy(self, name='run_dummy', train=False):
    # Create the dummy classifier
    dummy_clf = DummyClassifier(strategy="most_frequent")

    # Fit the model
    dummy_clf.fit(self.X_train, self.y_train)

    if train:
      y_pred = dummy_clf.predict(self.X_train)
      return self.results(y_pred, train=True), self.show_confusion(y_pred, train=True)
    else:
      # Make predictions
      y_pred = dummy_clf.predict(self.X_test)
      return self.results(y_pred), self.show_confusion(y_pred)

class TagAugmentor:

  def __init__(self, df):
      self.df = df
      self.nlp = spacy.load("en_core_web_sm")
      # self.wiki_api = wikipediaapi.Wikipedia('en')
      self.site = mwclient.Site('en.wikipedia.org')

  def _augment_tag_list(self, tag_list, method, limit):
      if method == "wordnet":
          return self._augment_wordnet(tag_list, limit)
      elif method == "wikidata":
          return self._augment_wikidata(tag_list, limit)
      else:
          raise ValueError("Invalid augmentation method. Choose 'wordnet' or 'wikidata'.")

  def _augment_wordnet(self, tag_list, limit):
      augmented_tags = []
      for tag in tag_list:
          synsets = wn.synsets(tag)
          synonyms = []
          for syn in synsets:
              for lemma in syn.lemmas():
                  if lemma.name() not in synonyms:
                      synonyms.append(lemma.name())
                      if len(synonyms) >= limit:
                          break
              if len(synonyms) >= limit:
                  break
          augmented_tags.extend([tag] + synonyms[:limit])
      return augmented_tags

  def _augment_wikidata(self, tag_list, limit):
      augmented_tags = []
      for tag in tag_list:
          named_entities = self._wikidata_named_entities(tag, limit)
          augmented_tags.extend([tag] + named_entities)
      return augmented_tags

  def _wikidata_named_entities(self, tag, limit):
      try:
        page = self.site.pages[tag]
        if page.exists:
            summary = page.text()[0:50]  # Get the first 250 characters of the summary
            doc = self.nlp(summary)
            entities = [ent.text for ent in doc.ents]
            return entities[:limit]
        else:
            return []
      except Exception as e:
          print(f"Error: {e}")
          return []
  
  @timer_decorator
  def augment_tags(self, method="wordnet", limit=3, n_jobs=-1):
        if method == "wordnet":
            augmented_tags = Parallel(n_jobs=n_jobs)(
                delayed(self._augment_wordnet)(tag_list, limit) for tag_list in self.df["tag_list"]
            )
        elif method == "wikidata":
            augmented_tags = Parallel(n_jobs=n_jobs)(
                delayed(self._augment_wikidata)(tag_list, limit) for tag_list in self.df["tag_list"]
            )
        else:
            raise ValueError("Invalid augmentation method. Choose 'wordnet' or 'wikidata'.")

        self.df["augmented_tags"] = [', '.join(augmented_tag_list) for augmented_tag_list in augmented_tags]
  
  def compare_random_tags(self, num_samples=10):
      if "augmented_tags" not in self.df.columns:
          print("Please run the augment_tags method first.")
          return

      random_indices = random.sample(range(len(self.df)), num_samples)
      for idx in random_indices:
          original_tags = ', '.join(self.df["tag_list"].iloc[idx])
          augmented_tags = self.df["augmented_tags"].iloc[idx]
          print(f"Sample #{idx}:")
          print(f"Original tags   : {original_tags}")
          print(f"Augmented tags  : {augmented_tags}\n")

class Results_EDA:
  def __init__(self, **kwargs):
    self.classes = kwargs
    self.explanation = str()
    self.train_results = dict()
    self.test_results = dict()
    self.train_confusion_results = dict()
    self.test_confusion_results = dict()

    self.results = update_results(kwargs, "results")
    self.confusion_results = update_results(kwargs, "confusion_results")

    self.train_results, self.test_results = self.train_test_result_split(self.results)
    self.train_confusion_results, self.test_confusion_results = self.train_test_result_split(self.confusion_results)

    self.show_confusion_results()
    self.show_results()

  def check_same_explanation(self):
    explanations = [getattr(obj, 'explanation', '') for obj in self.classes.values()]
    if all(exp == explanations[0] for exp in explanations):
        print("All classes have the same explanation.")
        self.explanation = explanations[0]
        return True
    else:
        print("The classes have different explanations.")
        return False

  def update_results(self, kwargs, attribute):
      combined_results = {}
      for key, obj in kwargs.items():
          result_dict = getattr(obj, attribute, {})
          combined_results.update(result_dict)
      return combined_results
  
  def train_test_result_split(self, r_dict):
    train_results = {}
    test_results = {}

    for key, value in r_dict.items():
        if value is None:
            continue
        if 'train' in key:
            train_results[key] = value
        elif 'test' in key:
            test_results[key] = value
    return train_results, test_results

  def show_confusion_results(self, train=False):
    if train:
      confusion_results = self.train_confusion_results
    else:
      confusion_results = self.test_confusion_results

    fig, ax = plt.subplots()
    labels = list(confusion_results.keys())
    x = np.arange(len(labels))
    bottoms = np.zeros(len(labels))
    for category in ['True Negative', 'False Positive', 'False Negative', 'True Positive']:
        bar_values = [model_results[category] for model_results in confusion_results.values()]
        ax.bar(x, bar_values, bottom=bottoms, label=category)
        bottoms += bar_values
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.title('Stacked Confusion Matrix Results')
    plt.xlabel('Models')
    plt.ylabel('Values')
    plt.xticks(rotation=90)
    plt.show

  def show_results(self, train=False, export_latex=False):
    if train:
      results = pd.DataFrame(self.train_results).T
    else:
      results = pd.DataFrame(self.test_results).T
    styled_results = results.style.background_gradient(cmap='Blues').set_caption(
      'Test Results')
    display(styled_results)
    if export_latex:
      if self.check_same_explanation:
        latex_code = results.to_latex(index=False)
        latex_name = self.explanation + '.tex'
        with open(latex_name, 'w') as f:
            f.write(latex_code)

      








