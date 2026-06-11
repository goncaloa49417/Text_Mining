
# Standard library imports
import os
import re
import json
import pickle
import warnings
from copy import deepcopy
from datetime import datetime
from collections import Counter
from typing import List, Union, Optional


# numerical and data handling
import numpy as np
import pandas as pd


# visualization tools for EDA
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud


# NLP (NLTK)
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer, PorterStemmer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)


# scikit-learn for traditional ML models and overall utilities
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    precision_recall_fscore_support
)
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.utils.class_weight import compute_class_weight


# Traditional ML models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
import xgboost as xgb


# Embeddings (Word2Vec)
import gensim
from gensim.models import Word2Vec
from sklearn.decomposition import PCA



# Transformer ML pipeline utilities
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader



# Transformers (Hugging Face)
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    BertForSequenceClassification,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    T5Tokenizer,
    T5ForConditionalGeneration
)


# Langchain for agentic AI based workflow (Extra Challenge 2)
import langchain
from langchain_openai import AzureChatOpenAI
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

#-----------------------------------------------------------------------------------------------------~

# Configuration and constants

# Download NLTK data (only run once)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# Suppress warnings
warnings.filterwarnings('ignore')

# Visualization settings
sns.set_theme(style='whitegrid', palette='muted')
plt.style.use('seaborn-v0_8-darkgrid')

# Label mapping for display
LABEL_MAP = {0: 'Bearish', 1: 'Bullish', 2: 'Neutral'}
PALETTE = {0: '#E24B4A', 1: '#1D9E75', 2: '#378ADD'}  # Red, Green, Blue
STOP_WORDS = set(stopwords.words('english'))

# Words that should NEVER be removed (extra safety)
SENTIMENT_CRITICAL_WORDS = {
    # Directional (price movement)
    'up', 'down', 'higher', 'lower', 'rise', 'fall', 'gain', 'loss',
    'increase', 'decrease', 'climb', 'drop', 'surge', 'plunge',
    'rally', 'decline', 'advance', 'retreat',
    
    # Sentiment strong indicators
    'upgrade', 'downgrade', 'upgraded', 'downgraded',
    'outperform', 'underperform', 'buy', 'sell', 'hold',
    'bullish', 'bearish', 'neutral',
    'positive', 'negative', 'mixed',
    
    # Action verbs with sentiment
    'beat', 'miss', 'exceed', 'disappoint',
    'cut', 'raise', 'trim', 'boost',
    'warn', 'caution', 'optimistic', 'pessimistic',
}

# Define constants as class attributes (move inside class, but keep as module-level if needed elsewhere)
LEMMATIZER = WordNetLemmatizer()
STEMMER    = PorterStemmer()
TOKENIZER  = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)

CONTRACTIONS = {
    "don't":"do not", "won't":"will not", "can't":"cannot",
    "isn't":"is not", "aren't":"are not", "wasn't":"was not",
    "weren't":"were not", "haven't":"have not", "hasn't":"has not",
    "hadn't":"had not", "didn't":"did not", "doesn't":"does not",
    "wouldn't":"would not", "couldn't":"could not", "shouldn't":"should not",
    "i'm":"i am", "i've":"i have", "i'll":"i will", "i'd":"i would",
    "it's":"it is", "that's":"that is", "there's":"there is",
    "they're":"they are", "they've":"they have", "they'll":"they will",
    "we're":"we are", "we've":"we have", "we'll":"we will",
    "you're":"you are", "you've":"you have", "you'll":"you will",
}

FINANCIAL_STOPWORDS = {
    'shares', 'stock', 'market', 'trading', 'investors', 'analyst', 
    'company', 'companies', 'price', 'target', 'outlook', 'firm',
    'according', 'report', 'reported', 'reports', 'says', 'said'
}
FINANCIAL_STOPWORDS.update(FINANCIAL_STOPWORDS)

#-------------------------------------------------------------------------------------------------------------

# Classes used in the notebook

class FinancialTextPreprocessor(BaseEstimator, TransformerMixin):
    """
    Financial text preprocessor with sentiment-critical word preservation.
    Inherits from scikit-learn's BaseEstimator and TransformerMixin.
    Applies a preprocessing pipeline that includes:
    - Lowercasing
    - Preserving sentiment indicators (symbols to special tokens)
    - Expanding contractions using CONTRACTIONS dictionary
    - Processing stock tickers (keep as 'STICKER_{TICKER}' or 'STOCK_TICKER')
    - Conditional number handling (keep percentages and dollar amounts with context)
    - Removing noise (URLs, mentions, punctuation)
    - Tokenization using TweetTokenizer
    - Removing stopwords but preserving sentiment-critical words
    - Handling negations (adding NOT_ prefix to words following negation words)
    - Lemmatization using WordNetLemmatizer, preserving directional words even if they look like stopwords
    - Optional stemming using PorterStemmer (applied after lemmatization to minimize loss of meaning). It will stem words like "rising" to "rise" but will not stem "up" or "down" since they are preserved as critical words.

    Returns:
    --------
    Preprocessed text as a single string (tokens joined by space)
    """
    
    def __init__(self, use_stemming=False, preserve_directional=True):
        """
        Args:
            use_stemming: Whether to apply stemming after lemmatization
            preserve_directional: Keep directional words (up/down/rise/fall)
        """
        self.use_stemming = use_stemming
        self.preserve_directional = preserve_directional
        
        # Initialize constants
        self.stop_words = STOP_WORDS.copy()
        self.lemmatizer = LEMMATIZER
        self.stemmer = STEMMER
        self.tokenizer = TOKENIZER
        self.contractions = CONTRACTIONS
        self.known_tickers = ['aapl', 'tsla', 'amzn', 'nflx', 'googl', 'msft', 'fb', 'nvda']
        
        # Words that should be preserved even if they look like stopwords
        self.preserved_words = SENTIMENT_CRITICAL_WORDS.copy()
    
    def fit(self, X, y=None):
        """
        Required for scikit-learn compatibility - just returns self
        """
        return self
    
    def transform(self, X):
        """
        Apply preprocessing to a list/corpus of texts
        X can be list, numpy array, or pandas Series
        """
        # Handle different input types
        if hasattr(X, 'values'):  # pandas Series/DataFrame
            X = X.values
        elif isinstance(X, list):
            X = X
        elif isinstance(X, np.ndarray):
            X = X
        
        # Preprocess each text
        return [self._preprocess_single(text) for text in X]
    
    def _preprocess_single(self, text):
        """
        preprocess a single text string with multiple steps, ensuring that sentiment-critical words are preserved throughout the process.
        """
        # Step 1: Lowercase (must be first for consistent pattern matching)
        text = self._to_lowercase(text)
        
        # Step 2: Preserve sentiment indicators before they're removed
        text = self._preserve_sentiment_indicators(text)
        
        # Step 3: Expand contractions before noise removal
        text = self._expand_contractions(text)
        
        # Step 4: Process stock tickers before they're removed
        text = self._process_stock_tickers(text)
        
        # Step 5: Conditional number handling before they're removed
        text = self._conditional_number_handling(text)
        
        # Step 6: Remove noise 
        text = self._remove_noise(text)
        
        # Step 7: Tokenize after cleaning
        tokens = self._tokenize(text)

        # Step 8: Remove stopwords (but preserve sentiment-critical words)
        tokens = self._remove_stopwords_preserve_critical(tokens)
        
        # Step 9: Handle negations after stopword removal
        tokens = self._handle_negations(tokens)
        
        # Step 10: Lemmatize after stopword removal
        tokens = self._lemmatize(tokens)
        
        # Step 11: Optional stemming
        if self.use_stemming:
            tokens = self._stem(tokens)
        
        # Return joined tokens
        return ' '.join(tokens)
    
    # Step-by-step methods for each preprocessing step
    
    def _to_lowercase(self, text):
        """Step 1 — must be first so all pattern matching below is case-safe."""
        return text.lower()
    
    def _preserve_sentiment_indicators(self, text):
        """Keep symbols that explicitly indicate sentiment before they're removed"""
        text = re.sub(r'⬆️?', ' UP_ARROW ', text)
        text = re.sub(r'⬇️?', ' DOWN_ARROW ', text)
        text = re.sub(r'\+\s*\d+%', ' POSITIVE_PCT ', text)
        text = re.sub(r'-\s*\d+%', ' NEGATIVE_PCT ', text)
        text = re.sub(r'(!{2,})', ' STRONG_EMPHASIS ', text)
        return text
    
    def _expand_contractions(self, text):
        """Step 2 — before noise removal."""
        for contraction, expansion in self.contractions.items():
            text = re.sub(r'\b' + re.escape(contraction) + r'\b', expansion, text)
        return text
    
    def _process_stock_tickers(self, text):
        """Keep as special tokens (recommended for small vocab)"""
        for ticker in self.known_tickers:
            text = re.sub(rf'\${ticker}\b', f'TICKER_{ticker.upper()}', text)
        text = re.sub(r'\$[a-z]{1,5}\b', 'STOCK_TICKER', text)
        return text
    
    def _conditional_number_handling(self, text):
        """Numbers are important in financial text - keep with context"""
        text = re.sub(r'(\d+(?:\.\d+)?)%', r' NUM_PERCENT ', text)
        text = re.sub(r'\$(\d+(?:\.\d+)?)', r' DOLLAR_AMOUNT ', text)
        text = re.sub(r'\d+(?:\.\d+)?', '', text)
        return text
    
    def _remove_noise(self, text):
        """Remove URLs, mentions, punctuation, etc."""
        text = re.sub(r'http\S+|www\.\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#', '', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _tokenize(self, text):
        """Tokenize after cleaning"""
        return self.tokenizer.tokenize(text)
    
    def _remove_stopwords_preserve_critical(self, tokens):
        """
        Remove stopwords but preserve words that are critical for sentiment
        """
        result = []
        for t in tokens:
            # Always keep sentiment-critical words
            if t in self.preserved_words:
                result.append(t)
            # Keep words that are not in stopwords and have meaningful length
            elif t not in self.stop_words and len(t) > 1:
                result.append(t)
            # Special case: lemmatized forms of directional words
            elif self.lemmatizer.lemmatize(t) in self.preserved_words:
                result.append(t)
        return result
    
    def _handle_negations(self, tokens):
        """Add NOT_ prefix to words following negation words"""
        negation_words = {'not', 'no', 'never', 'neither', 'nor'}
        processed = []
        negate = False
        
        for token in tokens:
            if token in negation_words:
                negate = True
                processed.append(token)
            elif negate:
                processed.append(f'NOT_{token}')
                negate = False
            else:
                processed.append(token)
        return processed
    
    def _lemmatize(self, tokens):
        """Lemmatize after stopword removal"""
        return [self.lemmatizer.lemmatize(t) for t in tokens]
    
    def _stem(self, tokens):
        """Stem as last step since it's destructive"""
        return [self.stemmer.stem(t) for t in tokens]
    
class StratifiedKFoldTrainer:
    """
    Complete training pipeline with StratifiedKFold + Class Weights
    Integrates with FinancialTextPreprocessor
    """
    
    def __init__(self, n_splits=5, weight_strategy='balanced', random_state=42):
        self.n_splits = n_splits
        self.weight_strategy = weight_strategy
        self.random_state = random_state
        self.skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        self.fold_results = []
        
    def _get_class_weights(self, y_train, strategy='balanced'):
        """Get class weights using different strategies"""
        classes = np.unique(y_train)
        
        if strategy == 'balanced':
            weights = compute_class_weight('balanced', classes=classes, y=y_train)
        elif strategy == 'inverse_frequency':
            freq = np.bincount(y_train)
            weights = 1.0 / freq
            weights = weights / weights.min()
        elif strategy == 'custom':
            # Custom weights for financial sentiment
            # Class 0: neutral/positive (lowest importance)
            # Class 1: mixed/neutral (medium importance)
            # Class 2: negative (highest importance)
            class_weights_map = {0: 1.0, 1: 1.5, 2: 3.0}
            return class_weights_map
        else:
            weights = compute_class_weight('balanced', classes=classes, y=y_train)
        
        return dict(zip(classes, weights))
    
    def train(self, texts, labels, model=None, preprocessor=None, vectorizer=None):
        """
        Main training method with cross-validation
        
        Parameters:
        -----------
        texts : array-like, shape (n_samples,)
            Raw text data
        labels : array-like, shape (n_samples,)
            Target labels (0, 1, 2)
        model : sklearn classifier
            Model to train (default: LogisticRegression)
        preprocessor : FinancialTextPreprocessor
            Custom text preprocessor
        vectorizer : sklearn vectorizer
            Converts text to features (default: TfidfVectorizer)
        """

        # DecoderClassifier handles its own training loop and does not use the standard preprocessor/vectorizer pipeline
        if isinstance(model, DecoderClassifier):
            return self._train_decoder(texts, labels, model)
    
        # Set defaults if not provided
        if model is None:
            model = LogisticRegression(max_iter=1000, random_state=self.random_state)
        
        if preprocessor is None:
            preprocessor = FinancialTextPreprocessor(use_stemming=False)
        
        if vectorizer is None:
            vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        
        # Store for reuse
        self.preprocessor = preprocessor
        self.vectorizer = vectorizer
        self.model_class = type(model)
        self.model_params = model.get_params()
        
        print(f"{'='*60}")
        print(f"STRATIFIED {self.n_splits}-FOLD CROSS-VALIDATION")
        print(f"{'='*60}")
        print(f"Total samples: {len(texts)}")
        print(f"Class distribution: {np.bincount(labels)}")
        print(f"Preprocessing: use_stemming={preprocessor.use_stemming}")
        print(f"Vectorizer: {type(vectorizer).__name__} with {vectorizer.max_features if hasattr(vectorizer, 'max_features') else 'default'} features")
        print(f"Model: {type(model).__name__}")
        print(f"Weight strategy: {self.weight_strategy}")
        print(f"{'='*60}\n")
        
        # Perform stratified cross-validation
        for fold, (train_idx, val_idx) in enumerate(self.skf.split(texts, labels)):
            print(f"\n{'─'*50}")
            print(f"FOLD {fold + 1}/{self.n_splits}")
            print(f"{'─'*50}")
            
            # Split data
            X_train_raw = [texts[i] for i in train_idx]
            X_val_raw = [texts[i] for i in val_idx]
            y_train = labels[train_idx]
            y_val = labels[val_idx]
            
            # Print fold class distribution
            print(f"Train class distribution: {np.bincount(y_train)}")
            print(f"Val class distribution:   {np.bincount(y_val)}")
            
            # PREPROCESS (fit on train, transform both)
            print("Preprocessing texts...")
            X_train_processed = preprocessor.transform(X_train_raw)
            X_val_processed = preprocessor.transform(X_val_raw)
            
            # VECTORIZE (fit on train, transform both)
            print("Vectorizing texts...")
            X_train_vec = vectorizer.fit_transform(X_train_processed)
            X_val_vec = vectorizer.transform(X_val_processed)
            print(f"Feature matrix shape: {X_train_vec.shape}")
            
            # Get class weights for this fold
            weight_dict = self._get_class_weights(y_train, self.weight_strategy)
            print(f"Class weights: {weight_dict}")
            
            # Train model
            print("Training model...")
            model_clone = self.model_class(**self.model_params)
            
            if hasattr(model_clone, 'class_weight'):
                model_clone.set_params(class_weight=weight_dict)
                model_clone.fit(X_train_vec, y_train)
            else:
                # For models without class_weight parameter
                sample_weights = np.array([weight_dict[y] for y in y_train])
                model_clone.fit(X_train_vec, y_train, sample_weight=sample_weights)
            
            # Predict and evaluate
            y_pred = model_clone.predict(X_val_vec)
            
            # Calculate metrics
            f1_macro = f1_score(y_val, y_pred, average='macro')
            f1_weighted = f1_score(y_val, y_pred, average='weighted')
            f1_per_class = f1_score(y_val, y_pred, average=None)
            
            print(f"\nFold {fold + 1} Results:")
            print(f"  Macro F1:    {f1_macro:.4f}")
            print(f"  Weighted F1: {f1_weighted:.4f}")
            print(f"  Per-class F1: {f1_per_class}")
            print(f"\nClassification Report:")
            print(classification_report(y_val, y_pred, target_names=['Class 0', 'Class 1', 'Class 2']))
            
            # Store results
            self.fold_results.append({
                'fold': fold,
                'model': model_clone,
                'preprocessor': preprocessor,
                'vectorizer': vectorizer,
                'f1_macro': f1_macro,
                'f1_weighted': f1_weighted,
                'f1_per_class': f1_per_class,
                'y_true': y_val,
                'y_pred': y_pred
            })
        
        # Aggregate results
        self._print_summary()
        return self.fold_results
    
    def _train_decoder(self, texts, labels, model):
        """
        Stratified K-Fold loop for DecoderClassifier.
        No vectorizer or preprocessor. Raw text goes straight to the model.
        """
        texts  = np.array(texts)
        labels = np.array(labels)

        print(f"{'='*60}")
        print(f"STRATIFIED {self.n_splits}-FOLD CV  —  DecoderClassifier")
        print(f"{'='*60}")
        print(f"Total samples : {len(texts)}")
        print(f"Class dist.   : {np.bincount(labels)}")
        print(f"Model         : {model.model_name}")
        print(f"{'='*60}\n")

        for fold, (train_idx, val_idx) in enumerate(self.skf.split(texts, labels)):
            print(f"\n{'─'*50}")
            print(f"FOLD {fold + 1}/{self.n_splits}")
            print(f"{'─'*50}")

            X_train_raw = texts[train_idx].tolist()
            X_val_raw   = texts[val_idx].tolist()
            y_train     = labels[train_idx]
            y_val       = labels[val_idx]

            print(f"Train dist: {np.bincount(y_train)}")
            print(f"Val   dist: {np.bincount(y_val)}")

            fold_model = DecoderClassifier(
                model_name        = model.model_name,
                max_input_length  = model.max_input_length,
                max_target_length = model.max_target_length,
                batch_size        = model.batch_size,
                learning_rate     = model.learning_rate,
            )

            fold_model.fit(X_train_raw, y_train.tolist(),
                        val_texts=X_val_raw, val_labels=y_val.tolist())

            y_pred    = fold_model.predict(X_val_raw)
            f1_macro  = f1_score(y_val, y_pred, average='macro')
            f1_weighted = f1_score(y_val, y_pred, average='weighted')
            f1_per_class = f1_score(y_val, y_pred, average=None)

            print(f"\nFold {fold + 1} Results:")
            print(f"  Macro F1   : {f1_macro:.4f}")
            print(f"  Weighted F1: {f1_weighted:.4f}")
            print(classification_report(y_val, y_pred,
                                        target_names=['bearish', 'bullish', 'neutral']))

            self.fold_results.append({
                'fold'        : fold,
                'model'       : fold_model,
                'preprocessor': None,
                'vectorizer'  : None,
                'f1_macro'    : f1_macro,
                'f1_weighted' : f1_weighted,
                'f1_per_class': f1_per_class,
                'y_true'      : y_val,
                'y_pred'      : y_pred,
            })

        self._print_summary()
        return self.fold_results
    
    def _print_summary(self):
        """Print aggregated results across all folds"""
        print(f"\n{'='*60}")
        print(f"FINAL RESULTS - {self.n_splits}-FOLD STRATIFIED CV")
        print(f"{'='*60}")
        
        macro_f1_scores = [r['f1_macro'] for r in self.fold_results]
        weighted_f1_scores = [r['f1_weighted'] for r in self.fold_results]
        
        # Per-class across folds
        per_class_f1 = np.array([r['f1_per_class'] for r in self.fold_results])
        
        print(f"\n{'─'*70}")
        print("PER-CLASS F1 SCORES VISUALIZATION:")
        print(f"{'─'*70}")
        
        classes = [
            ("BEARISH", 0, 0.90),  # Target 0.90 for bearish
            ("BULLISH", 1, 0.90),   # Target 0.90 for bullish
            ("NEUTRAL", 2, 0.95)    # Target 0.95 for neutral 
        ]
        
        for name, idx, target in classes:
            mean_score = np.mean(per_class_f1[:, idx])
            std_score = np.std(per_class_f1[:, idx])
            
            # Create bar
            bar_length = int(mean_score * 50)
            bar = "█" * bar_length
            spaces = " " * (50 - bar_length)
            
            # Target marker
            target_bar_length = int(target * 50)
            target_indicator = "▼" if target_bar_length > bar_length else "▲" if target_bar_length < bar_length else "●"
            
            print(f"{name}: {bar}{spaces} {mean_score:.3f} (±{std_score:.3f}) {target_indicator} target={target}")
        
        print(f"{'='*70}")
        # Best fold
        best_fold_idx = np.argmax(macro_f1_scores)
        print(f"\nBest fold: Fold {best_fold_idx + 1} (Macro F1: {macro_f1_scores[best_fold_idx]:.4f})")
    
    def predict(self, texts):
        """Predict using the best model from cross-validation"""
        best_fold = np.argmax([r['f1_macro'] for r in self.fold_results])
        best_model = self.fold_results[best_fold]['model']
        best_preprocessor = self.fold_results[best_fold]['preprocessor']
        best_vectorizer = self.fold_results[best_fold]['vectorizer']
        
        processed = best_preprocessor.transform(texts)
        vectorized = best_vectorizer.transform(processed)
        return best_model.predict(vectorized)

class BoWFeatureExtractor:
    """
    Bag of Words with financial domain optimizations
    """
    
    def __init__(self, max_features=10000, use_ngrams=True, use_svd=False):
        self.max_features = max_features
        self.use_ngrams = use_ngrams
        self.use_svd = use_svd
        
        # TF-IDF with financial-specific parameters
        ngram_range = (1, 3) if use_ngrams else (1, 1)
        
        self.tfidf = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,           # Ignore terms that appear in <2 documents
            max_df=0.85,        # Ignore terms that appear in >85% docs (too common)
            sublinear_tf=True,  # Use 1+log(tf) scaling
            norm='l2',          # Euclidean normalization
            use_idf=True
        )
        
        if use_svd:
            self.svd = TruncatedSVD(n_components=500, random_state=42)
    
    def fit_transform(self, texts, y=None):
        X = self.tfidf.fit_transform(texts)
        if self.use_svd:
            X = self.svd.fit_transform(X)
        return X
    
    def transform(self, texts):
        X = self.tfidf.transform(texts)
        if self.use_svd:
            X = self.svd.transform(X)
        return X
    
    def get_top_features(self, n=20):
        """Get top features by TF-IDF score for interpretability"""
        feature_names = self.tfidf.get_feature_names_out()
        idf_scores = self.tfidf.idf_
        top_idx = np.argsort(idf_scores)[-n:][::-1]
        
        print("\nTop features by IDF (most distinctive):")
        for idx in top_idx:
            print(f"  {feature_names[idx]}: {idf_scores[idx]:.3f}")
        return feature_names[top_idx]    

class Word2VecFeatureExtractor:
    """
    Word2Vec with document-level aggregation
    """
    
    def __init__(self, vector_size=300, window=5, min_count=2, 
                 workers=4, aggregation='mean', use_pretrained=False):
        """
        Args:
            aggregation: 'mean', 'sum', 'tfidf_weighted', or 'concat_pool'
        """
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.aggregation = aggregation
        self.use_pretrained = use_pretrained
        self.model = None
        self.tfidf_vectorizer = None
    
    def fit(self, texts, y=None):
        """Train Word2Vec model on corpus"""
        tokenized_texts = [text.split() for text in texts]
        
        # Train from scratch
        print(f"Training Word2Vec on {len(tokenized_texts)} documents...")
        self.model = Word2Vec(
            sentences=tokenized_texts,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            sg=1,  # Skip-gram (better for small data, handles rare words)
            epochs=10
        )
        
        # Fit TF-IDF for weighted aggregation
        if self.aggregation == 'tfidf_weighted':
            self.tfidf_vectorizer = TfidfVectorizer()
            self.tfidf_vectorizer.fit(texts)
        
        return self
    
    def transform(self, texts):
        """Convert texts to document vectors"""
        tokenized_texts = [text.split() for text in texts]
        doc_vectors = []
        
        for tokens in tokenized_texts:
            vectors = []
            weights = []
            
            for token in tokens:
                if token in self.model.wv:
                    vec = self.model.wv[token]
                    vectors.append(vec)
                    
                    # Get weight for this token
                    if self.aggregation == 'tfidf_weighted' and self.tfidf_vectorizer:
                        weight = self._get_tfidf_weight(token, texts)
                        weights.append(weight)
            
            if not vectors:
                # Empty document - return zero vector
                doc_vectors.append(np.zeros(self.vector_size))
                continue
            
            vectors = np.array(vectors)
            
            if self.aggregation == 'mean':
                doc_vec = np.mean(vectors, axis=0)
            elif self.aggregation == 'sum':
                doc_vec = np.sum(vectors, axis=0)
            elif self.aggregation == 'tfidf_weighted':
                weights = np.array(weights).reshape(-1, 1)
                doc_vec = np.sum(vectors * weights, axis=0) / (np.sum(weights) + 1e-8)
            else:
                # Concat min, max, mean pooling
                doc_vec = np.concatenate([
                    np.mean(vectors, axis=0),
                    np.max(vectors, axis=0),
                    np.min(vectors, axis=0),
                    np.median(vectors, axis=0)
                ])
            
            doc_vectors.append(doc_vec)
        
        return np.array(doc_vectors)
    
    def fit_transform(self, texts, y=None):
        self.fit(texts, y)
        return self.transform(texts)
    
    def _get_tfidf_weight(self, token, texts):
        """Get TF-IDF weight for a token using the vectorizer."""
        if self.tfidf_vectorizer is None:
            return 1.0
        
        # Check if token exists in vocabulary
        if token in self.tfidf_vocabulary:
            # Get the index of the token
            token_idx = self.tfidf_vocabulary[token]
            # Return the IDF score (global importance across corpus)
            # This is the inverse document frequency component of TF-IDF
            return self.tfidf_idf_scores[token_idx]
        else:
            # Token not in vocabulary (rare or unseen)
            return 0.0
    
    def get_most_similar_words(self, word, topn=10):
        """Explore word relationships"""
        if self.model and word in self.model.wv:
            return self.model.wv.most_similar(word, topn=topn)
        return []
    
    def visualize_embeddings(self, words, save_path=None):
        """Visualize word embeddings in 2D"""
        import matplotlib.pyplot as plt
        
        vectors = []
        valid_words = []
        
        for word in words:
            if word in self.model.wv:
                vectors.append(self.model.wv[word])
                valid_words.append(word)
        
        vectors = np.array(vectors)
        pca = PCA(n_components=2)
        vectors_2d = pca.fit_transform(vectors)
        
        plt.figure(figsize=(12, 8))
        for i, word in enumerate(valid_words):
            x, y = vectors_2d[i]
            plt.scatter(x, y)
            plt.annotate(word, (x, y), fontsize=10)
        
        plt.title("Word Embeddings Visualization")
        if save_path:
            plt.savefig(save_path)
        plt.show()        

class TokenizedTweetDataset(Dataset):
    """Tokenized dataset for both feature extraction and fine-tuning."""
    
    def __init__(self, texts, tokenizer, max_length, labels=None):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors='pt'
        )
        self.labels = labels

    def __len__(self):
        return self.encodings['input_ids'].shape[0]

    def __getitem__(self, idx):
        item = {
            'input_ids'     : self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
        }
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
    
class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop('labels')
        outputs = model(**inputs)
        loss    = nn.CrossEntropyLoss(weight=self.class_weights)(
                      outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss
    
class TraditionalMLClassifier:
    """
    Wrapper for traditional ML classifiers with hyperparameter tuning
    Tuning happens ONCE before cross-validation
    """
    
    def __init__(self, model_type='logistic_regression', random_state=42):
        self.model_type = model_type
        self.random_state = random_state
        self.model = self._get_model()
        self.best_params = None
        self.is_tuned = False
    
    def _get_model(self):
        """Get model with default parameters"""
        if self.model_type == 'logistic_regression':
            return LogisticRegression(
                max_iter=1000, 
                random_state=self.random_state,
                class_weight='balanced'
            )
        elif self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                class_weight='balanced'
            )
        elif self.model_type == 'xgboost':
            return xgb.XGBClassifier(
                n_estimators=100,
                random_state=self.random_state,
                eval_metric='mlogloss'
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def get_hyperparameters(self):
        """Reduced grid for 15/20/65 market sentiment - max ~50-100 combos per model"""
        
        if self.model_type == 'logistic_regression':
            return {
                'C': [0.1, 1.0, 10.0],                    # 3 values (was 8)
                'penalty': ['l2'],                         # l1 rarely better with text
                'solver': ['lbfgs'],                       # fastest for l2
                'class_weight': ['balanced', {0: 2.5, 1: 1.5, 2: 0.7}],  # 2 options
                'max_iter': [1000]                         # fixed, sufficient
            }  
        
        elif self.model_type == 'random_forest':
            return {
                'n_estimators': [200, 500],                # 2 values (was 5)
                'max_depth': [20, None],                   # 2 values (was 5)
                'min_samples_split': [5, 10],              # 2 values (was 5)
                'min_samples_leaf': [2, 4],                # 2 values (was 4)
                'class_weight': ['balanced_subsample'],    # 1 best choice
                'max_features': ['sqrt']                   # 1 standard choice
            }  
        
        elif self.model_type == 'xgboost':
            return {
                'n_estimators': [150, 300],                # 2 values
                'max_depth': [4, 6],                       # 2 values
                'learning_rate': [0.05, 0.1],              # 2 values
                'subsample': [0.8],                        # fixed
                'colsample_bytree': [0.8],                 # fixed
                'scale_pos_weight': [2.5]                  # bullish=0, bearish=2 → ratio ~2.5
            }
        
        return {}
    
    def tune_before_cv(self, X, y, cv_inner=3, test_size=0.2):
        """
        Tune hyperparameters ONCE before cross-validation
        
        This prevents data leakage and ensures fair evaluation.
        
        Args:
            X: Feature matrix
            y: Labels
            cv_inner: Number of folds for inner CV during tuning
            test_size: Proportion to hold out for final testing (0 = no holdout)
        """
        param_grid = self.get_hyperparameters()
        if not param_grid:
            print(f"No hyperparameters to tune for {self.model_type}")
            self.is_tuned = True
            return self.model, None
        
        print(f"\nTuning {self.model_type} hyperparameters...")
        
        # Hold out a small test set for final validation
        if test_size > 0:
            X_tune, X_test, y_tune, y_test = train_test_split(
                X, y, test_size=test_size, stratify=y, random_state=self.random_state
            )
        else:
            X_tune, y_tune = X, y
            X_test, y_test = None, None
        
        # Grid search with inner cross-validation
        grid_search = GridSearchCV(
            self.model, param_grid, cv=cv_inner, 
            scoring='f1_macro', n_jobs=-1, verbose=0
        )
        grid_search.fit(X_tune, y_tune)
        
        self.best_params = grid_search.best_params_
        self.model = grid_search.best_estimator_
        self.is_tuned = True
        
        print(f"Best params: {self.best_params}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        # Evaluate on held-out test set if available
        if X_test is not None:
            y_pred = self.model.predict(X_test)
            test_f1 = f1_score(y_test, y_pred, average='macro')
            print(f"Hold-out test F1: {test_f1:.4f}")
            return self.model, test_f1
        
        return self.model, None
    
    def fit(self, X_train, y_train, sample_weight=None):
        """Fit the model (uses tuned parameters if available)"""
        if self.model_type == 'xgboost' and sample_weight is not None:
            self.model.fit(X_train, y_train, sample_weight=sample_weight)
        else:
            self.model.fit(X_train, y_train)
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        return None

class TransformerClassifier:
    """
    Single class handling both feature extraction and fine-tuning.

    Usage:
        # Feature extraction only (e.g. embeddings → feed into sklearn model)
        model = TransformerClassifier()
        embeddings = model.extract_features(texts, pooling='cls')

        # Fine-tuning for direct classification
        model = TransformerClassifier()
        model.fine_tune(X_train, y_train, X_val, y_val)
        preds = model.predict(X_test)
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        use_finbert : bool = True,
        num_labels  : int  = 3,
        max_length  : int  = 128,
        batch_size  : int  = 16,
        learning_rate: float = 2e-5,
    ):
        self.model_name    = model_name
        self.num_labels    = num_labels
        self.max_length    = max_length
        self.batch_size    = batch_size
        self.learning_rate = learning_rate
        self.device        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_finetuned  = False
        self.class_weights = None

        print(f"Model : {self.model_name}")
        print(f"Device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Base encoder — used for feature extraction
        self._encoder = AutoModel.from_pretrained(self.model_name).to(self.device)

        # Classification head — used for fine-tuning
        self._classifier = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        ).to(self.device)

    # Feature extraction

    def extract_features(self, texts: List[str], pooling: str = 'cls') -> np.ndarray:
        """
        Extract embeddings without fine-tuning.
        pooling: 'cls' | 'mean' | 'max'
        Returns array of shape (n_samples, hidden_size)
        """
        self._encoder.eval()
        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            inputs = self.tokenizer(
                list(texts[i:i + self.batch_size]),
                truncation=True, padding=True,
                max_length=self.max_length,
                return_tensors='pt'
            ).to(self.device)

            with torch.no_grad():
                hidden = self._encoder(**inputs).last_hidden_state  # (B, S, H)
                mask   = inputs['attention_mask']

                if pooling == 'cls':
                    emb = hidden[:, 0, :]
                elif pooling == 'mean':
                    emb = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
                elif pooling == 'max':
                    emb = (hidden * mask.unsqueeze(-1)).max(1)[0]
                else:
                    raise ValueError(f"Unknown pooling: {pooling}")

            all_embeddings.append(emb.cpu().numpy())

        result = np.vstack(all_embeddings)
        print(f"Embeddings shape: {result.shape}")
        return result

    # Fine-tuning 

    def fine_tune(
        self,
        train_texts : List[str],
        train_labels: List[int],
        val_texts   : Optional[List[str]] = None,
        val_labels  : Optional[List[int]] = None,
        epochs      : int  = 3,
        use_class_weights: bool = True,
        use_hf_trainer   : bool = False,
    ):
        """Fine-tune the classifier head on your data."""

        if use_class_weights:
            classes = np.unique(train_labels)
            weights = compute_class_weight('balanced', classes=classes, y=train_labels)
            self.class_weights = torch.tensor(weights, dtype=torch.float).to(self.device)
            print(f"Class weights: {dict(zip(classes, weights.round(3)))}")

        train_ds = TokenizedTweetDataset(train_texts, self.tokenizer, self.max_length, train_labels)
        val_ds   = TokenizedTweetDataset(val_texts,   self.tokenizer, self.max_length, val_labels) \
                   if val_texts is not None else None

        if use_hf_trainer:
            self._train_with_trainer(train_ds, val_ds, epochs)
        else:
            self._train_custom_loop(train_ds, val_ds, epochs)

        self.is_finetuned = True

    def _train_custom_loop(self, train_ds, val_ds, epochs):
        """Custom training loop — more control, easier to debug."""
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=self.batch_size) if val_ds else None

        optimizer = torch.optim.AdamW(self._classifier.parameters(), lr=self.learning_rate)
        loss_fn   = nn.CrossEntropyLoss(weight=self.class_weights)

        best_f1, best_state = 0, None

        for epoch in range(1, epochs + 1):
            # train 
            self._classifier.train()
            total_loss = 0

            for batch in train_loader:
                ids   = batch['input_ids'].to(self.device)
                mask  = batch['attention_mask'].to(self.device)
                lbls  = batch['labels'].to(self.device)

                optimizer.zero_grad()
                loss = loss_fn(self._classifier(ids, attention_mask=mask).logits, lbls)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._classifier.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)

            # validate
            if val_loader:
                preds, truth = [], []
                self._classifier.eval()

                with torch.no_grad():
                    for batch in val_loader:
                        ids  = batch['input_ids'].to(self.device)
                        mask = batch['attention_mask'].to(self.device)
                        out  = self._classifier(ids, attention_mask=mask)
                        preds.extend(torch.argmax(out.logits, 1).cpu().numpy())
                        truth.extend(batch['labels'].numpy())

                macro_f1 = f1_score(truth, preds, average='macro')
                print(f"Epoch {epoch}/{epochs} — Loss: {avg_loss:.4f} | Val Macro F1: {macro_f1:.4f}")

                if macro_f1 > best_f1:
                    best_f1    = macro_f1
                    best_state = {k: v.clone() for k, v in self._classifier.state_dict().items()}
                    print(f"  New best model (F1={best_f1:.4f})")
            else:
                print(f"Epoch {epoch}/{epochs} — Loss: {avg_loss:.4f}")

        if best_state:
            self._classifier.load_state_dict(best_state)

            print(f"\nBest validation F1 during training: {best_f1:.4f}")

            # sanity check
            preds = []
            truth = []

            self._classifier.eval()

            with torch.no_grad():
                for batch in val_loader:
                    ids = batch['input_ids'].to(self.device)
                    mask = batch['attention_mask'].to(self.device)

                    logits = self._classifier(
                        ids,
                        attention_mask=mask
                    ).logits

                    preds.extend(
                        torch.argmax(logits, dim=1).cpu().numpy()
                    )

                    truth.extend(
                        batch['labels'].cpu().numpy()
                    )

            restored_f1 = f1_score(
                truth,
                preds,
                average='macro'
            )

            print(
                f"Validation F1 after restoring best checkpoint: "
                f"{restored_f1:.4f}"
            )

    def _train_with_trainer(self, train_ds, val_ds, epochs):
        """HuggingFace Trainer — simpler but less flexible."""
        args = TrainingArguments(
            output_dir              = './results',
            num_train_epochs        = epochs,
            per_device_train_batch_size = self.batch_size,
            per_device_eval_batch_size  = self.batch_size,
            warmup_steps            = 100,
            weight_decay            = 0.01,
            eval_strategy           = 'epoch' if val_ds else 'no',
            save_strategy           = 'epoch',
            load_best_model_at_end  = val_ds is not None,
            metric_for_best_model   = 'eval_loss',
            fp16                    = torch.cuda.is_available(),
            report_to               = 'none',
        )
        WeightedTrainer(
            model         = self._classifier,
            args          = args,
            train_dataset = train_ds,
            eval_dataset  = val_ds,
            class_weights = self.class_weights,
        ).train()

    # inference

    def predict(self, texts: List[str]) -> np.ndarray:
        """Returns predicted class labels (0, 1, 2)."""
        return np.argmax(self.predict_proba(texts), axis=1)

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Returns softmax probabilities, shape (n_samples, 3)."""
        if not self.is_finetuned:
            raise ValueError("Call fine_tune() before predicting.")
        
        self._classifier.eval()

        all_probs = []

        for i in range(0, len(texts), self.batch_size):
            inputs = self.tokenizer(
                list(texts[i:i + self.batch_size]),
                truncation=True, padding=True,
                max_length=self.max_length,
                return_tensors='pt'
            ).to(self.device)

            with torch.no_grad():
                logits = self._classifier(**inputs).logits
                probs = torch.softmax(logits, dim=1)

            all_probs.append(probs.cpu().numpy())

        return np.vstack(all_probs)  
              
class DecoderClassifier:
    """
    Decoder-based classifier using FLAN-T5 (seq2seq) for financial sentiment.

    Frames classification as text generation:
        Input : "classify sentiment: $AAPL upgraded to buy..."
        Output: "bullish"/"bearish"/"neutral"

    Compatible with ModelBundle (use is_decoder=True).
    """

    LABEL2TEXT = {0: 'bearish', 1: 'bullish', 2: 'neutral'}
    TEXT2LABEL = {'bearish': 0, 'bullish': 1, 'neutral': 2}

    def __init__(
        self,
        model_name       : str   = 'google/flan-t5-small',
        max_input_length : int   = 128,
        max_target_length: int   = 8,
        batch_size       : int   = 8,
        learning_rate    : float = 3e-4,
        device           : Optional[str] = None,
    ):
        self.model_name          = model_name
        self.max_input_length    = max_input_length
        self.max_target_length   = max_target_length
        self.batch_size          = batch_size
        self.learning_rate       = learning_rate
        self.device = torch.device(
            device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        )
        self.is_fitted = False

        print(f"DecoderClassifier - model : {self.model_name}")
        print(f"DecoderClassifier - device: {self.device}")

        # Load tokenizer and model
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.model     = T5ForConditionalGeneration.from_pretrained(
            self.model_name
        ).to(self.device)

    def get_params(self, deep=True):
        return {
            'model_name'       : self.model_name,
            'max_input_length' : self.max_input_length,
            'max_target_length': self.max_target_length,
            'batch_size'       : self.batch_size,
            'learning_rate'    : self.learning_rate,
        }

    def fine_tune(
        self,
        train_texts : List[str],
        train_labels: List[int],
        val_texts   : Optional[List[str]] = None,
        val_labels  : Optional[List[int]] = None,
        epochs      : int = 3,
        **kwargs,
    ):
        # Alias for fit() so callers can treat this like TransformerClassifier.
        return self.fit(train_texts, train_labels, val_texts, val_labels, epochs)

    # Internal dataset class for seq2seq fine-tuning
    class _Seq2SeqDataset(Dataset):
        def __init__(self, input_encodings, target_encodings):
            self.input_encodings  = input_encodings
            self.target_encodings = target_encodings

        def __len__(self):
            return self.input_encodings['input_ids'].shape[0]

        def __getitem__(self, idx):
            labels = self.target_encodings['input_ids'][idx].clone()
            labels[labels == 0] = -100
            return {
                'input_ids'     : self.input_encodings['input_ids'][idx],
                'attention_mask': self.input_encodings['attention_mask'][idx],
                'labels'        : labels,
            }
        
    def _format_inputs(self, texts: List[str]) -> List[str]:
        return [f"classify sentiment: {t}" for t in texts]

    # Tokenize inputs for seq2seq fine-tuning
    def _encode_inputs(self, texts: List[str]):
        return self.tokenizer(
            self._format_inputs(texts),
            truncation=True, padding=True,
            max_length=self.max_input_length,
            return_tensors='pt',
        )

    # Encode labels as text and tokenize for seq2seq training
    def _encode_labels(self, labels: List[int]):
        label_texts = [self.LABEL2TEXT[l] for l in labels]
        return self.tokenizer(
            label_texts,
            truncation=True, padding=True,
            max_length=self.max_target_length,
            return_tensors='pt',
        )

    def fit(
        self,
        train_texts : List[str],
        train_labels: List[int],
        val_texts   : Optional[List[str]] = None,
        val_labels  : Optional[List[int]] = None,
        epochs      : int = 3,
    ):
        print(f"\nFine-tuning {self.model_name} for {epochs} epoch(s)...")

        train_ds     = self._Seq2SeqDataset(
            self._encode_inputs(train_texts),
            self._encode_labels(train_labels),
        )
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        val_loader = None

        if val_texts is not None and val_labels is not None:
            val_ds     = self._Seq2SeqDataset(
                self._encode_inputs(val_texts),
                self._encode_labels(val_labels),
            )
            val_loader = DataLoader(val_ds, batch_size=self.batch_size)

        optimizer           = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        best_f1, best_state = 0.0, None

        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0.0

            # We feed input and target together for seq2seq training
            for batch in train_loader:
                input_ids      = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels         = batch['labels'].to(self.device)

                optimizer.zero_grad()
                loss = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                ).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)

            if val_loader:
                val_preds = self._generate_predictions(val_texts)
                macro_f1  = f1_score(val_labels, val_preds, average='macro')
                print(f"Epoch {epoch}/{epochs} — Loss: {avg_loss:.4f} | Val Macro F1: {macro_f1:.4f}")

                if macro_f1 > best_f1:
                    best_f1    = macro_f1
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                    print(f"  New best model (F1={best_f1:.4f})")
            else:
                print(f"Epoch {epoch}/{epochs} — Loss: {avg_loss:.4f}")

        if best_state:
            self.model.load_state_dict(best_state)
            print(f"\nRestored best checkpoint (Val Macro F1: {best_f1:.4f})")

        self.is_fitted = True
        return self

    def _generate_predictions(self, texts: List[str]) -> List[int]:
        self.model.eval()
        all_preds = []

        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
            inputs = self.tokenizer(
                self._format_inputs(batch_texts),
                truncation=True, padding=True,
                max_length=self.max_input_length,
                return_tensors='pt',
            ).to(self.device)

            with torch.no_grad():
                generated = self.model.generate(
                    input_ids      = inputs['input_ids'],
                    attention_mask = inputs['attention_mask'],
                    max_new_tokens = self.max_target_length,
                )

            decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            for token in decoded:
                all_preds.append(self.TEXT2LABEL.get(token.strip().lower(), 2))

        return all_preds

    def predict(self, texts: List[str]) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Call fit() before predict().")
        return np.array(self._generate_predictions(texts))

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Call fit() before predict_proba().")

        self.model.eval()

        label_token_ids = [
            self.tokenizer(self.LABEL2TEXT[i], return_tensors='pt').input_ids[0, 0]
            for i in range(3)
        ]
        all_probs = []

        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
            inputs = self.tokenizer(
                self._format_inputs(batch_texts),
                truncation=True, padding=True,
                max_length=self.max_input_length,
                return_tensors='pt',
            ).to(self.device)

            decoder_input_ids = torch.full(
                (len(batch_texts), 1),
                self.model.config.decoder_start_token_id,
                dtype=torch.long,
                device=self.device,
            )

            with torch.no_grad():
                logits = self.model(
                    **inputs,
                    decoder_input_ids=decoder_input_ids,
                ).logits[:, 0, :]

                label_logits = torch.stack(
                    [logits[:, tid] for tid in label_token_ids], dim=1
                )
                probs = torch.softmax(label_logits, dim=1)

            all_probs.append(probs.cpu().numpy())

        return np.vstack(all_probs)

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> 'DecoderClassifier':
        with open(path, 'rb') as f:
            return pickle.load(f)

class ModelBundle:
    """
    Unified wrapper for preprocessing + feature extraction + trained model.

    Flags:
        is_transformer : True for encoder fine-tuned models (BERT, DistilBERT, FinBERT)
        is_decoder     : True for seq2seq model (FLAN-T5 via DecoderClassifier)
    Both flags cause predict_proba() to call model.predict_proba(texts) directly,
    skipping the preprocessor and vectorizer steps.
    """

    def __init__(
        self,
        model,
        preprocessor   = None,
        vectorizer     = None,
        is_transformer : bool = False,
        is_decoder     : bool = False,
    ):
        self.model          = model
        self.preprocessor   = preprocessor
        self.vectorizer     = vectorizer
        self.is_transformer = is_transformer
        self.is_decoder     = is_decoder

    def predict(self, texts):
        return np.argmax(self.predict_proba(texts), axis=1)

    def predict_proba(self, texts):
        if self.is_transformer or self.is_decoder:
            return self.model.predict_proba(texts)

        # Traditional ML pipeline
        if self.preprocessor:
            texts = self.preprocessor.transform(texts)
        X = self.vectorizer.transform(texts) if self.vectorizer else texts
        return self.model.predict_proba(X)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)