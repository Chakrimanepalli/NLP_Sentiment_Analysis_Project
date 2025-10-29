"""
Sentiment Analysis and Classification Model
Combines VADER sentiment analysis with LSTM neural network
NO EMOJIS VERSION - STREAMLIT COMPATIBLE
"""

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Deep Learning imports
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
import pickle

# Download required NLTK data
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except:
    pass

class SentimentAnalyzer:
    """Comprehensive sentiment analysis using VADER, TextBlob, and LSTM"""

    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        self.lemmatizer = WordNetLemmatizer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()
        self.lstm_model = None
        self.tokenizer = None
        self.max_len = 200

    def clean_text(self, text):
        """Clean and preprocess text"""
        if pd.isna(text):
            return ""

        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def preprocess_text(self, text):
        """Advanced preprocessing with lemmatization"""
        text = self.clean_text(text)

        try:
            tokens = word_tokenize(text)
        except:
            tokens = text.split()

        if self.stop_words:
            tokens = [self.lemmatizer.lemmatize(word) for word in tokens 
                     if word not in self.stop_words and len(word) > 2]
        else:
            tokens = [self.lemmatizer.lemmatize(word) for word in tokens if len(word) > 2]

        return ' '.join(tokens)

    def vader_sentiment(self, text):
        """Get VADER sentiment scores"""
        scores = self.vader.polarity_scores(str(text))

        if scores['compound'] >= 0.05:
            sentiment = 'Positive'
        elif scores['compound'] <= -0.05:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'

        return {
            'vader_compound': scores['compound'],
            'vader_positive': scores['pos'],
            'vader_negative': scores['neg'],
            'vader_neutral': scores['neu'],
            'vader_sentiment': sentiment
        }

    def textblob_sentiment(self, text):
        """Get TextBlob sentiment scores"""
        blob = TextBlob(str(text))
        polarity = blob.sentiment.polarity

        if polarity > 0.1:
            sentiment = 'Positive'
        elif polarity < -0.1:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'

        return {
            'textblob_polarity': polarity,
            'textblob_subjectivity': blob.sentiment.subjectivity,
            'textblob_sentiment': sentiment
        }

    def analyze_sentiment(self, text):
        """Comprehensive sentiment analysis"""
        vader_results = self.vader_sentiment(text)
        textblob_results = self.textblob_sentiment(text)

        results = {**vader_results, **textblob_results}

        sentiments = [vader_results['vader_sentiment'], textblob_results['textblob_sentiment']]
        results['ensemble_sentiment'] = max(set(sentiments), key=sentiments.count)

        return results

    def prepare_lstm_data(self, df, text_column='review_text', sentiment_column='sentiment'):
        """Prepare data for LSTM model"""
        df['cleaned_text'] = df[text_column].apply(self.preprocess_text)

        sentiment_map = {'Positive': 2, 'Neutral': 1, 'Negative': 0}
        df['sentiment_encoded'] = df[sentiment_column].map(sentiment_map)

        self.tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')
        self.tokenizer.fit_on_texts(df['cleaned_text'])

        sequences = self.tokenizer.texts_to_sequences(df['cleaned_text'])
        padded = pad_sequences(sequences, maxlen=self.max_len, padding='post', truncating='post')

        return padded, df['sentiment_encoded'].values

    def build_lstm_model(self, vocab_size=5000, embedding_dim=128, lstm_units=64):
        """Build LSTM neural network for sentiment classification"""
        model = Sequential([
            Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=self.max_len),
            SpatialDropout1D(0.2),
            LSTM(lstm_units, dropout=0.2, recurrent_dropout=0.2, return_sequences=True),
            LSTM(lstm_units//2, dropout=0.2, recurrent_dropout=0.2),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(3, activation='softmax')
        ])

        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )

        return model

    def train_lstm_model(self, df, text_column='review_text', sentiment_column='sentiment', 
                        epochs=10, batch_size=128, validation_split=0.2):
        """Train LSTM model on review data"""
        X, y = self.prepare_lstm_data(df, text_column, sentiment_column)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.lstm_model = self.build_lstm_model()

        early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        history = self.lstm_model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stop],
            verbose=1
        )

        test_loss, test_accuracy = self.lstm_model.evaluate(X_test, y_test, verbose=0)
        print(f"\nTest Accuracy: {test_accuracy:.4f}")

        y_pred = np.argmax(self.lstm_model.predict(X_test), axis=1)

        sentiment_labels = ['Negative', 'Neutral', 'Positive']
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=sentiment_labels))

        return history

    def predict_sentiment_lstm(self, text):
        """Predict sentiment using trained LSTM model"""
        if self.lstm_model is None or self.tokenizer is None:
            return "Model not trained"

        cleaned = self.preprocess_text(text)
        sequence = self.tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(sequence, maxlen=self.max_len, padding='post')

        prediction = self.lstm_model.predict(padded, verbose=0)
        sentiment_idx = np.argmax(prediction[0])

        sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        confidence = float(prediction[0][sentiment_idx])

        return {
            'lstm_sentiment': sentiment_map[sentiment_idx],
            'lstm_confidence': confidence,
            'lstm_probabilities': {
                'negative': float(prediction[0][0]),
                'neutral': float(prediction[0][1]),
                'positive': float(prediction[0][2])
            }
        }

    def save_model(self, model_path='lstm_sentiment_model.h5', tokenizer_path='tokenizer.pkl'):
        """Save trained model and tokenizer"""
        if self.lstm_model:
            self.lstm_model.save(model_path)
        if self.tokenizer:
            with open(tokenizer_path, 'wb') as f:
                pickle.dump(self.tokenizer, f)
        print(f"Model saved to {model_path}")
        print(f"Tokenizer saved to {tokenizer_path}")

    def load_model(self, model_path='lstm_sentiment_model.h5', tokenizer_path='tokenizer.pkl'):
        """Load trained model and tokenizer"""
        self.lstm_model = load_model(model_path)
        with open(tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)
        print("Model and tokenizer loaded successfully")


class TraditionalSentimentClassifier:
    """Logistic Regression and Random Forest classifier for sentiment"""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.lr_model = LogisticRegression(max_iter=1000, random_state=42)
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

    def train(self, df, text_column='review_text', sentiment_column='sentiment'):
        """Train traditional ML models"""
        X = self.vectorizer.fit_transform(df[text_column])
        y = df[sentiment_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print("Training Logistic Regression...")
        self.lr_model.fit(X_train, y_train)
        lr_pred = self.lr_model.predict(X_test)
        lr_acc = accuracy_score(y_test, lr_pred)

        print(f"Logistic Regression Accuracy: {lr_acc:.4f}")
        print("\nLogistic Regression Classification Report:")
        print(classification_report(y_test, lr_pred))

        print("\nTraining Random Forest...")
        self.rf_model.fit(X_train, y_train)
        rf_pred = self.rf_model.predict(X_test)
        rf_acc = accuracy_score(y_test, rf_pred)

        print(f"Random Forest Accuracy: {rf_acc:.4f}")
        print("\nRandom Forest Classification Report:")
        print(classification_report(y_test, rf_pred))

        return {
            'lr_accuracy': lr_acc,
            'rf_accuracy': rf_acc
        }


if __name__ == "__main__":
    try:
        df = pd.read_csv('amazon_reviews.csv')

        analyzer = SentimentAnalyzer()

        print("Analyzing sentiments...")
        sentiment_results = df['review_text'].apply(analyzer.analyze_sentiment)
        sentiment_df = pd.DataFrame(sentiment_results.tolist())

        result_df = pd.concat([df, sentiment_df], axis=1)
        result_df['sentiment'] = result_df['ensemble_sentiment']

        print("\nTraining LSTM model...")
        history = analyzer.train_lstm_model(result_df)

        analyzer.save_model()

        result_df.to_csv('sentiment_analysis_results.csv', index=False)
        print("\nResults saved to 'sentiment_analysis_results.csv'")

    except Exception as e:
        print(f"Error: {e}")
