import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

def load_data(file_path):
    df = pd.read_csv(file_path, sep='\t', header=None, names=['label', 'message'])
    return df['message'].values, df['label'].values

def train_and_save_model():
    # Load data
    train_messages, train_labels = load_data("train-data.tsv")
    
    # Create and train model
    model = make_pipeline(
        TfidfVectorizer(max_features=1000),
        MultinomialNB()
    )
    model.fit(train_messages, train_labels)
    
    # Save model
    joblib.dump(model, 'spam_model.joblib')
    print("Model trained and saved successfully!")

if __name__ == '__main__':
    train_and_save_model()