import json
import pickle
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import classification_report
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
# from scipy.stats import randint
# from imblearn.over_sampling import ADASYN, BorderlineSMOTE
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re


# Download nltk data (Use this if you haven't downloaded it yet)
# nltk.download('punkt') 

# Load intents file
with open('intents.json') as file:
    intents = json.load(file)

# Prepare data
patterns = []
tags = []
for intent in intents['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        tags.append(intent['tag'])
        
# Preprocessing function
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = re.sub(r'\W', ' ', text.lower())
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Apply preprocessing
patterns = [preprocess_text(pattern) for pattern in patterns]

# Vectorizer patterns with n-grams(1,2)
vectorizer = TfidfVectorizer(ngram_range=(1, 2))  # Using TfidfVectorizer

X = vectorizer.fit_transform(patterns)
y = np.array(tags)

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42, k_neighbors=2)
X_smote, y_smote = smote.fit_resample(X, y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier with chosen best paramaters
rfc_model = RandomForestClassifier(
    class_weight ='balanced', 
    random_state = 42,
    max_depth = 20,
    max_features = 'log2',
    min_samples_leaf = 1,
    min_samples_split = 6,
    n_estimators = 330,
)

# Train the model
rfc_model.fit(X_train, y_train)

# Make predictions
y_pred = rfc_model.predict(X_test)
print("\nClassification report:")
print(classification_report(y_test, y_pred, zero_division=1))
print("\n Show Accuracy Score:")
print(rfc_model.score(X_test, y_test))


# Perform cross-validation
def cross_validation(model, x, y):
    skf = StratifiedKFold(n_splits=5)
    scores = cross_val_score(model, x, y, cv=skf)
    print(f"\nCross-validation scores: {scores}")
    print(f"\nAverage cross-validation score: {scores.mean()} \n")
    
# Cross-Validation for Random Forest
cross_validation(rfc_model, X, y)

with open('random forest.pkl', 'wb') as file:
    pickle.dump(rfc_model, file)

with open('vectorizer.pkl', 'wb') as file:
    pickle.dump(vectorizer, file)
