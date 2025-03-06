import pandas as pd
from nltk.corpus import stopwords
from sklearn.preprocessing import OrdinalEncoder
import string
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import re
import pickle

df = pd.read_csv('data/Dataset-SA.csv')
df.head()

print(df.isnull().sum())
print(df.shape)
print(df['Review'].isnull().sum())
print("null val = {:.2f} %".format((df['Review'].isnull().sum()/df.shape[0])*100))

df.dropna(inplace=True)
data = df[['Rate','Review','Summary','Sentiment']]
print(data['Sentiment'].value_counts())

data = data[data['Sentiment'] !='neutral']


stpwords = set(stopwords.words('english'))  # Convert stopwords list to a set for fast lookup
stem = PorterStemmer()  # Create an instance

def clean(i):
    i = re.sub(r'READ MORE', '', i)
    i = i.lower()  # Convert to lowercase
    words = word_tokenize(i)  # Tokenize sentence
    words = [stem.stem(word) for word in words if word not in stpwords]
    words = [word for word in words if word not in string.punctuation]
    return ' '.join(words)  # Join words back into a sentence
    
print(data['Sentiment'].value_counts())

data['Review'] = data['Review'].apply(clean)
data['Summary'] = data['Summary'].apply(clean)
ord = OrdinalEncoder(categories=[['negative','positive']])
data['Sentiment'] = ord.fit_transform(data[['Sentiment']])

from imblearn.under_sampling import RandomUnderSampler

# Assuming df has 'Review', 'Summary', and 'Sentiment' columns
X = data[['Review', 'Summary']]  # Selecting both text columns
y = data['Sentiment']  # Target column (1 = Positive, 0 = Negative)

# Initialize the undersampler
rus = RandomUnderSampler(sampling_strategy='auto', random_state=42)

# Resample the dataset
X_resampled, y_resampled = rus.fit_resample(X, y)

# Create the new balanced DataFrame
df_balanced = pd.DataFrame({'Review': X_resampled['Review'], 
                            'Summary': X_resampled['Summary'], 
                            'Sentiment': y_resampled})

# Display the new balanced dataset
print(df_balanced.head())

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

X = df_balanced[['Review','Summary']]
y = df_balanced['Sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
vectorizer_review = CountVectorizer(ngram_range=(1,2))
vectorizer_summary = CountVectorizer(ngram_range=(1,2))

review_bow_train = vectorizer_review.fit_transform(X_train['Review'])
summary_bow_train = vectorizer_summary.fit_transform(X_train['Summary'])

review_bow_test = vectorizer_review.transform(X_test['Review'])
summary_bow_test = vectorizer_summary.transform(X_test['Summary'])

# Merge Sparse Matrices Horizontally
from scipy.sparse import hstack

X_train_bow = hstack([review_bow_train, summary_bow_train])
X_test_bow = hstack([review_bow_test, summary_bow_test])

model = MultinomialNB()
# model.fit(X_train_bow, y_train)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X_train_bow, y_train, cv=5)    
print("Mean CV Accuracy:", scores.mean())

# Predict & Evaluate
pred = model.predict(X_test_bow)
print("Accuracy:", accuracy_score(y_test, pred))
with open('sentiment_model2.pkl', 'wb') as file:
    pickle.dump((model, vectorizer_review, vectorizer_summary), file)
