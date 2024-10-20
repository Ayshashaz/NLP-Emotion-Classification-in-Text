#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
data = pd.read_csv('nlp_dataset.csv')
data.head()


# In[2]:


import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)
data['Processed_Comment'] = data['Comment'].apply(preprocess_text)
data[['Comment', 'Processed_Comment', 'Emotion']].head()


# In[4]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X_tfidf = tfidf_vectorizer.fit_transform(data['Processed Comment'])
tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
tfidf_df.head()


# In[5]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=1000)  
X_tfidf = tfidf_vectorizer.fit_transform(data['Processed_Comment'])
tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
tfidf_df.head()


# In[6]:


from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, data['Emotion'], test_size=0.2, random_state=42)
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)
svm_model = SVC(kernel='linear')  
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
nb_report = classification_report(y_test, y_pred_nb)
svm_report = classification_report(y_test, y_pred_svm)
nb_report, svm_report


# In[7]:


from sklearn.metrics import accuracy_score, classification_report
print("Naive Bayes Classification Report")
print(classification_report(y_test, y_pred_nb))
print("Accuracy: ", accuracy_score(y_test, y_pred_nb))

print("\nSVM Classification Report")
print(classification_report(y_test, y_pred_svm))
print("Accuracy: ", accuracy_score(y_test, y_pred_svm))


# In[ ]:




