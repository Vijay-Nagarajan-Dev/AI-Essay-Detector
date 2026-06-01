#imports
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
import openai
import google.generativeai as genai
from sklearn.model_selection import cross_val_score
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, classification_report
import sys



essayData = "AI_Human.csv"
dataset = pd.read_csv(essayData).sample(n=10000, random_state=50)
data = dataset['text']
output = dataset['generated']

#lines 26 and 29 for reference is pretty much from geeksforgeeks: https://www.geeksforgeeks.org/nlp/text-classification-using-scikit-learn-in-nlp/
df = pd.DataFrame({'text': data, 'isAI': output})
#removing duplicates line was from Claude
df = df.drop_duplicates(subset='text')
df.head()
X = df['text']
y= df['isAI']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50, stratify=y)
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)  
clf = SVC(kernel='linear', C=0.1, class_weight='balanced', probability=True)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: ' + str(accuracy))


print(" " \
"" \
"" \
"" \
"" \
"" \
"")

topic = input("What was the essay topic: ")
essay = input("insert your student's essay: ")
wantToProceed = input("Do you want to check if it is AI: ")
if(wantToProceed == "y" or wantToProceed == "yes"):
    user_vectorized = vectorizer.transform([essay])
    prediction = clf.predict(user_vectorized)
    confidence = clf.predict_proba(user_vectorized)[0] #this line is written by claude

    genai.configure(api_key="API-KEY")
    gemModel = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction="Write an essay on the following topic"
    )
    response = gemModel.generate_content(topic)
    cosModel = SentenceTransformer('bert-base-nli-mean-tokens')
    embeddings = cosModel.encode([response, essay])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])
    percentage = float(similarity[0][0])
    
    if(prediction[0] == 1 and confidence[1] > 0.95):
        print("Per the first model, the student's essay was most likely written by some LLM")
    else:
        print("Per the first model, this essay is most likely written by human")

    print(" ")
    if(percentage >= 0.9):
        print("Per the second model, the essay is most likely written by AI" )
    else:
        print("Per the second model, the essay probably was not have written by AI")





