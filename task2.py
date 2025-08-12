# Simple Task: Scrape Machine Learning Wikipedia and apply NLP

import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from textblob import TextBlob

# Download NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except:
    print("NLTK data download completed")

# Sentiment function
def analyze_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity == 0:
        return 'Neutral'
    else:
        return 'Negative'

# Step 1: Scrape Wikipedia
url = "https://en.wikipedia.org/wiki/Machine_learning"
response = requests.get(url)
soup = BeautifulSoup(response.content, 'lxml')
text = soup.getText(strip=True)

# Step 2: Clean text
text = re.sub(r'\[\d+\]', "", text)
text = re.sub(r'\[\w+\]', "", text)

# Step 3: Tokenize sentences
sentences = sent_tokenize(text)
df = pd.DataFrame(sentences, columns=['sentence'])

# Step 4: Sentiment analysis
df['sentiment'] = [analyze_sentiment(sentence) for sentence in df['sentence']]

# Step 5: Word processing
words = word_tokenize(text)
words = [word for word in words if word.isalnum()]
stop_words = set(stopwords.words('english'))
words = [word for word in words if word.lower() not in stop_words]

# Step 6: Word frequency
word_counts = Counter(words)

# Results
print("=== MACHINE LEARNING WIKIPEDIA ANALYSIS ===")
print(f"Total sentences: {len(sentences)}")
print(f"Total words: {len(words)}")
print(f"Unique words: {len(set(words))}")

print(f"\nSentiment Distribution:")
print(df['sentiment'].value_counts())

print(f"\nTop 10 words:")
for word, count in word_counts.most_common(10):
    print(f"{word}: {count}")

# Save results
df.to_csv('ml_analysis.csv', index=False)
print(f"\nResults saved to ml_analysis.csv")
