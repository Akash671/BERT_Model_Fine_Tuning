
# !pip install nltk

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download('punkt_tab')

def clean_text(text):
    # Lowercase
    text = text.lower()

    # Remove HTML tags or markup
    text = re.sub(r"<.*?>", "", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove emojis, symbols, and non-ASCII
    text = re.sub(r"[^\x00-\x7F]+", " ", text)

    # Remove hashtags, mentions
    text = re.sub(r"[@#]\w+", "", text)

    # Remove repeated punctuations & weird characters
    text = re.sub(r"[!]{2,}|[?]{2,}|[#]{2,}|[\-]{2,}", " ", text)

    # Remove special characters but keep words and basic punctuation
    text = re.sub(r"[^\w\s.,!?]", "", text)

    # Tokenize
    #tokens = word_tokenize(text)

    # Remove stopwords
    #stop_words = set(stopwords.words("english"))
    #tokens = [word for word in tokens if word not in stop_words]

    # Lemmatize
    #lemmatizer = WordNetLemmatizer()
    #tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return text