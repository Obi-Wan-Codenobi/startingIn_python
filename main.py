#using skylearn and nltk

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Example sentences
sentence1 = "I love science so much!"
sentence2 = "I dont understand science at all."
sentence3 = "I really like science."
sentence4 = "I like science"
sentence5 = "I hate science!"
sentence7 = "I like potatoes."
sentence8 = "I really love fruit"
sentence9 = "fruit and potatoes are awful?"
sentence10 = "I eat apples."

# Combine sentences into a list
sentences = [sentence1, sentence2, sentence3, sentence4, sentence5, sentence7, sentence8, sentence9, sentence10]

# Text preprocessing
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum()]
    tokens = [token for token in tokens if token not in stop_words]
    return " ".join(tokens)

# Preprocess sentences
preprocessed_sentences = [preprocess_text(sentence) for sentence in sentences]

# Vectorize sentences using TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(preprocessed_sentences)

# Compute cosine similarity
similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Group similar sentences
threshold = 0.35  # Adjust threshold to determine similarity cutoff
groups = []
used_indices = set()

for i in range(len(sentences)):
    if i not in used_indices:
        group = [sentences[i]]
        used_indices.add(i)
        for j in range(i+1, len(sentences)):
            if j not in used_indices and similarity_matrix[i, j] > threshold:
                group.append(sentences[j])
                used_indices.add(j)
        groups.append(group)

# Print groups
for i, group in enumerate(groups):
    print(f"Group {i+1}:")
    print(group)
    print("---")


