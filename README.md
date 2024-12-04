import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download required NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

# Input sentence
text = "Natural Language Processing is an exciting field of Artificial Intelligence."

# Tokenize the sentence
tokens = word_tokenize(text)

# Get the list of stop words in English
stop_words = set(stopwords.words('english'))

# Filter out stop words using two for loops
filtered_tokens = []
for word in tokens:
    if word.lower() not in stop_words:
        filtered_tokens.append(word)

# Perform POS tagging
pos_tags = nltk.pos_tag(filtered_tokens)

# Display the results
print("Original Tokens:", tokens)
print("Filtered Tokens (without stop words):", filtered_tokens)
print("POS Tags:", pos_tags)































import math
from sklearn.feature_extraction.text import CountVectorizer

# Input documents
documents = [
    "Natural Language Processing with Python is interesting.",
    "Python helps in analyzing and understanding human language.",
    "TF-IDF is a statistical measure for text analysis.",
]

# Step 1: Tokenization and TF Calculation
def compute_tf(doc):
    tf_dict = {}
    total_words = doc.split()
    total_count = len(total_words)
    for word in total_words:
        tf_dict[word] = tf_dict.get(word, 0) + 1
    # Normalize by total word count
    for word in tf_dict:
        tf_dict[word] = tf_dict[word] / total_count
    return tf_dict

# Step 2: IDF Calculation
def compute_idf(corpus):
    idf_dict = {}
    total_docs = len(corpus)
    for doc in corpus:
        for word in set(doc.split()):
            idf_dict[word] = idf_dict.get(word, 0) + 1
    for word in idf_dict:
        idf_dict[word] = math.log(total_docs / idf_dict[word]) + 1
    return idf_dict

# Step 3: TF-IDF Calculation
def compute_tfidf(tf_dict, idf_dict):
    tfidf_dict = {}
    for word, tf_value in tf_dict.items():
        tfidf_dict[word] = tf_value * idf_dict.get(word, 0)
    return tfidf_dict

# Preprocess: Convert documents to lowercase for consistency
documents = [doc.lower() for doc in documents]

# Calculate TF, IDF, and TF-IDF
tf_results = [compute_tf(doc) for doc in documents]
idf_result = compute_idf(documents)
tfidf_results = [compute_tfidf(tf, idf_result) for tf in tf_results]

# Display results
for i, tfidf in enumerate(tfidf_results):
    print(f"Document {i+1} TF-IDF:")
    for word, score in tfidf.items():
        print(f"  {word}: {score:.4f}")
    print()





























from collections import defaultdict, Counter
import math

class NGramLanguageModel:
    def __init__(self, n):
        self.n = n  # The size of the n-grams
        self.ngrams = defaultdict(Counter)  # Stores n-grams and their counts
        self.context_counts = Counter()  # Stores context counts for probability calculations

    def tokenize(self, text):
        # Tokenize text into words and add special start/end tokens
        tokens = text.lower().split()
        tokens = ['<s>'] * (self.n - 1) + tokens + ['</s>']
        return tokens

    def train(self, corpus):
        # Train the model with the given corpus
        for sentence in corpus:
            tokens = self.tokenize(sentence)
            for i in range(len(tokens) - self.n + 1):
                context = tuple(tokens[i:i + self.n - 1])
                word = tokens[i + self.n - 1]
                self.ngrams[context][word] += 1
                self.context_counts[context] += 1

    def calculate_probability(self, context, word):
        # Calculate the probability of a word given its context
        if context in self.ngrams:
            word_count = self.ngrams[context][word]
            context_count = self.context_counts[context]
            return word_count / context_count
        return 0.0

    def generate_sentence(self, max_words=20):
        # Generate a sentence using the language model
        context = ('<s>',) * (self.n - 1)
        sentence = list(context)

        for _ in range(max_words):
            if context not in self.ngrams:
                break
            word = max(self.ngrams[context], key=self.ngrams[context].get)  # Greedy selection
            if word == '</s>':
                break
            sentence.append(word)
            context = tuple(sentence[-(self.n - 1):])

        return ' '.join(sentence[(self.n - 1):])  # Exclude the start tokens

# Example usage                                                                                    
if __name__ == "__main__":
    corpus = [
        "I love natural language processing.",
        "Language models are a part of AI.",
        "I love building AI models.",
    ]

    ngram_model = NGramLanguageModel(n=2)  # Bigram model
    ngram_model.train(corpus)

    # Test the model
    context = ('i',)
    word = 'love'
    prob = ngram_model.calculate_probability(context, word)
    print(f"Probability of '{word}' given context {context}: {prob:.4f}")

    # Generate a sentence
    print("Generated Sentence:", ngram_model.generate_sentence())










from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk


nltk.download("punkt")

corpus = [
    "Natural language processing is a fascinating field.",
    "Word embeddings like Word2Vec are used in NLP.",
    "Deep learning models have revolutionized text analysis.",
]


tokenized_corpus = []
for sentence in corpus:
    sentence_lower = sentence.lower()
    tokenized_sentence = word_tokenize(sentence_lower)
    tokenized_corpus.append(tokenized_sentence)


word2vec_model = Word2Vec(sentences=tokenized_corpus, vector_size=100, window=5, min_count=1, workers=4)

word = "nlp"
if word in word2vec_model.wv:
    print(f"Word2Vec embedding for '{word}':\n{word2vec_model.wv[word]}")
else:
    print(f"Word '{word}' not in vocabulary.")


import gensim.downloader as api

# Download pre-trained GloVe embeddings
model = api.load("glove-wiki-gigaword-100")

# Get the vector representation of a word
word_vector = model['dog']
print(word_vector)


from gensim.models import FastText

# Sample text corpus (or load from a file)
sentences = [
    "This is a sentence about dogs.",
    "Dogs are furry friends.",
    "Cats are also cute animals."
]

# Create a FastText model (using vector_size instead of size)
model = FastText(sentences, min_count=1, vector_size=100, window=5)

# Get the vector representation of a word (optional)
word_vector = model.wv['dog']
print(word_vector)
