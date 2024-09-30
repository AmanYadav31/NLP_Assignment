import nltk
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('sentiwordnet')
nltk.download('wordnet')

# Load the data with fallback encoding to handle potential issues
def load_dataset(pos_file, neg_file):
    with open(pos_file, 'r', encoding='ISO-8859-1') as file:
        pos_texts = file.readlines()
    with open(neg_file, 'r', encoding='ISO-8859-1') as file:
        neg_texts = file.readlines()
    return pos_texts, neg_texts

# Helper function to map NLTK POS tags to WordNet POS tags
def map_pos_to_wordnet(nltk_pos_tag):
    if nltk_pos_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_pos_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_pos_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


# Function to compute SentiWordNet sentiment scores for a word
def calculate_sentiwordnet_score(token, pos):
    synsets = list(swn.senti_synsets(token, pos))
    if synsets:
        avg_pos_score = sum([syn.pos_score() for syn in synsets]) / len(synsets)
        avg_neg_score = sum([syn.neg_score() for syn in synsets]) / len(synsets)
        return avg_pos_score, avg_neg_score
    return 0.0, 0.0


# Function to assign sentiment-based weights to words in a document
def assign_sentiment_weights(doc):
    tokens = nltk.word_tokenize(doc)
    pos_tags = nltk.pos_tag(tokens)
    word_sentiment_weights = {}

    for word, pos in pos_tags:
        wn_pos = map_pos_to_wordnet(pos)
        if wn_pos:
            pos_score, neg_score = calculate_sentiwordnet_score(word, wn_pos)
            sentiment_strength = pos_score - neg_score  
            word_sentiment_weights[word] = 1 + sentiment_strength

    return word_sentiment_weights


# Custom TF-IDF vectorizer that incorporates sentiment-based weights
class SentimentBasedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(SentimentBasedTfidfVectorizer, self).build_analyzer()
        return lambda doc: [(token, self.sentiment_weights.get(token, 1)) for token in analyzer(doc)]

    def fit_transform(self, docs, y=None):
        self.sentiment_weights = {}
        for doc in docs:
            self.sentiment_weights.update(assign_sentiment_weights(doc))
        return super(SentimentBasedTfidfVectorizer, self).fit_transform(docs, y)

    def transform(self, docs):
        return super(SentimentBasedTfidfVectorizer, self).transform(docs)


# Load the dataset
pos_text_file = r'C:\Users\AMAN YADAV\Downloads\NLP Assignment\rt-polarity.pos'
neg_text_file = r'C:\Users\AMAN YADAV\Downloads\NLP Assignment\rt-polarity.neg'
positive_texts, negative_texts = load_dataset(pos_text_file, neg_text_file)
print(f"Loaded {len(positive_texts)} positive and {len(negative_texts)} negative texts.")

# Preprocess the texts
train_docs = positive_texts[:4000] + negative_texts[:4000]
train_labels = [1] * 4000 + [0] * 4000

val_docs = positive_texts[4000:4500] + negative_texts[4000:4500]
val_labels = [1] * 500 + [0] * 500

test_docs = positive_texts[4500:5331] + negative_texts[4500:5331]
test_labels = [1] * 831 + [0] * 831

vectorizer = SentimentBasedTfidfVectorizer(ngram_range=(1, 2))
X_train = vectorizer.fit_transform(train_docs)
X_val = vectorizer.transform(val_docs)
X_test = vectorizer.transform(test_docs)

# Classifiers to train
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Support Vector Machine (SVM)": SVC(),
    "Random Forest": RandomForestClassifier(n_estimators=100)
}

# Train and evaluate the classifiers
for model_name, model in models.items():
    print(f"Training {model_name}...")

    model.fit(X_train, train_labels)

    val_predictions = model.predict(X_val)

    print(f"Validation results for {model_name}:")
    print(f"Accuracy: {accuracy_score(val_labels, val_predictions):.4f}")
    print("Classification Report:")
    print(classification_report(val_labels, val_predictions))
    print("*" * 80)


best_model = models["Naive Bayes"]  
test_predictions = best_model.predict(X_test)

print("Final Evaluation on Test Set:")
print(f"Accuracy: {accuracy_score(test_labels, test_predictions):.4f}")
print("Classification Report:")
print(classification_report(test_labels, test_predictions))

conf_matrix = confusion_matrix(test_labels, test_predictions)
print("Confusion Matrix:")
print(conf_matrix)

tn, fp, fn, tp = conf_matrix.ravel()
print(f"True Positives (TP): {tp}")
print(f"True Negatives (TN): {tn}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
