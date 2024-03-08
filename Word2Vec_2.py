import os
import numpy as np
from gensim.models import Word2Vec
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale

# Veri setini etiketleme ve yükleme fonksiyonu
def load_data(directory):
    texts = []
    labels = []
    label_dict = {}
    current_label = 0

    # Yazar adlarına göre klasörleri gez
    for author in os.listdir(directory):
        author_path = os.path.join(directory, author)
        if os.path.isdir(author_path):
            # Her bir yazar için tüm dosyaları oku
            for filename in os.listdir(author_path):
                if filename.endswith('.txt'):
                    filepath = os.path.join(author_path, filename)
                    with open(filepath, 'r', encoding='ISO-8859-1') as file:
                        texts.append(file.read())
                        labels.append(current_label)
            label_dict[author] = current_label
            current_label += 1

    return texts, np.array(labels), label_dict

# Metinleri kelime listelerine dönüştürme
def text_to_wordlist(texts):
    return [text.split() for text in texts]

# Word2Vec modelini eğitme ve ortalama vektör temsili hesaplama
def build_word2vec_features(texts):
    word_lists = text_to_wordlist(texts)
    model = Word2Vec(sentences=word_lists, vector_size=100, window=5, min_count=1, workers=4)
    word_vectors = model.wv
    features = np.array([np.mean([word_vectors[w] for w in words if w in word_vectors] 
                                  or [np.zeros(model.vector_size)], axis=0)
                         for words in word_lists])
    return features

texts, labels, label_dict = load_data('datasets/270koseyazisi')
X = build_word2vec_features(texts)
y = np.array(labels)

# X verilerini ölçeklendirme
X = scale(X)

# Sınıflandırıcıları tanımlama ve random_state ayarlama
classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'SVM': SVC(),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'IBk (k-NN)': KNeighborsClassifier(n_neighbors=3)
}

# Sınıflandırıcıları karşılaştırma ve 5 fold çapraz doğrulama uygulama
for name, clf in classifiers.items():
    scores = cross_val_score(clf, X, y, cv=5)
    print(f"{name} Average Accuracy: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
