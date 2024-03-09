import os
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Veri setini etiketleme fonksiyonu
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

# Veri setini yeni dizinle yükleme
texts, labels, label_dict = load_data('datasets/270koseyazisi')

# GloVe vektörlerini okuma
def load_glove_model(glove_file):
    print("GloVe modeli yükleniyor")
    model = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            try:
                embedding = np.array([float(val) for val in split_line[1:]])
                model[word] = embedding
            except ValueError:
                print(f"Uyarı: Float olmayan değere sahip şu kelime atlandı: '{word}'")
    print(f"{len(model)} adet kelime yüklendi!")
    return model


# GloVe dosya yolu
glove_path = 'GloVe-Vectors/glove.840B.300d.txt'
glove_model = load_glove_model(glove_path)

# Metinleri vektör temsillerine dönüştürme
def text_to_glove_vector(texts, glove_model):
    # Örnek bir kelimenin vektör boyutunu al
    vector_size = glove_model[next(iter(glove_model))].shape[0]
    features = np.zeros((len(texts), vector_size), dtype=np.float32)

    for i, text in enumerate(texts):
        words = text.split()
        word_vectors = np.array([glove_model[word]
                                for word in words if word in glove_model])

        if len(word_vectors) > 0:
            features[i] = np.mean(word_vectors, axis=0)

    return features

X = text_to_glove_vector(texts, glove_model)
y = np.array(labels)

# Sınıflandırıcıları tanımlama (random_state parametresi için sabit değer atama)
classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'SVM': SVC(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'IBk (k-NN)': KNeighborsClassifier(n_neighbors=3)
}

# 5 fold çapraz doğrulama uygulama
for name, clf in classifiers.items():
    scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    print(f"{name} Average Accuracy: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
