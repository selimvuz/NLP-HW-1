import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

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

# Bag of Words vektör temsilleri
vectorizer = CountVectorizer(max_features=10000)
X = vectorizer.fit_transform(texts)
y = labels

# Sınıflandırıcıları tanımlama ve random_state ayarlama
classifiers = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'SVM': SVC(),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'IBk (k-NN)': KNeighborsClassifier(n_neighbors=3)
}

# Her bir sınıflandırıcı için eğitim ve test
for name, clf in classifiers.items():
    cv_scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    print(f"{name} Average Accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
