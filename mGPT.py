import os
from transformers import GPT2Tokenizer, GPT2Model
import torch
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm

# Veri setini etiketleme fonksiyonu
def load_data(directory):
    texts = []
    labels = []
    label_dict = {'negatif': 0, 'pozitif': 1, 'tarafsiz': 2}

    for label in label_dict:
        dir_path = os.path.join(directory, label)
        for filename in os.listdir(dir_path):
            if filename.endswith('.txt'):
                with open(os.path.join(dir_path, filename), 'r', encoding='ISO-8859-1') as file:
                    texts.append(file.read())
                    labels.append(label_dict[label])
    return texts, labels

# Veri setini ve etiketlerini yükleme
texts, labels = load_data('datasets/film_yorumlari')

# GPT-2 tokenizer ve modelin yüklenmesi
tokenizer = GPT2Tokenizer.from_pretrained('ai-forever/mGPT')
model = GPT2Model.from_pretrained('ai-forever/mGPT')

# Metinleri GPT-2 vektör temsillerine dönüştürme
def gpt2_encode(texts, tokenizer, model, max_length):
    features = []
    for text in tqdm(texts):
        encoded_input = tokenizer.encode(text, max_length=max_length, truncation=True, padding='max_length', return_tensors='pt')
        with torch.no_grad():
            output = model(encoded_input)
        # Son katmandaki tüm tokenlerin ortalamasını alıyoruz
        feature = output.last_hidden_state.mean(dim=1).squeeze().numpy()
        features.append(feature)
    return np.array(features)

# Metinleri GPT-2 vektör temsillerine dönüştürme ve veri seti hazırlama
X = gpt2_encode(texts, tokenizer, model, max_length=128)
y = np.array(labels)

# Sınıflandırıcıları tanımlama ve 5 fold çapraz doğrulama uygulama
classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'SVM': SVC(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'IBk (k-NN)': KNeighborsClassifier(n_neighbors=3)
}

for name, clf in classifiers.items():
    scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    print(f"{name} Average Accuracy: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
