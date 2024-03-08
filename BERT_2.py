import os
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
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

# BERT Tokenizer ve Model'in yüklenmesi
tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-base-turkish-cased')
model = BertModel.from_pretrained('dbmdz/bert-base-turkish-cased')

# Metinleri BERT vektör temsillerine dönüştürme
def bert_encode(texts, tokenizer, model, max_length):
    input_ids = []
    attention_masks = []
    for text in tqdm(texts):
        encoded = tokenizer.encode_plus(text, max_length=max_length, truncation=True,
                                        pad_to_max_length=True, add_special_tokens=True,
                                        return_attention_mask=True, return_tensors='pt')
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    with torch.no_grad():
        last_hidden_states = model(input_ids, attention_mask=attention_masks)

    # Burada, BERT çıktısının son katmanındaki [CLS] token'inin temsillerini alıyoruz
    features = last_hidden_states[0][:, 0, :].numpy()
    return features


# Metinleri BERT vektör temsillerine dönüştürme ve veri seti hazırlama
X = bert_encode(texts, tokenizer, model, max_length=128)
y = np.array(labels)

# Sınıflandırıcıları tanımlama (random_state parametresi olanlar için sabit değer atama)
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