import numpy as np
from transformers import BertTokenizer, BertModel
import torch
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
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

# BERT Tokenizer ve Model'in yüklenmesi
tokenizer = BertTokenizer.from_pretrained('"dbmdz/bert-base-turkish-cased')
model = BertModel.from_pretrained('"dbmdz/bert-base-turkish-cased')

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
    features = last_hidden_states[0][:,0,:].numpy()
    return features

# Metinlerin BERT temsillerini hesaplama (Bu adım biraz zaman alabilir)
X = bert_encode(texts, tokenizer, model, max_length=128)
y = np.array(labels)

# Eğitim ve test setlerine bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Basit bir sınıflandırıcı ile eğitim ve test
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Doğruluk skorunun hesaplanması
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
