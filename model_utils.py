import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import pickle

# Cihaz seçimi
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tokenizer ve modeli yükleme
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert_specialty_model')
model.to(device)

# Etiket kodlayıcısını yükleme (önceden kaydedilmiş olmalı!)
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Tahmin fonksiyonu
def predict_specialty(complaint):
    encoding = tokenizer(complaint, return_tensors='pt', padding=True, truncation=True, max_length=128)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    predicted_label = torch.argmax(logits, dim=1).item()
    predicted_specialty = label_encoder.inverse_transform([predicted_label])[0]
    return predicted_specialty
