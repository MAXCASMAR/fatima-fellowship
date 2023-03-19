import pandas as pd
import numpy as np
import re
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

class FakeNewsDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        encoding['label'] = torch.tensor(label)
        return encoding

class FakeNewsClassifier:
    def __init__(self, true_csv, fake_csv, model_name):
        self.true_csv = true_csv
        self.fake_csv = fake_csv
        self.model_name = model_name
        self.tokenizer, self.model = self.init_model()

    @staticmethod
    def preprocess_text(text):
        text = re.sub(r'http\S+', '', text) # Remove URLs
        return text

    def load_and_preprocess_data(self):
        true_df = pd.read_csv(self.true_csv)
        fake_df = pd.read_csv(self.fake_csv)

        true_df['label'] = 1
        fake_df['label'] = 0

        df = pd.concat([true_df, fake_df], ignore_index=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True) # Shuffle data
        df['text'] = df['text'].apply(self.preprocess_text)

        return df

    def split_data(self, df, test_size=0.2):
        X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=test_size, random_state=42, stratify=df['label'])
        return X_train, X_test, y_train, y_test

    def init_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        return tokenizer, model

    def train_model(self, X_train, y_train):
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            per_device_train_batch_size=16,
            logging_steps=100,
            save_strategy='no',
            evaluation_strategy='no',
            load_best_model_at_end=False,
            metric_for_best_model=None,
            seed=42,
        )

        train_dataset = FakeNewsDataset(X_train, y_train, self.tokenizer, max_length=512)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
        )

        trainer.train()

    def evaluate_model(self, X_test, y_test):
        test_dataset = FakeNewsDataset(X_test, y_test, self.tokenizer, max_length=512)
        logits = self.model(test_dataset['input_ids'], attention_mask=test_dataset['attention_mask']).logits
        preds = torch.sigmoid(logits).numpy()

        auc = roc_auc_score(y_test, preds)
        return auc

    def upload_model(self):
        self.model.push_to_hub('fake-news-classifier')

    def run(self):
        df = self.load_and_preprocess_data()
        X_train, X_test, y_train, y_test = self.split_data(df)
        self.train_model(X_train, y_train)
        auc = self.evaluate_model(X_test, y_test)
        print(f'AUC: {auc}')
        self.upload_model()

if __name__ == '__main__':
    true_csv = '/workspaces/fatima-fellowship/src/nlp/data/True.csv'
    fake_csv = '/workspaces/fatima-fellowship/src/nlp/data/Fake.csv'
    pretrained_model_name = 'roberta-base'
    pretrained_model_name = "ghanashyamvtatti/roberta-fake-news"

    classifier = FakeNewsClassifier(true_csv, fake_csv, pretrained_model_name)
    print("perro")
    classifier.run()


