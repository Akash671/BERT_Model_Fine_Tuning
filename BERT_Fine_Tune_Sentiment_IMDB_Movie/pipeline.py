"""
author : @akash
"""

from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import data_cleaning_pipeline
import pandas as pd
import numpy as np


#--------------------------------------------------------------+
data_file_path=input("Enter file path  : ")
df=pd.read_csv(data_file_path)


#--------------------------------------------------------------+
# do cleaning on df if required.


df["clean_review"] = df["review"].apply(data_cleaning_pipeline.clean_text)
df.drop('review', axis=1,inplace=True)

print(df.head())


#--------------------------------------------------------------
dataset = Dataset.from_pandas(df[['clean_review', 'sentiment']]) 
dataset = dataset.rename_column("clean_review", "text") # text -> this is the review description
dataset = dataset.rename_column("sentiment", "labels")
dataset = dataset.class_encode_column("labels")  # label → class ids
dataset = dataset.train_test_split(test_size=0.2)

print(dataset)

model_name = "bert-base-uncased"
tokenizer = BertTokenizerFast.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)


def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

tokenized = dataset.map(tokenize, batched=True)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


training_args = TrainingArguments(
    output_dir="bert-sentiment-model",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
)



trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()


trainer.evaluate()


sample_text = "total waste of time even popcorn was better"
inputs = tokenizer(sample_text, return_tensors="pt", truncation=True, padding=True)
outputs = model(**inputs)
logits = outputs.logits
pred = logits.argmax().item()

print("Predicted Sentiment:", "Positive" if pred == 1 else "Negative")

