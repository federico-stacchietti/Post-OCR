import os
import pandas as pd
import numpy as np
from datasets import Dataset
import evaluate
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
import re

id2label = {
    0: "O",
    1: "I-MISP",
    2: "I-SEG",
    3: "I-UN",
}
label2id = {
    "O": 0,
    "I-MISP": 1,
    "I-SEG": 2,
    "I-UN": 3,
}

label_list = list(label2id.keys())

model_name = 'bert-base-multilingual-cased'

data = pd.read_csv('altered_dataset_04-15k.csv')

data['labels'] = data['labels'].map(lambda l: [int(match.group()) for match in re.finditer(r'\d', l)])
data['altered'] = data['altered'].map(lambda l: [match.group() for match in re.finditer(r'\S+', l)])

dataset = data.drop(columns=['texts'], axis=1)
dataset.columns = ['altered', 'labels']

train_dataset, test_dataset = train_test_split(dataset, test_size=.2, random_state=42)

task = "ner"

batch_size = 32
tokenizer = BertTokenizerFast.from_pretrained(model_name)


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples['altered'], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples['labels']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs['labels'] = labels
    return tokenized_inputs


train_dataset = Dataset.from_pandas(train_dataset)
test_dataset = Dataset.from_pandas(test_dataset)
print(train_dataset)
train_tokenized_datasets = train_dataset.map(tokenize_and_align_labels, batched=True)
test_tokenized_datasets = test_dataset.map(tokenize_and_align_labels, batched=True)

model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=4, id2label=id2label, label2id=label2id)

args = TrainingArguments(
    f"test-{task}",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    save_strategy="no",
    num_train_epochs=1,
    evaluation_strategy="epoch",
    weight_decay=0.01,
    fp16=True
)

data_collator = DataCollatorForTokenClassification(tokenizer)
metric = evaluate.load("seqeval")


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [[label_list[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in
                        zip(predictions, labels)]
    true_labels = [[label_list[l] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in
                   zip(predictions, labels)]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {"precision": results["overall_precision"], "recall": results["overall_recall"], "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"]}


trainer = Trainer(
    model,
    args,
    train_dataset=train_tokenized_datasets,
    eval_dataset=test_tokenized_datasets,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.save_model('tagger-04-1ep')
