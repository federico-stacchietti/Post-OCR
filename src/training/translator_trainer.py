from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
import pandas as pd
from datasets import Dataset

path = 'altered_dataset_04-15k.csv'
data = pd.read_csv(path)

dataset = data.drop(columns=['labels'], axis=1)
dataset.columns = ['texts', 'altered']

train_dataset = dataset

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-itc-itc")

model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-itc-itc")

train_dataset = pd.DataFrame(data=train_dataset, columns=['texts', 'altered'])

train_dataset = Dataset.from_pandas(train_dataset)

padding = "max_length"


def preprocess_function(examples):
    inputs = [ex for ex in examples['altered']]
    targets = [ex for ex in examples['texts']]
    model_inputs = tokenizer(inputs, max_length=512, padding='max_length', truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=512, padding='max_length', truncation=True)

    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


train_dataset = train_dataset.map(preprocess_function, batched=True, desc="Running tokenizer on train dataset")

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

training_args = Seq2SeqTrainingArguments(
    f"translation",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    weight_decay=0.01,
    save_total_limit=1,
    num_train_epochs=1,
    fp16=True
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model('translator-04')

