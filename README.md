# Sentiment Analysis with DistilBERT on SST-2

This notebook demonstrates how to perform sentiment analysis using a pre-trained DistilBERT model on the SST-2 (Stanford Sentiment Treebank) dataset. The process involves loading the dataset, tokenizing text, fine-tuning DistilBERT for sequence classification, evaluating its performance, and making custom predictions.

## Table of Contents
1.  [Installation](#1-installation)
2.  [Imports](#2-imports)
3.  [Load Dataset (SST-2)](#3-load-dataset-sst-2)
4.  [Tokenizer + Model](#4-tokenizer--model)
5.  [Tokenization](#5-tokenization)
6.  [Training Configuration](#6-training-configuration)
7.  [Trainer](#7-trainer)
8.  [Train Model](#8-train-model)
9.  [Evaluate](#9-evaluate)
10. [Confusion Matrix](#10-confusion-matrix)
11. [Custom Prediction](#11-custom-prediction)

## 1. Installation
This section ensures all necessary libraries are installed, including `transformers`, `datasets`, `scikit-learn`, and `torch`.

```python
!pip install transformers datasets scikit-learn torch
```

## 2. Imports
Key libraries for data handling, model loading, training, and evaluation are imported here.

## 3. Load Dataset (SST-2)
The SST-2 dataset, part of the GLUE benchmark, is loaded. This dataset is designed for binary sentiment classification.

```python
dataset = load_dataset("glue", "sst2")
```

## 4. Tokenizer + Model
A `DistilBertTokenizer` and `DistilBertForSequenceClassification` model are loaded from Hugging Face's pre-trained models. The model is configured for 2 output labels (positive/negative sentiment).

## 5. Tokenization
The `tokenize_function` prepares the text data by converting sentences into numerical input IDs, attention masks, and truncation/padding to a fixed length (128).

```python
def tokenize_function(example):
    return tokenizer(
        example["sentence"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
```

## 6. Training Configuration
`TrainingArguments` defines parameters for the training process, such as output directory, evaluation strategy, learning rate, batch sizes, number of epochs, and logging settings.

## 7. Trainer
The `Trainer` class from `transformers` is used to orchestrate the training and evaluation loop. It takes the model, training arguments, datasets, and a `compute_metrics` function.

## 8. Train Model
The model is fine-tuned on the SST-2 training data.

## 9. Evaluate
After training, the model's performance is evaluated on the validation set using accuracy and F1-score.

## 10. Confusion Matrix
A confusion matrix is generated to provide a detailed view of the model's classification performance, showing true positives, true negatives, false positives, and false negatives.

## 11. Custom Prediction
A `predict_sentiment` function allows you to test the fine-tuned model with your own custom text inputs, returning the probabilities for negative and positive sentiment.

```python
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return {"negative": float(probs[0][0].cpu()), "positive": float(probs[0][1].cpu())}

# Interactive prediction loop
while True:
    text = input("Enter a sentence: ")
    result = predict_sentiment(text)
    print(f"Sentiment: {result}")

    answer = input("Do you want to predict another sentence? (yes/no): ").lower()
    if answer != 'yes':
        break
```
