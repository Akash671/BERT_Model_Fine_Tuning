# -*- coding: utf-8 -*-
"""BERT_Model_FineTuning.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1N8SaRqnxzdxL-VdoBo86ca-pgP42FnB9
"""

import json
from datasets import Dataset, DatasetDict # Using Hugging Face's datasets library
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer
import torch
import os

# --- 1. Load the dataset ---
def load_qa_dataset(file_path="fake_qa_dataset.json"):
    """
    Loads the fake Q&A dataset from a JSON file.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Successfully loaded dataset from {file_path}. Found {len(data)} samples.")
        return data
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {file_path}. Please run the data generation script first.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}. Check file format.")
        return None

# Load your dataset
raw_datasets = load_qa_dataset()

if raw_datasets is None:
    exit() # Exit if dataset loading failed

# Convert the list of dictionaries to a Hugging Face Dataset object
# For simplicity, we'll put all data into the 'train' split for this example.
# In a real scenario, you'd split into train, validation, and test.
hf_dataset = Dataset.from_list(raw_datasets)
# Create a DatasetDict if you want to define splits (e.g., train, validation)
# For this example, let's create a small train/test split (80/20)
train_test_split = hf_dataset.train_test_split(test_size=0.2, seed=42)
dataset_dict = DatasetDict({
    'train': train_test_split['train'],
    'validation': train_test_split['test'] # Using test set as validation for this example
})
print(f"Dataset split into: {dataset_dict}")


# --- 2. Load pre-trained BERT model and tokenizer ---
# You can choose different BERT-like models, e.g., 'bert-base-uncased', 'distilbert-base-uncased'
model_checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

print(f"Loaded tokenizer and model: {model_checkpoint}")

# --- 3. Preprocess the dataset ---
# This is the most crucial part for Q&A fine-tuning.
# We need to map answers to token spans.

max_length = 384  # The maximum length of a feature (context and question)
doc_stride = 128   # The authorized overlap between two consecutive chunks

def preprocess_training_examples(examples):
    """
    Preprocesses the training examples for BERT Q&A.
    This involves tokenization, handling long contexts, and finding answer spans.
    """
    questions = [q.strip() for q in examples["question"]]
    contexts = examples["context"]
    answers = examples["answer"]

    # Tokenize contexts and questions together.
    # `truncation="only_second"` truncates the context if the combined length exceeds max_length.
    # `return_offsets_mapping` is crucial for mapping token spans back to original text.
    # `padding="max_length"` pads to max_length.
    # `stride` handles overlapping chunks for long contexts.
    tokenized_examples = tokenizer(
        questions,
        contexts,
        max_length=max_length,
        truncation="only_second",
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
        stride=doc_stride,
    )

    # Since one example can give us several features if it has a long context,
    # we need to ensure that each feature has the correct `example_id`
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # The offsets mapping will give us a tuple of (start_char, end_char) for each token.
    offset_mapping = tokenized_examples.pop("offset_mapping")

    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # We find the example that this feature came from
        sample_idx = sample_mapping[i]
        answer = answers[sample_idx]
        context = contexts[sample_idx]

        # Start and end character positions of the answer in the original context
        start_char = answer["answer_start"]
        end_char = start_char + len(answer["text"])

        # Sequence ID tells us if a token belongs to the question (0) or context (1)
        sequence_ids = tokenized_examples.sequence_ids(i)

        # Find the start and end of the context in the tokenized input
        # This is where the context begins after the [CLS] and question tokens
        idx = 0
        while sequence_ids[idx] != 1: # Find first token of context (sequence_id = 1)
            idx += 1
        context_start_token = idx

        # Find the end of the context
        idx = len(sequence_ids) - 1
        while sequence_ids[idx] != 1: # Find last token of context (sequence_id = 1)
            idx -= 1
        context_end_token = idx

        # If the answer is not fully contained in this chunk, set positions to 0 (CLS token)
        # This is a common practice when the answer is not found or spans across chunks.
        # The model will learn to predict the [CLS] token in such cases, indicating no answer.
        if (offsets[context_start_token][0] > start_char or
            offsets[context_end_token][1] < end_char):
            tokenized_examples["start_positions"].append(0)
            tokenized_examples["end_positions"].append(0)
        else:
            # Otherwise, find the start and end token positions
            # Iterate over tokens and check if their character offsets overlap with the answer.
            token_start_index = context_start_token
            while token_start_index <= context_end_token and offsets[token_start_index][0] <= start_char:
                token_start_index += 1
            tokenized_examples["start_positions"].append(token_start_index - 1)

            token_end_index = context_end_token
            while token_end_index >= context_start_token and offsets[token_end_index][1] >= end_char:
                token_end_index -= 1
            tokenized_examples["end_positions"].append(token_end_index + 1)
    return tokenized_examples

# Apply preprocessing to the dataset
print("Preprocessing training examples...")
tokenized_datasets = dataset_dict.map(
    preprocess_training_examples,
    batched=True,
    remove_columns=dataset_dict["train"].column_names # Remove original columns not needed for training
)
print("Preprocessing complete.")

# --- 4. Define training arguments ---
# Define output directory for saving checkpoints
output_dir = "./results"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy="epoch", # CORRECTED: Use eval_strategy instead of evaluation_strategy
    learning_rate=2e-5,
    per_device_train_batch_size=8, # Adjust based on your GPU memory
    per_device_eval_batch_size=8,
    num_train_epochs=3, # Number of epochs to train for
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch", # Save model every epoch
    load_best_model_at_end=True, # Load the best model at the end of training
    metric_for_best_model="eval_loss", # Metric to use for determining the best model
    push_to_hub=False, # Set to True if you want to push to Hugging Face Hub (requires login)
    do_train=True, # Explicitly enable training
    do_eval=True,  # Explicitly enable evaluation
)

# --- 5. Initialize Trainer ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer, # Pass tokenizer to Trainer for proper saving
)

# --- 6. Train the model ---
print("Starting model training...")
trainer.train()
print("Training complete!")

# Save the fine-tuned model and tokenizer
model_save_path = "./fine_tuned_bert_qa"
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

trainer.save_model(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f"Fine-tuned model and tokenizer saved to {model_save_path}")

print("\nYou can now load this model for inference:")
print(f"""
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
tokenizer = AutoTokenizer.from_pretrained("{model_save_path}")
model = AutoModelForQuestionAnswering.from_pretrained("{model_save_path}")
""")



from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import torch
import os

# Define the path where your fine-tuned model and tokenizer are saved
model_path = "./fine_tuned_bert_qa"

# --- 1. Load the fine-tuned model and tokenizer ---
print(f"Loading fine-tuned model and tokenizer from: {model_path}")
if not os.path.exists(model_path):
    print(f"Error: Model directory not found at {model_path}. "
          "Please ensure the fine-tuning script completed successfully "
          "and saved the model to this location.")
    exit()

try:
    # We use Auto classes to ensure compatibility with the saved model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    print("Model and tokenizer loaded successfully!")
except Exception as e:
    print(f"Failed to load model or tokenizer. Error: {e}")
    print("This might happen if the model was not saved correctly or if there's a version mismatch.")
    exit()

# --- 2. Create a Question-Answering pipeline ---
# The pipeline handles tokenization, model inference, and post-processing (extracting the answer span)
# You can specify the device if you have a GPU (e.g., device=0 for the first GPU)
qa_pipeline = pipeline(
    "question-answering",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1 # Use GPU if available, else CPU
)

print(f"QA pipeline initialized. Using device: {'GPU' if torch.cuda.is_available() else 'CPU'}")

# --- 3. Define a function for prediction ---
def get_answer(question: str, context: str):
    """
    Uses the fine-tuned BERT model to get an answer from a given context and question.
    """
    try:
        # The pipeline returns a dictionary with 'score', 'start', 'end', and 'answer'
        result = qa_pipeline(question=question, context=context)
        return result['answer']
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return "Could not find an answer."

# --- 4. Test with example questions and contexts ---

print("\n--- Running Predictions ---")

# Example 1: Based on one of your generated contexts
context1 = "The capital of France is Paris. Paris is also known as the City of Lights. It is famous for the Eiffel Tower."
question1 = "What is the capital of France?"
print(f"\nContext: {context1}")
print(f"Question: {question1}")
print(f"Predicted Answer: {get_answer(question1, context1)}")
print("-" * 30)

# Example 2: Another context from your generated data
context2 = "Mount Everest is the highest mountain in the world, located in the Himalayas. Its peak is 8,848.86 meters above sea level."
question2 = "Where is Mount Everest located?"
print(f"\nContext: {context2}")
print(f"Question: {question2}")
print(f"Predicted Answer: {get_answer(question2, context2)}")
print("-" * 30)

# Example 3: A slightly different question to test generalization
context3 = "Photosynthesis is the process used by plants, algae and cyanobacteria to convert light energy into chemical energy."
question3 = "What do plants do with light energy?"
print(f"\nContext: {context3}")
print(f"Question: {question3}")
print(f"Predicted Answer: {get_answer(question3, context3)}")
print("-" * 30)

# Example 4: A question for which the answer might not be directly in the context
# (the model might return a less specific answer or an empty string depending on training)
context4 = "Artificial intelligence (AI) is intelligence demonstrated by machines, unlike the natural intelligence displayed by humans and animals."
question4 = "What is the future of AI?" # This specific answer is not directly in the context
print(f"\nContext: {context4}")
print(f"Question: {question4}")
print(f"Predicted Answer: {get_answer(question4, context4)}")
print("-" * 30)




print("\nPrediction process complete.")

context4 = "Sam Altman is the CEO of OpenAI and he is also co-founder of OpenAI"
question4 = "who is the ceo of OpenAI?" # This specific answer is not directly in the context
print(f"\nContext: {context4}")
print(f"Question: {question4}")
print(f"Predicted Answer: {get_answer(question4, context4)}")
print("-" * 30)







