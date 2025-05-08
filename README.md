Installation Guide

Step 1: Set Up Your Google Colab Environment

Go to Hugging Face and create a write token
Go to wandb.ai and create a account
Open Google Colab: Go to Google Colab and create a new notebook.
In the first cell of your notebook, install the required libraries by running the following:

!pip install transformers peft datasets accelerate huggingface_hub

This will install the transformers, peft (Low-Rank Adapters for efficient fine-tuning), datasets (to handle datasets), and huggingface_hub (for Hugging Face interaction) libraries.

Step 2: Step 2: Authenticate with Hugging Face

Set the Hugging Face Token in Colab:

from huggingface_hub import login
login(token="YOUR_HUGGINGFACE_TOKEN")

This will authenticate your session and allow you to access models and datasets from Hugging Face.
Click 'Secrets' on the lefthand side. Add the Hugging Face token as a new secret.

Step 3: Load the Pre-trained Model and Tokenizer
We'll load GPT-2 as the base model for our chatbot.

from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the GPT-2 model and tokenizer
model = AutoModelForCausalLM.from_pretrained('gpt2')
tokenizer = AutoTokenizer.from_pretrained('gpt2')

# Set pad token id to eos_token_id
tokenizer.pad_token = tokenizer.eos_token

Step 3: Prepare Your Dataset
Now, we'll prepare the pub_faqs.txt

# Read the uploaded .txt file
data = []

with open('/content/pub_faqs.txt', 'r') as file:
    lines = file.readlines()
    for i in range(0, len(lines) - 1, 2):  # Ensure we're not going out of range
        question = lines[i].strip()  # Clean up any extra whitespace or newline characters
        answer = lines[i + 1].strip()  # Similarly clean the answer
        data.append({"question": question, "answer": answer})

# Verify the first few items
print(data[:5])  # Print the first 5 Q&A pairs

Step 5: Create the Dataset 

Now, we can create a Hugging Face Dataset from the data list:

from datasets import Dataset

# Create the dataset from the 'data' list
dataset = Dataset.from_list(data)

Step 6: Tokenize the Dataset
We need to tokenize the dataset for training. We will use padding and truncation to ensure consistency across input sequences.

# Tokenization function with padding and truncation
def tokenize_function(examples):
    return tokenizer(
        examples['question'], 
        examples['answer'], 
        padding="max_length",  # Pad sequences to a fixed length
        truncation=True,       # Truncate sequences that are too long
        max_length=512         # Set a maximum sequence length
    )

# Apply tokenization
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Verify tokenized dataset
print(tokenized_dataset[0])  # Check the first tokenized example

