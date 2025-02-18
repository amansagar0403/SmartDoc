import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertForSequenceClassification, AdamW, DistilBertTokenizer
from sklearn.metrics import classification_report
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK resources for stopwords and lemmatization if not already downloaded
nltk.download("stopwords")
nltk.download("wordnet")

# Check for GPU availability; if not found, default to using the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Step 1: Load and preprocess data using NLP techniques
class TextPreprocessor:
    def __init__(self):
        # Initialize stopwords and a lemmatizer
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

    def preprocess(self, text):
        """
        Preprocess the given text:
        - Tokenizes the text by splitting into words.
        - Removes stopwords and applies lemmatization.
        - Converts text to lowercase.
        """
        words = text.split()  # Tokenize by spaces
        
        # Remove stopwords and apply lemmatization to each word
        words = [self.lemmatizer.lemmatize(word.lower()) for word in words if word.lower() not in self.stop_words]
        
        # Join the processed words back into a single string
        return " ".join(words)

# Function to load and preprocess data from CSV files
def load_data(train_path, test_path):
    # Load training and testing data from CSV files
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Ensure that the necessary columns are present in the CSV files
    if not {'Class Index', 'Title', 'Description'}.issubset(train_df.columns):
        raise ValueError("CSV file does not contain the expected columns")

    # Initialize the text preprocessor
    preprocessor = TextPreprocessor()
    
    # Apply preprocessing to the 'Description' column in both training and testing datasets
    train_df['Description'] = train_df['Description'].apply(preprocessor.preprocess)
    test_df['Description'] = test_df['Description'].apply(preprocessor.preprocess)

    # Convert preprocessed DataFrames to lists of tuples (text, label)
    train_data = list(zip(train_df['Description'].tolist(), train_df['Class Index'].tolist()))
    test_data = list(zip(test_df['Description'].tolist(), test_df['Class Index'].tolist()))

    return train_data, test_data

# Step 2: Dataset class for preparing data for DistilBERT
class DistilBertTextDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        """
        Tokenize the given data using the DistilBERT tokenizer:
        - Truncates text to a maximum length of 128 tokens.
        - Pads shorter sequences to ensure uniform input size.
        """
        self.encodings = tokenizer([text for text, _ in data], truncation=True, padding=True, max_length=max_length)
        self.labels = [label - 1 for _, label in data]  # Adjust labels to start from 0

    def __len__(self):
        # Return the number of data samples
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Retrieve a single sample from the dataset:
        - Returns tokenized inputs and corresponding labels as tensors.
        """
        item = {key: torch.tensor(val[idx]).to(device) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx]).to(device)
        return item

# Step 3: Function for training the model for one epoch
def train_epoch(model, data_loader, optimizer):
    model.train()  # Set the model to training mode
    total_loss, correct_predictions = 0, 0

    # Loop through each batch in the data loader
    for batch in data_loader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        optimizer.zero_grad()  # Clear previous gradients
        
        # Forward pass: compute model outputs and loss
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits
        
        # Get predictions and calculate accuracy
        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        total_loss += loss.item()
        
        # Backward pass: compute gradients and update model parameters
        loss.backward()
        optimizer.step()

    # Return the accuracy and average loss for the epoch
    return correct_predictions.double() / len(data_loader.dataset), total_loss / len(data_loader)

# Step 4: Function to evaluate the model on the test set
def evaluate_model(model, data_loader):
    model.eval()  # Set the model to evaluation mode
    predictions, true_labels = [], []

    with torch.no_grad():
        # Loop through each batch in the test data loader
        for batch in data_loader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']

            # Get model outputs without computing gradients
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Get predictions and collect them
            _, preds = torch.max(logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    # Return the predicted and actual labels for evaluation
    return predictions, true_labels

# Step 5: Main function to run the entire pipeline
def main(train_path, test_path):
    # Load and preprocess training and testing data
    train_data, test_data = load_data(train_path, test_path)

    # Initialize the DistilBERT tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Calculate the number of unique classes in the dataset
    num_classes = len(set([label for _, label in train_data]))
    
    # Load the pre-trained DistilBERT model for sequence classification
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_classes).to(device)

    # Prepare datasets and data loaders
    train_dataset = DistilBertTextDataset(train_data, tokenizer)
    test_dataset = DistilBertTextDataset(test_data, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)

    # Initialize the optimizer
    optimizer = AdamW(model.parameters(), lr=2e-5)

    # Training loop over multiple epochs
    epochs = 5
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_acc, train_loss = train_epoch(model, train_loader, optimizer)
        print(f"Train loss: {train_loss:.4f}, accuracy: {train_acc:.4f}")

    # Evaluate the model on the test set and print the classification report
    predictions, true_labels = evaluate_model(model, test_loader)
    print("Classification Report:")
    print(classification_report(true_labels, predictions))

    # Save the trained model and tokenizer for later use
    model_save_path = "distilbert_model"
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"Model and tokenizer saved to {model_save_path}")

# Entry point of the script
if __name__ == "__main__":
    main("train.csv", "test.csv")
