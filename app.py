# Import necessary libraries
import streamlit as st
import torch
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from PyPDF2 import PdfReader
from docx import Document
import re
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering

# Load pre-trained DistilBERT model and tokenizer for extractive question answering
qa_model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")
qa_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")

# Check if GPU is available for acceleration, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the saved classification model and tokenizer
model_path = "distilbert_model"  # Path to your saved fine-tuned model
model = DistilBertForSequenceClassification.from_pretrained(model_path).to(device)
tokenizer = DistilBertTokenizer.from_pretrained(model_path)


# Text Preprocessing Class: Handles cleaning, stopword removal, and lemmatization
class TextPreprocessor:
    def __init__(self, remove_stopwords=True):
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()
        self.remove_stopwords = remove_stopwords

    # Function to preprocess text by tokenizing, removing stopwords, and lemmatizing
    def preprocess(self, text):
        words = text.split()
        if self.remove_stopwords:
            words = [word.lower() for word in words if word.lower() not in self.stop_words]
        else:
            words = [word.lower() for word in words]
        words = [self.lemmatizer.lemmatize(word) for word in words]
        return " ".join(words)


# Function to answer user questions based on context using the QA model
def answer_question(question, context):
    # Tokenize and encode the inputs for the question-answering model
    inputs = qa_tokenizer.encode_plus(question, context, return_tensors="pt", max_length=512, truncation=True)
    input_ids = inputs["input_ids"].tolist()[0]

    # Get the model's predictions for the start and end positions of the answer
    with torch.no_grad():
        outputs = qa_model(**inputs)
        answer_start_scores = outputs.start_logits
        answer_end_scores = outputs.end_logits

    # Find the indices with the highest scores for the start and end of the answer
    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1

    # Convert the token IDs back to a string to form the final answer
    answer = qa_tokenizer.convert_tokens_to_string(qa_tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    return answer


# Function to highlight search terms within text using markdown for visibility
def highlight_text(text, query):
    pattern = re.compile(re.escape(query), re.IGNORECASE)
    highlighted_text = pattern.sub(f"**{query}**", text)
    return highlighted_text


# Function to classify text into categories using the pre-trained model
def classify_text(text):
    preprocessor = TextPreprocessor()
    processed_text = preprocessor.preprocess(text)
    inputs = tokenizer(processed_text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    
    # Predict the class of the text using the model
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=-1).item()
    
    # Map numeric predictions to human-readable class labels
    class_labels = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
    return class_labels.get(predicted_class, "Unknown Class")


# Function to create a dictionary of words from the document for search correction
def build_word_dictionary(document_text):
    preprocessor = TextPreprocessor()
    processed_text = preprocessor.preprocess(document_text)
    words = set(processed_text.split())
    return list(words)


# Function to suggest corrections for misspelled search queries using cosine similarity
def suggest_correction(search_query, word_dictionary):
    corpus = [search_query] + word_dictionary
    vectorizer = TfidfVectorizer().fit_transform(corpus)
    query_vector = vectorizer[0]  # Vector for the search term
    dictionary_vectors = vectorizer[1:]  # Vectors for words in the dictionary
    similarities = cosine_similarity(query_vector, dictionary_vectors).flatten()
    
    # Find the word with the highest similarity score
    max_similarity_index = similarities.argmax()
    max_similarity_score = similarities[max_similarity_index]
    
    # Suggest correction only if similarity score is above a threshold
    similarity_threshold = 0.7
    if max_similarity_score >= similarity_threshold:
        return word_dictionary[max_similarity_index]
    return None


# Function to extract context around a search term (returns full sentences)
def extract_context(text, query):
    pattern = re.compile(rf"([^.]*\b{re.escape(query)}\b[^.]*\.)", re.IGNORECASE)
    matches = pattern.findall(text)
    return matches if matches else []


# Function to search within the text with typo correction and display results
def search_in_text_with_correction(text, search_query):
    word_dictionary = build_word_dictionary(text)
    suggested_word = suggest_correction(search_query, word_dictionary)
    
    # Suggest correction if a similar word is found
    if suggested_word and suggested_word.lower() != search_query.lower():
        st.write(f"Did you mean **'{suggested_word}'** instead of **'{search_query}'**?")
        if st.button("Use Suggested Term"):
            search_query = suggested_word

    # Extract and display context around the found word
    context_sentences = extract_context(text, search_query)
    if context_sentences:
        st.write(f"Found **'{search_query}'** in the text. Displaying results:")
        for idx, sentence in enumerate(context_sentences, 1):
            highlighted_sentence = highlight_text(sentence, search_query)
            st.markdown(f"{idx}. {highlighted_sentence}")
    else:
        st.write(f"The term **'{search_query}'** was not found.")


# Streamlit Application for user interaction
# Function to split text into manageable chunks
def split_text_into_chunks(text, tokenizer, max_length=512):
    tokens = tokenizer.tokenize(text)
    chunks = [tokens[i:i + max_length] for i in range(0, len(tokens), max_length)]
    return [" ".join(chunk) for chunk in chunks]


# Function to answer user questions based on context using the QA model with chunking
def answer_question_with_chunks(question, context, tokenizer, model, max_length=512):
    chunks = split_text_into_chunks(context, tokenizer, max_length)
    answers = []

    for chunk in chunks:
        # Tokenize and encode the inputs for the question-answering model
        inputs = tokenizer.encode_plus(question, chunk, return_tensors="pt", max_length=max_length, truncation=True)
        input_ids = inputs["input_ids"].tolist()[0]

        # Get the model's predictions for the start and end positions of the answer
        with torch.no_grad():
            outputs = model(**inputs)
            answer_start_scores = outputs.start_logits
            answer_end_scores = outputs.end_logits

        # Find the indices with the highest scores for the start and end of the answer
        answer_start = torch.argmax(answer_start_scores)
        answer_end = torch.argmax(answer_end_scores) + 1

        # Convert the token IDs back to a string to form the final answer
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
        answers.append(answer)

    # Combine all the answers from chunks
    combined_answer = " ".join([ans.strip() for ans in answers if ans.strip()])
    return combined_answer if combined_answer else "No relevant answer found."


# Updated Streamlit interface for answering questions with chunking
st.title("Document & Query Classification with Enhanced Search and QA")

# Sidebar navigation for different features
st.sidebar.title("Options")
option = st.sidebar.selectbox("Choose an option", ["Classify Document", "Classify Query"])

# Handling file upload and document classification
if option == "Classify Document":
    st.subheader("Upload a PDF or DOCX file")
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx"])

    document_text = ""
    if uploaded_file:
        ext = os.path.splitext(uploaded_file.name)[1].lower()
        # Extract text from PDF or DOCX
        if ext == ".pdf":
            reader = PdfReader(uploaded_file)
            document_text = "".join(page.extract_text() for page in reader.pages)
        elif ext == ".docx":
            doc = Document(uploaded_file)
            document_text = "\n".join(para.text for para in doc.paragraphs)
        else:
            st.error("Unsupported file format.")

    # Classify and interact with the uploaded document
    if document_text:
        st.subheader("Document Classification")
        if st.button("Classify Document"):
            class_name = classify_text(document_text)
            st.write(f"**Classified as:** {class_name}")

        # Enable search with typo correction
        search_enabled = st.checkbox("Enable Search with Correction")
        if search_enabled:
            search_query = st.text_input("Enter the search term")
            if st.button("Search") and search_query:
                search_in_text_with_correction(document_text, search_query)

        # Enable extractive question answering
        qa_enabled = st.checkbox("Enable Question Answering")
        if qa_enabled:
            user_question = st.text_input("Enter your question")
            if st.button("Get Answer") and user_question:
                answer = answer_question_with_chunks(user_question, document_text, qa_tokenizer, qa_model)
                if answer.strip():
                    st.subheader("Extracted Answer")
                    st.write(answer)
                else:
                    st.write("No relevant answer found.")

# Query classification and search feature
elif option == "Classify Query":
    st.subheader("Query Classification")
    user_query = st.text_area("Enter your query")

    if st.button("Classify Query") and user_query:
        class_name = classify_text(user_query)
        st.write(f"**Classified as:** {class_name}")

    search_enabled = st.checkbox("Enable Search in Query")
    if search_enabled:
        search_query = st.text_input("Enter the search term")
        if st.button("Search in Query") and search_query:
            search_in_text_with_correction(user_query, search_query)
