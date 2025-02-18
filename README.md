# SmartDoc: Intelligent Text Search & Classification

SmartDoc is an AI-powered system designed for efficient **text classification, semantic search, and question answering**. It leverages **DistilBERT**, a lightweight transformer model, to enhance **document categorization** and **intelligent search functionalities**.

## Features
- ✅ **Text Classification** – Categorizes documents into predefined labels like Business, Sports, and Technology.
- ✅ **Semantic Search** – Enables typo-tolerant keyword search and retrieves relevant text segments.
- ✅ **Question Answering** – Extracts precise answers from documents using NLP techniques.
- ✅ **User-Friendly Interface** – Built with **Streamlit** for easy document upload, classification, and search.
- ✅ **Scalability & Efficiency** – Optimized for fast, context-aware information retrieval.

## Installation
To set up SmartDoc locally, follow these steps:

```bash
# Clone the repository
git clone https://github.com/yourusername/SmartDoc.git
cd SmartDoc

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

## Usage
1. Upload a document (PDF/Word/Text).
2. Choose classification, search, or question-answering.
3. View categorized results or retrieve relevant text snippets.

## Model Architecture
SmartDoc integrates:
- **DistilBERT** for text classification & extractive QA.
- **TF-IDF & Cosine Similarity** for semantic search.
- **Preprocessing techniques** like tokenization, stopword removal, and lemmatization.

## Dataset & Training
The model is fine-tuned on a **news dataset** with multiple categories. Traditional models (Naive Bayes, Random Forest) were tested, but **DistilBERT achieved 94% accuracy**.

## Future Enhancements
- 🌍 **Multilingual Support** (e.g., Hindi, Spanish).
- 🔍 **Enhanced Search** with neural retrieval techniques.
- 🚀 **Scalability Improvements** using FAISS or Elasticsearch.

## Contributing
Feel free to open issues or pull requests for improvements!

---

🚀 **SmartDoc: Making Text Search Smarter & Faster!**
