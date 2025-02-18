How to Run SmartDoc
===================

Step 1: Install Dependencies
----------------------------
Use the following command to install all required Python libraries:

pip install torch transformers streamlit nltk scikit-learn numpy pandas PyPDF2 python-docx protobuf

Step 2: Train the Model
-----------------------
Run the training script to train the classification model:

python DistilBert_model_train.py

Step 3: Run the Streamlit Application
-------------------------------------
Launch the web interface:

streamlit run app.py
