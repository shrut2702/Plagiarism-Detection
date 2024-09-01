import streamlit as st
import torch
import nltk
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def preprocess_doc(text):
    #Tokenize and remove stop words and punctuation
    tokens = nltk.word_tokenize(text.lower())
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
    return ' '.join(tokens)

def extract_features_bert(docs):
    #Tokenize documents using BERT tokenizer
    tokenized_docs = [tokenizer.encode(doc, add_special_tokens=True) for doc in docs]
    #Pad and truncate token sequences to fixed length
    max_len = max([len(doc) for doc in tokenized_docs])
    padded_docs = [doc + [0]*(max_len-len(doc)) for doc in tokenized_docs]
    #Convert token sequences to tensors and pass through BERT model
    input_ids = torch.tensor(padded_docs)
    with torch.no_grad():
        outputs = model(input_ids)
        bert_vectors = outputs[0][:,0,:].numpy()
    return bert_vectors

def detect_plagiarism(docs, method, threshold=0.9):
    if method == 'bert':
        features = extract_features_bert(docs)
        similarity = cosine_similarity([features[0]], [features[1]])[0][0]
    else:
        raise ValueError("Invalid method")
    return similarity * 100

st.title("Plagiarism Detection App")
st.write("Compare two documents to detect potential plagiarism using.")

input_type = st.selectbox("Select the method for plagiarism detection", ('Text', 'File'))

if input_type == 'Text':
    #Text input for two documents
    doc1 = st.text_area("Document 1", height=150)
    doc2 = st.text_area("Document 2", height=150)
elif input_type == 'File':
    #File upload for two documents
    uploaded_file1 = st.file_uploader("Upload Document 1", type=["txt", "pdf", "docx"])
    uploaded_file2 = st.file_uploader("Upload Document 2", type=["txt", "pdf", "docx"])

    if uploaded_file1 and uploaded_file2:
        #Reading content from the files
        doc1 = uploaded_file1.read().decode("utf-8")
        doc2 = uploaded_file2.read().decode("utf-8")
    else:
        doc1, doc2 = None, None


#method = st.selectbox("Select the method for plagiarism detection", ('tfidf', 'bert'))
method='bert'
threshold = st.slider("Set similarity threshold", 0.0, 100.0, 80.0)

if st.button("Detect Plagiarism"):
    if doc1 and doc2:
        similarity = detect_plagiarism([doc1, doc2], method=method, threshold=threshold)
        if similarity > threshold:
            st.warning(f"Plagiarism detected with similarity score: {similarity:.2f}%")
        else:
            st.success(f"No plagiarism detected with similarity score: {similarity:.2f}%")
    else:
        st.error("Please enter both documents or upload the files.")