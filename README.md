# Plagiarism-Detection

Using the BERT (Bidirectional Encoder Representations from Transformers) model, users can compare two documents in this Streamlit-based web tool to identify possible instances of plagiarism. For document comparison, the program allows for both file uploads and text input.

## **Features**
-**Text Input:** Manually input text into the app for plagiarism detection.  
-**File Upload:** Upload `.txt`, `.pdf`, or `.docx` files for plagiarism detection.  
-**Cosine Similarity:** Calculates similarity between documents using cosine similarity.  
-**Threshold Adjustment:** Allows users to set a similarity threshold for detecting plagiarism.

## **Install Dependencies**
```bash
pip install -r requirements.txt
```

## **Running the App Locally**
```bash
streamlit run Plagiarism_Detector.py
```



