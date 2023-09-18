import requests
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import re, os
import torch
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction import _stop_words
import string
import numpy as np
import pandas as pd
from newspaper import Article
import base64
import docx2txt
from io import StringIO
from PyPDF2 import PdfFileReader
import validators
import nltk
import warnings
import streamlit as st
from PIL import Image
nltk.download ('punkt')
from nltk import sent_tokenize
warnings.filterwarnings('ignore')


def url_text_extract(url:str):
    try:
        article = Article(url)
        article.download()
        article.parse()
        text = article.text
        title = article.title
        return title, text
    except Exception as e:
        st.error(f"Error occurred while fetching data from URL: {e}")
        return None, None

from PyPDF2 import PdfReader

def file_text_extract(file):
    pdf_title = None
    text = ""

    if file.type == "application/pdf":
        pdfReader = PdfReader(file)
        for page in pdfReader.pages:
            text += page.extract_text()

        pdf_title = file.name  # Set the PDF title to the uploaded file's name

    return text, pdf_title

def preprocess_text(text, window_size=5):
    text = text.encode("ascii", "ignore").decode()
    # ... other preprocessing steps ...

    paragraphs = []
    for paragraph in text.replace('\n', " ").split("\n\n"):
        if len(paragraph.strip()) > 0:
            paragraphs.append(sent_tokenize(paragraph.strip()))

    p = []
    for paragraph in paragraphs:
        for start_idx in range(0, len(paragraph), window_size):
            end_idx = min(start_idx + window_size, len(paragraph))
            p.append(' '.join(paragraph[start_idx:end_idx]))

    st.write(f"Sentence: {sum([len(p) for p in paragraphs])}")
    st.write(f'Paragraph:{len(p)}')

    return p

def bi_encode(bi_enc, passages):
    try:
        bi_encoder = SentenceTransformer(bi_enc)
        with st.spinner('Computing embeddings for the document...'):
            corpus_embeddings = bi_encoder.encode(passages, convert_to_tensor=True, show_progress_bar=True)
        st.success(f"Embeddings Shape: {corpus_embeddings.shape}")
        return bi_encoder, corpus_embeddings
    except Exception as e:
        st.error(f"Error occurred while initializing the model: {e}")
        return None, None

from sentence_transformers import CrossEncoder  

def cross_encode():
    global cross_encoder
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-6")
    return cross_encoder

from sklearn.feature_extraction import _stop_words


def bm25_tokenizer(text):
    tokenized_doc = []
    for token in text.lower().split():
        token = token.strip(string.punctuation)
        if len(token) > 0 and token not in _stop_words.ENGLISH_STOP_WORDS:
            tokenized_doc.append(token)
    
    return tokenized_doc



def bm25_api(passages):
    if not passages:
        raise ValueError("No passages provided for BM25 initialization.")

    tokenized_corpus = [bm25_tokenizer(passage) for passage in passages]
    tokenized_corpus = [doc for doc in tokenized_corpus if doc]  # Remove empty documents
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25



def df_results(model, top_k, score='score'):
    df = pd.DataFrame([[hit[score], passages[hit['corpus_id']]] for hit in model[0:top_k]],
        columns=["Score", "Text"]
    )
    df["Score"] = round(df["Score"], 2)
    return df

st.title("Document Search ")
st.subheader("Document Content Semantic Search")
bi_enc_options = ['msmarco-distilbert-base-dot-prod-v3']
bi_encoder_type = st.selectbox("Choose the model you want to run...", options=bi_enc_options, key='sbox')
window_size = st.slider("ParagraphSize: Choose how many sentence you want in the paragraph", min_value=1, max_value=12, value=5, key='slider')
window_size = window_size

def clear_text():
    if input_type == "URL":
        st.session_state["text_url"] = " "
        st.session_state["text_input"] = " "

def clear_search_text():
    st.session_state["text-input"] = " "
    st.session_state["text_url"] = " "

input_score_options = ["upload file(.txt, .pdf, .doc file)", "URL"]


input_type = st.selectbox("Choose input Source", options=input_score_options, key='sbox1')
from validators import url as validate_url 



# if input_type == "URL":  
#     url_text = st.text_input("Please enter URL here", value="https://www.example.com", key="text-url")
#     if validate_url(url_text):  # Using the imported URL validation function
#         title, text = url_text_extract(url_text)
#         passages = preprocwss_text(text, window_size=window_size)
#     else:
#         upload_doc = st.file_uploader("Upload a .txt or .pdf file", type=["txt", "pdf"], key="upload-doc")
#         if upload_doc:
#             text, pdf_title = file_text_extract(upload_doc)
#             passages = preprocwss_text(text, window_size=window_size)


# if input_type == "URL":
#     url_text = st.text_input("Please enter URL here", value="https://www.example.com", key="text-url")
#     if validate_url(url_text):  # Using the imported URL validation function
#         title, text = url_text_extract(url_text)
#         passages = preprocess_text(text, window_size = window_size)
#     else:
#         upload_doc = st.file_uploader("Upload a .txt or .pdf file", type=["txt", "pdf"], key="upload-doc")
#         if upload_doc:
#             text, pdf_title = file_text_extract(upload_doc)
#             passages = preprocess_text(text, window_size = window_size)
# else:
#     # Handle the case when input_type is "upload file"
#     uploaded_file = st.file_uploader("Upload a file", type = ["txt", "pdf", "docx"], key = "upload-file")
#     if uploaded_file:
#         text, pdf_title = file_text_extract(uploaded_file)
#         passages = preprocess_text(text, window_size = window_size)

if input_type == "URL":
    url_text = st.text_input("Please enter URL here", value="https://www.example.com", key="text-url")
    if validate_url(url_text):
        title, text = url_text_extract(url_text)
        passages = preprocess_text(text, window_size=window_size)
    else:
        upload_doc = st.file_uploader("Upload a .txt or .pdf file", type=["txt", "pdf"], key="upload-doc")
        if upload_doc:
            text, pdf_title = file_text_extract(upload_doc)
            passages = preprocess_text(text, window_size = window_size)
else:
    uploaded_file = st.file_uploader("Upload a file", type=["txt", "pdf", "docx"], key="upload-file")
    if uploaded_file:
        text, pdf_title = file_text_extract(uploaded_file)
        passages = preprocess_text(text, window_size=window_size)
        
search_query = st.text_input("Please enter your search query here", value="Common activation safety rule", key="text-input")
col1, col2 = st.columns(2)

with col1:
    search = st.button("Search", key="search-button", help="Check to search!")

with col2:
    clear = st.button('Clear text input', key="clear-button", help="Click to clear the URL input and search query")

top_k = 5

# def search_func(query, top_k=top_k):

#     global bi_encoder, cross_encoder
#     if input_type == "URL":
#         st.subheader(f'Document Header: {title}')  
#     else:
#         st.subheader(f'Document Header: {Pdf_title}') 
#     col3, col4 = st.columns(2)  

#     bm25_scores = bm25.get_scores(bm25_tokenizer(query))  # Use get_scores() instead of get_score()
    
#     # Keep the top 3 elements
#     top_n = np.argpartition(bm25_scores, -3)[-3:]  
#     bm25_hits = [{"corpus_id": idx, 'score': bm25_scores[idx]} for idx in top_n]  
#     bm25_hits = sorted(bm25_hits, key=lambda x: x["score"], reverse=True)  

# Inside the search_func function:
def search_func(query, top_k=top_k):
    global bi_encoder, cross_encoder, bm25
    # ... your code ...

    bm25_scores = bm25.get_scores(bm25_tokenizer(query))  # Use get_scores() instead of get_score()
    top_n = np.argpartition(bm25_scores, -3)[-3:]
    bm25_hits = [{"corpus_id": idx, 'score': bm25_scores[idx]} for idx in top_n]
    bm25_hits = sorted(bm25_hits, key=lambda x: x["score"], reverse=True)

    st.header("Lexical Search")
    st.write(pd.DataFrame(bm25_hits).to_html(index=False), unsafe_allow_html=True)

    question_embedding = bi_encoder.encode(query, convert_to_tensor=True)
    question_embedding = question_embedding.cpu()  
    hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=top_k, score_function=util.dot_score)
    hits = hits[0]

    cross_inp = [[query, passages[hit['corpus_id']]] for hit in hits]
    cross_scores = cross_encoder.predict(cross_inp)

    for idx in range(len(cross_scores)):  
        hits[idx]['cross-score'] = cross_scores[idx]

    hits = sorted(hits, key=lambda x: x['score'], reverse=True)

    st.header('Semantic Search')
    st.write(df_results(hits, top_k).to_html(index=False), unsafe_allow_html=True)


# def search_func(query, top_k=top_k):

#     global bi_encoder, cross_encoder
#     if input_type == "URL":
#         st.subheader(f'Document Header: {title}')  
#     else:
#         st.subheader(f'Document Header: {Pdf_title}') 
#     col3, col4 = st.columns(2)  

#     bm25_Score = bm25.get_score(bm25_tokenizer(query))
#     top_n = np.argpartition(bm25_Score, -3)[-3:]  
#     bm25_hits = [{"corpus_id": idx, 'score': bm25_score[idx]} for idx in top_n]  
#     bm25_hits = sorted(bm25_hits, key=lambda x: x["score"], reverse=True)  

#     with col3:
#         st.header("Lexical Search")
#         st.write(pd.DataFrame(bm25_hits).to_html(index=False), unsafe_allow_html=True)  

#     question_embedding = bi_encoder.encode(query, convert_to_tensor=True)
#     question_embedding = question_embedding.cpu()  
#     hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=top_k, score_function=util.dot_score)
#     hits = hits[0]

#     cross_inp = [[query, passages[hit['corpus_id']]] for hit in hits]
#     cross_scores = cross_encoder.predict(cross_inp)

#     for idx in range(len(cross_scores)):  
#         hits[idx]['cross-score'] = cross_scores[idx]

#     hits = sorted(hits, key=lambda x: x['score'], reverse=True)

#     cross_df = df_results(hits, top_k)
#     hits = sorted(hits, key=lambda x: x['cross-score'], reverse=True)

#     rerank_df = df_results(hits, top_k, 'cross-score')
#     with col4:
#         st.header('Semantic Search')
#         st.write(rerank_df.to_html(index=False), unsafe_allow_html=True)

if search:
    if bi_encoder_type:
        with st.spinner(
            text=f'Loading {bi_encoder_type} embeddings. This might take a few seconds depending on the length of your document...'
        ):
            bi_encoder, corpus_embeddings = bi_encode(bi_encoder_type, passages)
            cross_encoders = cross_encode()
            bm25 = bm25_api(passages)


    with st.spinner(
        text="Embedding completed, Searching for relevant text for the given query...."
    ):
        search_func(search_query, top_k)  
