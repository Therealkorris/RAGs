# docker pull chromadb/chroma
# Create local folder called chroma for persisting the Chroma DB files
# docker run -p 8000:8000 -v E:\GroqPDFFastChatbot\chroma:/chroma/chroma chromadb/chroma 
# docker run -p 8000:8000 chromadb/chroma (dont use, use the one above instead)
# localhost:8000/api/v1


import os
from io import BytesIO
import gtts
import streamlit as st
import PyPDF2
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain_community.chat_models import ChatOllama
import speech_recognition as sr
import tempfile
import requests

llm_groq = ChatOllama(model="mistral")
message_history = ChatMessageHistory()
memory = ConversationBufferMemory(memory_key="chat_history", output_key="answer", chat_memory=message_history, return_messages=True)
pdf_processed = False
chain = None

# def process_pdf(file):
#     pdf = PyPDF2.PdfReader(file)
#     pdf_text = "".join(page.extract_text() for page in pdf.pages)
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=50)
#     texts = text_splitter.split_text(pdf_text)
#     metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]
#     embeddings = OllamaEmbeddings(model="nomic-embed-text")
#     docsearch = Chroma.from_texts(texts, embeddings, metadatas=metadatas)
#     return docsearch

def process_pdf(file):
    pdf = PyPDF2.PdfReader(file)
    pdf_text = "".join(page.extract_text() for page in pdf.pages)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=50)
    texts = text_splitter.split_text(pdf_text)
    metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    collection_name = "Emilie" 
    docsearch = Chroma(collection_name=collection_name, embedding_function=embeddings)
    # Add the processed documents to the collection
    docsearch.add_documents(texts, metadatas)
    return docsearch

def list_chroma_collections():
    url = "http://localhost:8000/api/v1/collections"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            collections = [collection["name"] for collection in response.json()]
            return collections
        else:
            # Improved error handling: log the response content for debugging
            st.error(f"Failed to retrieve collections: HTTP Status Code {response.status_code}\nResponse: {response.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to Chroma API: {e}")
    return []


st.title("Kongsberg PDF LLM Q/A")
st.sidebar.title("Document Upload")
uploaded_files = st.sidebar.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

# Check if there are existing collections in Chroma DB
collections = list_chroma_collections()
if collections:
    # Use the existing collection to create the chain
    collection_name = collections[0]  # Get the first collection name
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    docsearch = Chroma(collection_name=collection_name, embedding_function=embeddings)
    chain = ConversationalRetrievalChain.from_llm(llm=llm_groq, chain_type="stuff", retriever=docsearch.as_retriever(), memory=memory, return_source_documents=True)
    st.sidebar.success("Existing collection loaded. You can now ask questions about the document(s) using text or speech input.")
    pdf_processed = True

if uploaded_files:
    for uploaded_file in uploaded_files:
        with st.spinner(f'Processing PDF {uploaded_file.name}...'):
            temp_dir = tempfile.TemporaryDirectory()
            temp_pdf_path = os.path.join(temp_dir.name, uploaded_file.name)
            with open(temp_pdf_path, 'wb') as temp_pdf:
                temp_pdf.write(uploaded_file.getvalue())
            docsearch = process_pdf(temp_pdf_path)
            # Assuming here you interface with your Docker-hosted Chroma without direct file handling
            chain = ConversationalRetrievalChain.from_llm(llm=llm_groq, chain_type="stuff", retriever=docsearch.as_retriever(), memory=memory, return_source_documents=True)
    st.sidebar.success("PDF(s) processed. You can now ask questions about the document(s) using text or speech input.")
    pdf_processed = True

def speak(text):
    tts = gtts.gTTS(text=text, lang='en')
    audio_data = BytesIO()
    tts.write_to_fp(audio_data)
    audio_data.seek(0)
    return audio_data

input_method = st.radio("Choose input method:", ("Text", "Speech"))
query_text_speech = ""

if input_method == "Text":
    query_text = st.text_input("Ask a question about the document (text input):", value="", key="text_query_text")

if input_method == "Speech":
    spoken_text = st.empty()
    speak_button = st.button("Speak your question")
    if speak_button:
        if not pdf_processed:
            st.warning("Please upload a PDF first before asking questions via speech.")
        else:
            st.text("Speak your question now (microphone icon might appear depending on your browser):")
            with sr.Microphone() as source:
                r = sr.Recognizer()
                audio = r.listen(source)
            try:
                recognizer = sr.Recognizer()
                query = recognizer.recognize_google(audio)
                spoken_text.text_area("You said:", value=query)
                query_text_speech = query
                query = query_text_speech
                if query:
                    res = chain.invoke(query)
                    answer = res["answer"]
                    st.write("Answer:", answer)
                    audio_data = speak(answer)
                    with st.spinner('Playing audio...'):
                        audio_bytes = audio_data.getvalue()
                        st.audio(audio_bytes, format='audio/mp3')
            except sr.UnknownValueError:
                st.error("Speech Recognition could not understand audio")
            except sr.RequestError as e:
                st.error(f"Could not request results from Google Speech Recognition service; {e}")

if input_method == "Text" and query_text:
    with st.spinner('Searching for answers...'):
        if chain is not None:
            res = chain.invoke(query_text)
            answer = res["answer"]
            st.write("Answer:", answer)
            audio_data = speak(answer)
            with st.spinner('Playing audio...'):
                audio_bytes = audio_data.getvalue()
                st.audio(audio_bytes, format='audio/mp3')
        else:
            st.warning("Please upload a PDF first before asking questions.")

import requests

def list_chroma_collections():
    url = "http://localhost:8000/api/v1/collections"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            collections = response.json()
            return collections
        else:
            st.error("Failed to retrieve collections: HTTP Status Code {}".format(response.status_code))
    except requests.exceptions.RequestException as e:
        st.error("Error connecting to Chroma API: {}".format(e))

    return []

# Displaying the existing collections in the sidebar, assuming you have a function to fetch those
st.sidebar.title("Existing Collections")
try:
    collections = list_chroma_collections()  # Get the list of collections from Chroma DB
    for collection in collections:
        st.sidebar.write(f"Collection: {collection}")
except Exception as e:
    st.sidebar.write("Error fetching collections:", str(e))
