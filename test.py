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

llm_ollama = ChatOllama(model="mistral")
message_history = ChatMessageHistory()
memory = ConversationBufferMemory(memory_key="chat_history", output_key="answer", chat_memory=message_history, return_messages=True)
pdf_processed = False
chain = None  # Initialize chain outside conditional blocks

def add_document_to_chroma(collection_name, texts, metadatas):
    url = f"http://localhost:8000/api/v1/collections/{collection_name}/documents"
    headers = {"Content-Type": "application/json"}
    payload = {
        "texts": texts,
        "metadatas": metadatas,
    }
    try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            st.success("Document added to Chroma DB successfully.")
        else:
            st.error(f"Failed to add document to Chroma DB: HTTP Status Code {response.status_code}. Response: {response.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to Chroma API: {e}")

def process_and_add_pdf(file, collection_name):
    # Process PDF
    pdf = PyPDF2.PdfReader(file)
    pdf_text = "".join(page.extract_text() for page in pdf.pages)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=50)
    texts = text_splitter.split_text(pdf_text)
    metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]
    
    # Directly use Chroma.from_texts to handle embedding and adding to collection
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    #persist_directory = f"chroma/{collection_name}"
    Chroma.from_texts(texts, embeddings, collection_name=collection_name, metadatas=metadatas)
    # vectordb = Chroma.from_texts(texts, embeddings, collection_name=collection_name, metadatas=metadatas, persist_directory=persist_directory)
    # vectordb.persist()



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


st.title("Korani PDF LLM Q/A")
st.sidebar.title("Document Upload")
uploaded_files = st.sidebar.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

# Attempt to connect to Chroma first and create the chain using existing collections
embeddings = OllamaEmbeddings(model="nomic-embed-text")
try:
    collections = list_chroma_collections()
    if collections:
        collection_name = collections[0]  # Get the first collection name
        docsearch = Chroma(collection_name=collection_name, embedding_function=embeddings)
        chain = ConversationalRetrievalChain.from_llm(llm=llm_ollama, chain_type="stuff", retriever=docsearch.as_retriever(), memory=memory, return_source_documents=True)
        st.sidebar.success("Existing collection loaded. You can now ask questions about the document(s) using text or speech input.")
        pdf_processed = True
except requests.exceptions.RequestException as e:
    st.error(f"Error connecting to Chroma API: {e}")

def determine_collection_name(file_name):
    # Remove the file extension to use as the collection name
    return os.path.splitext(file_name)[0]

# Updated workflow for handling uploaded files
if uploaded_files:
    for uploaded_file in uploaded_files:
        collection_name = determine_collection_name(uploaded_file.name)
        with st.spinner(f'Processing and adding PDF {uploaded_file.name}...'):
            temp_dir = tempfile.TemporaryDirectory()
            temp_pdf_path = os.path.join(temp_dir.name, uploaded_file.name)
            with open(temp_pdf_path, 'wb') as temp_pdf:
                temp_pdf.write(uploaded_file.getvalue())
            process_and_add_pdf(temp_pdf_path, collection_name)
            
        # Check if a chain for this collection already exists or create a new one
        if not chain:
            docsearch = Chroma(collection_name=collection_name, embedding_function=embeddings)
            chain = ConversationalRetrievalChain.from_llm(llm=llm_ollama, chain_type="stuff", retriever=docsearch.as_retriever(), memory=memory, return_source_documents=True)
            st.sidebar.success(f"PDF processed and added to collection '{collection_name}'. You can now ask questions about the document(s) using text or speech input.")
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
                    if chain is not None:
                        res = chain.invoke(query)
                        answer = res["answer"]
                        st.write("Answer:", answer)
                        audio_data = speak(answer)
                        with st.spinner('Playing audio...'):
                            audio_bytes = audio_data.getvalue()
                            st.audio(audio_bytes, format='audio/mp3')
                    else:
                        st.warning("An error occurred creating the search chain. Please try again.")
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
            st.warning("An error occurred creating the search chain. Please try again.")


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