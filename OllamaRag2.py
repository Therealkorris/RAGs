import streamlit as st
import os
import speech_recognition as sr
from gtts import gTTS
from io import BytesIO
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.base import BaseCallbackHandler
from langchain_community.llms import Ollama
from langchain.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# Define the StreamHandler class as before
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        super().__init__()
        self.container = container

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if 'generated_text' not in st.session_state:
            st.session_state.generated_text = ""
        st.session_state.generated_text += token
        self.container.markdown(f"<div style='background-color:#808080;padding:10px;border-radius:10px;'>{st.session_state.generated_text}</div>", unsafe_allow_html=True)


# Setup and initialization code
if not os.path.exists('files'):
    os.mkdir('files')

if not os.path.exists('chroma'):
    os.mkdir('chroma')

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'template' not in st.session_state:
    st.session_state.template = """You are a knowledgeable chatbot, here to help with questions of the user. Your tone should be professional and informative.

    Context: {context}
    History: {history}

    User: {question}
    Chatbot:"""

if 'prompt' not in st.session_state:
    st.session_state.prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=st.session_state.template,
    )

if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="history",
        return_messages=True,
        input_key="question")

if 'llm' not in st.session_state:
    st.session_state.llm = Ollama(
        base_url="http://localhost:11434",
        model="mistral:instruct",
        verbose=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler(), StreamHandler(st.empty())]),
    )

if 'vectorstores' not in st.session_state:
    st.session_state.vectorstores = {}

st.title("Kongsberg Rag-Assistant")

# Sidebar setup
with st.sidebar:
    uploaded_file = st.file_uploader("Upload your PDF", type='pdf')
    mode_selection = st.radio("Choose Input Mode:", ("Text", "Speech"))
    chroma_directories = [name for name in os.listdir('chroma') if os.path.isdir(os.path.join('chroma', name))]
    selected_doc_name = st.selectbox("Select a document to query:", chroma_directories)


if uploaded_file is not None:
    file_path = f"files/{uploaded_file.name}"
    if not os.path.isfile(file_path):
        bytes_data = uploaded_file.getvalue()
        with open(file_path, "wb") as f:
            f.write(bytes_data)

        loader = PyPDFLoader(file_path)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200, length_function=len)
        all_splits = text_splitter.split_documents(data)

        doc_vectorstore = Chroma(persist_directory=f'chroma/{uploaded_file.name.split(".")[0]}',
                                 embedding_function=OllamaEmbeddings(model="mistral:instruct"))
        doc_vectorstore.add_documents(documents=all_splits)
        doc_vectorstore.persist()

        st.session_state.vectorstores[uploaded_file.name] = doc_vectorstore

# Display the bot's and user's messages
for interaction in st.session_state.chat_history:
    # Bot messages
    if 'bot' in interaction:
        with st.container():
            col1, col2 = st.columns([1, 9], gap="small")
            with col1:
                st.image("https://cdn.zonebourse.com/static/instruments-logo-39430100", width=30)
            with col2:
                st.markdown(f"<div style='background-color:#808080;padding:10px;border-radius:10px;'>{interaction['bot']}</div>", unsafe_allow_html=True)
    # User messages
    if 'user' in interaction:
        with st.container():
            col1, col2 = st.columns([9, 1], gap="small")
            with col1:
                st.markdown(f"<div style='background-color:#2f2f2f;padding:10px;border-radius:10px;color:white;'>{interaction['user']}</div>", unsafe_allow_html=True)
            with col2:
                st.image("https://emojigraph.org/media/microsoft/man-technologist_1f468-200d-1f4bb.png", width=30)

user_input = None
if mode_selection == "Text":
    user_input = st.text_input("Type your question here:", key="text_input", on_change=lambda: st.session_state.chat_history.append({"user": st.session_state["text_input"]}))
elif mode_selection == "Speech":
    recognizer = sr.Recognizer()
    if st.button("Speak"):
        with sr.Microphone() as mic:
            st.info("Listening...")
            try:
                audio_data = recognizer.listen(mic, timeout=5)
                user_input = recognizer.recognize_google(audio_data)
                st.session_state.chat_history.append({"user": user_input})
                st.success(f"Recognized text: {user_input}")
            except sr.UnknownValueError:
                st.error("Google Speech Recognition could not understand the audio.")
            except sr.RequestError as e:
                st.error(f"Could not request results from Google Speech Recognition service; {e}")
            except Exception as e:
                st.error(f"An error occurred: {e}")

if user_input:
    st.session_state.generated_text = ""
    st.session_state.chat_history.append({"bot": "..."})
    bot_response_container = st.empty()
    stream_handler = StreamHandler(bot_response_container)

    if selected_doc_name:
        if selected_doc_name not in st.session_state.vectorstores:
            doc_vectorstore = Chroma(persist_directory=f'chroma/{selected_doc_name}',
                                     embedding_function=OllamaEmbeddings(model="mistral:instruct"))
            st.session_state.vectorstores[selected_doc_name] = doc_vectorstore
        else:
            doc_vectorstore = st.session_state.vectorstores[selected_doc_name]

        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm=st.session_state.llm,
            chain_type='stuff',
            retriever=doc_vectorstore.as_retriever(),
            verbose=True,
            chain_type_kwargs={
                "verbose": True,
                "prompt": st.session_state.prompt,
                "memory": st.session_state.memory,
            }
        )

    with st.spinner("Assistant is typing..."):
        st.session_state.qa_chain(user_input)

    if 'generated_text' in st.session_state and st.session_state.generated_text.strip():
        st.session_state.chat_history[-1]['bot'] = st.session_state.generated_text
        bot_response_container.markdown(f"<div style='background-color:#808080;padding:10px;border-radius:10px;'>{st.session_state.generated_text}</div>", unsafe_allow_html=True)
        
        tts = gTTS(text=st.session_state.generated_text, lang='en', slow=False)
        tts_audio = BytesIO()
        tts.write_to_fp(tts_audio)
        tts_audio.seek(0)
        st.audio(tts_audio, format='audio/mp3')
    else:
        st.error("No text was generated for speech conversion.")
