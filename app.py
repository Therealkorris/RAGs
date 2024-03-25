import streamlit as st
from components.sidebar import sidebar
from components.utils import (
    initialize_state,
    load_qa_chain
)
import speech_recognition as sr
from gtts import gTTS
import tempfile
from io import BytesIO

def main():
    st.set_page_config(page_title="KM RAG")
    st.title("KM RAG: Knowledge Management with Retrieval-Augmented Generation")

    saved_files_info = sidebar()
    st.markdown("***")
    st.subheader('Interaction with Documents')

    initialize_state()

    if saved_files_info and not st.session_state.qa_chain:
        st.session_state.qa_chain = load_qa_chain(saved_files_info)

    if st.session_state.qa_chain:
        st.success("Configuration complete")
        input_method = st.radio("Choose input method:", ("Text", "Voice"))
        prompt = None

        if input_method == "Text":
            prompt = st.text_input('Ask questions about the uploaded documents', key="text_input")
        elif input_method == "Voice":
            st.write("Click 'Record' to ask your question via voice.")
            record = st.button("Record")
            if record:
                r = sr.Recognizer()
                with sr.Microphone() as source:
                    st.write("Listening...")
                    audio_data = r.listen(source)
                    st.write("Recognizing...")
                    try:
                        prompt = r.recognize_google(audio_data)
                        st.write(f"You said: {prompt}")
                    except sr.UnknownValueError:
                        st.error("Google Speech Recognition could not understand audio")
                    except sr.RequestError as e:
                        st.error(f"Could not request results from Google Speech Recognition service; {e}")

        if prompt and (st.session_state.messages[-1]["content"] != prompt or st.session_state.messages[-1]["role"] != "user"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.spinner("Retrieving relevant information and generating output..."):
                response = st.session_state.qa_chain.run(prompt)
            st.session_state.messages.append({"role": "assistant", "content": response})
            # Text-to-Speech response
            tts = gTTS(text=response, lang='en')
            tts_file = tempfile.NamedTemporaryFile(delete=False)
            tts.save(tts_file.name)
            audio_file = open(tts_file.name, "rb")
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format="audio/mp3")

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
    else:
        st.info("Please complete the configuration in the sidebar to proceed.")

if __name__ == '__main__':
    main()
