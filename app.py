import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from htmlTemplates import css, bot_template, user_template


# Function to extract text from PDF
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


# Function to split the extracted text into chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


# Function to create a vector store using the extracted text chunks
def get_vectorstore(text_chunks):
    model_name = "hkunlp/instructor-xl"
    embeddings = HuggingFaceInstructEmbeddings(model_name=model_name)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


# Function to initialize the HuggingFaceHub LLM and set up the conversation chain
def get_conversation_chain(vectorstore):
    # Initialize HuggingFace LLM using HuggingFaceHub
    llm = HuggingFaceHub(repo_id="ibm-granite/granite-3b-code-base", model_kwargs={"temperature": 0.7})

    # Create memory buffer for conversation history
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True
    )

    # Create the conversation chain
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

    return conversation_chain


# Function to handle user input and manage the conversation
def handle_userinput(user_question):
    if st.session_state.conversation is None:
        st.error("Conversation chain is not initialized. Please process your PDFs first.")
        return

    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


# Main function to run the Streamlit app
def main():
    load_dotenv()  # Load environment variables
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")

    # User input for the question
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    # Sidebar for PDF file upload and processing
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # Extract text from PDFs
                raw_text = get_pdf_text(pdf_docs)

                # Split the text into chunks
                text_chunks = get_text_chunks(raw_text)

                # Create the vector store
                vectorstore = get_vectorstore(text_chunks)

                # Initialize the conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)


# Run the app
if __name__ == '__main__':
    main()
