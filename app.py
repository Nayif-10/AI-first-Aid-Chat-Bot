import streamlit as st
import os
from PyPDF2 import PdfReader    # to read documents
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings   # for embeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS  # to store vectors
from langchain_google_genai import ChatGoogleGenerativeAI   #This imports the ChatGoogleGenerativeAI class from the langchain_google_genai module, which is used for creating a chatbot using Google's generative AI.
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()  # Loads environment variables from a .env file

os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_pdf_text(pdf_path):
    with open(pdf_path, "rb") as pdf_file:
        pdf_reader = PdfReader(pdf_file)        #Defines a function get_pdf_text that takes a path to a PDF file, opens the file, reads its contents page by page, and returns the extracted text.
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):      #Defines a function get_text_chunks that takes a long text and splits it into smaller chunks of specified sizes using RecursiveCharacterTextSplitter.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)     
    return chunks


def get_vector_store(text_chunks):      #Defines a function get_vector_store that creates embeddings for the text chunks using GoogleGenerativeAIEmbeddings and stores them in a FAISS vector store.
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():        #Defines a function get_conversational_chain that sets up a conversational chain using the ChatGoogleGenerativeAI model for question answering.
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "Ask me about First Aid", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)     #loading a specific model (ChatGoogleGenerativeAI) that is capable of generating responses based on the input it receives.

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])      # This chain uses the loaded model and prompt to handle the question-answering process.
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def user_input(user_question, pdf_path):        #Defines a function user_input that takes a user question and a path to a PDF file, loads the FAISS vector store, searches for similar documents, and uses the conversational chain to generate a response.
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    print(response)
    st.write("Reply: \n", response["output_text"])


def main():
    st.set_page_config("FA")
    st.header("AID BASE⛑️")

    pdf_path = r"C:\Users\moham\OneDrive\Desktop\New folder (2)\First aid.pdf" # Specify the path to your PDF file here
    raw_text = get_pdf_text(pdf_path)
    text_chunks = get_text_chunks(raw_text)
    get_vector_store(text_chunks)

    user_question = st.text_input("Ask a First aid")
    submitted = st.button("Ask me")

    if submitted and user_question:
        user_input(user_question, pdf_path)

if __name__ == "__main__":
    main()
