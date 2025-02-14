import streamlit as st
import os

from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import ChatPromptTemplate

from langchain.chains import create_retrieval_chain

from langchain_community.vectorstores import FAISS

from langchain_community.document_loaders import PyPDFDirectoryLoader


from dotenv import load_dotenv()

load_dotenv()

##  lOAD GROQ API KEY and OPENAI API KEY

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

groq_api_key = os.getenv('GROQ_API_KEY')



st.title("ChatGroq with llama 3 demo")

llm=ChatGroq(groq_api_key ,model_name = "Llama3-Bb-8192")


prompt = ChatPromptTemplate.from_template(
    """
    answer the question baesd on the provided context only
    <context>
      {context}
    <context>
    question: {input}

"""
)


def vector_embedding():
   
   if "vector" not in st.session_state:
       
     st.session_state.embeddings =OpenAIEmbeddings()
     st.session_state.loader =PyPDFDirectoryLoader("./data")
     st.session_state.docs =st.session_state.loader.load()
     st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
     st.session_state.final_documents =st.session_state.text_splitter.split_documents(st.session.state.docs[:50])

     st.session_state.vectorstore = FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)




prompt1 =st.text_input("enter the question from document")

if st.button("Documents embedding"):
    vector_embedding()
    st.write("vector store db is ready")


   


import time
##
if prompt1:
    
   
   document_chain = create_stuff_documents_chain(llm,prompt) ##stuff documents responsible and returning the context

   retriever = st.session_state.vectorstore.as_retriever()


   retrieval_chain = create_retrieval_chain(retriever,document_chain) 

   
   start = time.process_time()
   response = retrieval_chain.invoke({'input':prompt1})

   print("Response time:", time.process_time() - start)
   st.write(response['answer'])

   with st.expander("Document similarity search"):
      
      for i,doc in enumerate(response["context"]):
         st.write(doc.page_content)
         st

