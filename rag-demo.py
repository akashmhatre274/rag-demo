import openai
import streamlit as st
import re, os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from pymongo import MongoClient

openai_api = st.secrets["OPEN_AI"]
openai.api_key = openai_api
os.environ["OPENAI_API_KEY"] = openai_api
embeddings = OpenAIEmbeddings()
MONGODB_ATLAS_CLUSTER_URI = st.secrets["MONGODB_ATLAS_CLUSTER_URI"]

st.set_page_config(page_title="Amotions Demo",
                   page_icon="https://www.amotionsinc.com/navbar-logo.svg")
st.image("https://www.amotionsinc.com/navbar-logo.svg")
st.title("Amotions RAG demo")

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

with st.form(key ='Form1'):
    with st.sidebar:
        email = st.sidebar.text_input("Enter your corporate email address", "")  

if not email:
    st.warning("Please enter corporate email address.")
domain = ""
email_pattern = r'@([A-Za-z0-9.-]+)'
match = re.search(email_pattern, email)
if match:
    domain = match.group(1)
    domain = re.sub(r'\..*$', '', domain)

client = MongoClient(MONGODB_ATLAS_CLUSTER_URI)

DB_NAME = "companies"
COLLECTION_NAME = "name"
ATLAS_VECTOR_SEARCH_INDEX_NAME = domain

MONGODB_COLLECTION = client[DB_NAME][COLLECTION_NAME]

vector_search = MongoDBAtlasVectorSearch.from_connection_string(
    MONGODB_ATLAS_CLUSTER_URI,
    DB_NAME + "." + COLLECTION_NAME,
    OpenAIEmbeddings(disallowed_special=()),
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME
)

qa_retriever = vector_search.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 100,
        "post_filter_pipeline": [{"$limit": 25}]
    }
)

prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)


if question := st.chat_input("What is up?"):
    
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        qa = RetrievalQA.from_chain_type(llm=OpenAI(),chain_type="stuff", retriever=qa_retriever, return_source_documents=True, chain_type_kwargs={"prompt": PROMPT})
        response = qa({"query": question})
        full_response += response["result"]+"\n"
        limit = 5
        for i in range(min(limit,len(response["source_documents"]))):
            full_response += "\n source: " + response["source_documents"][i].metadata["source"]+" \n"
            full_response += "\n page : " + str(response["source_documents"][i].metadata["page"])+" \n"
        message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )
