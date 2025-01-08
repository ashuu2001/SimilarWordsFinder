import streamlit as st

import os

# from langchain.embeddings import CohereEmbeddings
from langchain_community.embeddings.cohere import CohereEmbeddings

from langchain.vectorstores import FAISS

from dotenv import load_dotenv

load_dotenv()

os.environ['cohere_api_key'] = "your_cohere_api_key"

st.set_page_config(page_title="Educate Kids", page_icon=":robot:")
st.header("Hey, Ask me something & I will give out similar things.")


# embeddings = CohereEmbeddings()
embeddings = CohereEmbeddings(user_agent="MyAppName/1.0")

from langchain.document_loaders.csv_loader import CSVLoader

loader= CSVLoader(file_path='myData.csv', csv_args={
    'delimiter' : ',',
    'quotechar' : '"',
    'fieldnames' : ['words']
})

data = loader.load()

print(data)

db = FAISS.from_documents(data, embeddings)

def get_text():
    input_text = st.text_input("You: ", key= input)
    return input_text

user_input = get_text()
submit = st.button('Find similar things')

if submit:
    docs = db.similarity_search(user_input)
    st.subheader("Top Matches: ")
    st.text(docs[0].page_content)
    st.text(docs[1].page_content)
