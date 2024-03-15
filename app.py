# Import necessary libraries

pip install langchain[all]
pip install faiss-cpu
pip install openai
pip install huggingface_hub
pip install ransformers
gradio
pip install unstructured
pip install InstructorEmbedding
pip install pypdf

import gradio as gr
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, AIMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain import hub
from langchain.memory import ConversationBufferMemory
from langchain.schema.runnable import RunnablePassthrough

# Set up OpenAI API key
os.environ['OPENAI_API_KEY'] = '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'

# Load documents
loader = DirectoryLoader('"D:\llm\Splitted 935cmr500_2 (3)"', glob="*.txt")
loader.load()
docs = loader.load()

# Split documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3500,
    chunk_overlap=500,
    length_function=len,
    is_separator_regex=False,
)
texts = text_splitter.split_documents(docs)

# Set up embeddings
embeddings = OpenAIEmbeddings()

# Set up vectors and retriever
vectors = FAISS.from_documents(texts, embeddings)
retriever = vectors.as_retriever()

# Set up chat model
chat = ChatOpenAI(temperature=0)

# Set up conversation memory
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')

# Set up RAG prompt
rag_prompt = hub.pull("rlm/rag-prompt")

# Define the RAG chain
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | rag_prompt
    | chat
)

# Define Gradio chat interface
def chatllama(user_msg, history):
    result = rag_chain.invoke(user_msg + " Please mention regulation citation numbers or provide additional information if available; otherwise, offer a straightforward answer based on the context.")
    return result.content

# Launch Gradio interface
demo = gr.ChatInterface(
    fn=chatllama,
    title="Cannabis Chatbot",
)

# Launch the Gradio interface
demo.launch()
