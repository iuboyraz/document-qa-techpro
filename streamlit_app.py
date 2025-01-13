import streamlit as st
from openai import OpenAI
import pandas as pd
import PyPDF2
from io import StringIO
from dotenv import load_dotenv 
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
import os
from langchain_openai import ChatOpenAI
import random
from streamlit.components.v1 import html
import time

load_dotenv()

# Show title and description.

# Set up Streamlit page configuration
st.set_page_config(page_title="Mentoring Chatbot", page_icon="🤖", layout="wide")

# Add social media icons and links
st.markdown("""
    <div style="text-align: center; padding-bottom: 10px;">
        <a href="https://www.youtube.com/@TechProEducationUS" target="_blank"><img src="https://upload.wikimedia.org/wikipedia/commons/b/b8/YouTube_Logo_2017.svg" width="30" style="margin-right: 10px;"></a>
        <a href="https://api.whatsapp.com/send/?phone=%2B15853042959&text&type=phone_number&app_absent=0" target="_blank"><img src="https://upload.wikimedia.org/wikipedia/commons/6/6b/WhatsApp.svg" width="30" style="margin-right: 10px;"></a>
        <a href="https://t.me/joinchat/HH2qRvA-ulh4OWbb" target="_blank"><img src="https://upload.wikimedia.org/wikipedia/commons/8/82/Telegram_logo.svg" width="30" style="margin-right: 10px;"></a>
        <a href="https://www.instagram.com/techproeducation/" target="_blank"><img src="https://upload.wikimedia.org/wikipedia/commons/a/a5/Instagram_icon.png" width="30" style="margin-right: 10px;"></a>
        <a href="https://www.facebook.com/techproeducation" target="_blank"><img src="https://upload.wikimedia.org/wikipedia/commons/5/51/Facebook_f_logo_%282019%29.svg" width="30" style="margin-right: 10px;"></a>
        <a href="https://x.com/techproedu" target="_blank"><img src="https://abs.twimg.com/icons/apple-touch-icon-192x192.png" width="30" style="margin-right: 10px;"></a>
        <a href="https://www.linkedin.com/school/techproeducation/" target="_blank"><img src="https://upload.wikimedia.org/wikipedia/commons/0/01/LinkedIn_Logo.svg" width="30" style="margin-right: 10px;"></a>    
    </div>
""", unsafe_allow_html=True)

# Display the image at the top of the page with a clickable link
st.markdown("""
    <div style="text-align: center;">
        <a href="https://www.techproeducation.com/" target="_blank">
            <img src="https://yt3.googleusercontent.com/G16n52mulzjmDxMETa4OR5tPlYHeg-ZVkDqxnTqxjSy49ZOR07TJwJ_1izlPQzzWCJMGciRRAEE=w1707-fcrop64=1,00005a57ffffa5a8-k-c0xffffffff-no-nd-rj" 
            alt="Techpro Education Cover" width="100%" style="border-radius: 10px;"/>
        </a>
    </div>
""", unsafe_allow_html=True)

st.title("Chat with Techpro Education 💬")

# Add sidebar with information
with st.sidebar:
    st.title("About")
    st.markdown("""
    Company:    
    
    Techproeducation provides quality online IT courses and coding bootcamps with reasonable prices to prepare individuals for next-generation jobs from beginners to IT professionals. 
    
    We offer cutting-edge programs used today by leading corporations.

    Contact:
    
    https://www.techproeducation.com/
    info@techproeducation.com            
    +1 585 304 29 59       
    New York City, NY USA
                
    Programs:
    
    - FREE ONLINE IT COURSES                
    - AUTOMATION ENGINEER                
    - SOFTWARE DEVELOPMENT                
    - CLOUD ENGINEERING & SECURITY                
    - DATA SCIENCE                
    - DIGITAL MARKETING
    """)


# Upload Excel file
excel_file = r"/workspaces/document-qa-techpro/chatbot_techpro_final_questions.xlsx"
data = pd.read_excel(excel_file)

# Convert questions and answers to a list
questions = data['Questions'].tolist()  
answers = data['Answers'].tolist()      

# Create document objects only after data is read
documents = [Document(page_content=f"{row['Questions']}\n{row['Answers']}") for _, row in data.iterrows()]

# Determine embedding model
model_name = "BAAI/bge-base-en" 
# paraphrase-multilingual-MiniLM-L12-v2
# sentence-transformers/multi-qa-distilbert-cos-v1
# all-MiniLM-L6-v2
encode_kwargs = {'normalize_embeddings': True} 

# Establish embeddings model
bge_embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs={'device': 'cpu'},
    encode_kwargs=encode_kwargs
)

# Create database
persist_directory = 'db'
if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)  

# Establish vector database (vectorstore)
vectordb = Chroma.from_documents(documents=documents, # if "text_splitter()" is used use "documents=texts"
                                 collection_name="rag-chroma",
                                 embedding=bge_embeddings,
                                 persist_directory=persist_directory)

retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# Initialize message history
if "messages" not in st.session_state:  
    st.session_state.messages = [
        {"role": "assistant", "content": "Merhaba Ben Techie 🤖. Data Science, Mentoring ve IT alanındaki sorularınıza cevap vermeye çalışacağım."}]

# Wrap the prompt in a function
def prompt_fn(query: str, context: str) -> str:
    return f"""
    You are an experienced IT staff having expertise in Data Science, Information Technology, 
    Programming Languages, Statistics, Data Visualization, Cloud Systems, Deployment, Project Management and its tools, 
    Communication systems, Web sites for remote working and Mentoring. 
    If the user's query matches any question from the database, return the corresponding answer directly. 
    If the query is within the context, generate only one concise and accurate response in Turkish strictly based on the provided context. 
    If the query is outside the context, respond only with "Kapsam dışı sorduğunuz sorulara cevap veremiyorum."

    Context: {context}
    
    User's question: {query}"""

# LLM model
@st.cache_resource
def load_llm():
    return ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, max_tokens=150)

llm = load_llm()

@st.cache_resource
def create_rag_chain():
    from langchain_core.runnables import RunnableLambda
    prompt_runnable = RunnableLambda(lambda inputs: prompt_fn(inputs["query"], inputs["context"]))
    return prompt_runnable | llm | StrOutputParser()

rag_chain = create_rag_chain()

# Typewriter effect function (Browser-only version)
def typewriter_effect(text, delay=0.05):
    """Simulate a typewriter effect by progressively displaying characters on the same line."""
    display_text = ""
    placeholder = st.empty()  # Create a placeholder to update the content
    for char in text:
        display_text += char
        placeholder.markdown(f"{display_text}")  # Update the display progressively
        time.sleep(delay)

# Generate response
def generate_response(query):
    # Search the database for the exact answer
    for _, row in data.iterrows():
        if query.strip().lower() in row["Questions"].strip().lower():
            suggestions = "\n".join([f"- {q}" for q in random.sample(questions, k=3)])  
            return row["Answers"], suggestions  

    results = retriever.get_relevant_documents(query)[:3]  
    context = "\n".join([doc.page_content for doc in results])
    inputs = {"query": query, "context": context}
    response = rag_chain.invoke(inputs)

    related_questions = random.sample(questions, k=3)  
    suggestions = "\n".join([f"- {q}" for q in related_questions])

    return response, suggestions

# Display chat messages from session state
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Handle user query input
if query := st.chat_input("Your question"):
    st.session_state["messages"].append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    # Generate response with typewriter effect
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response, suggestions = generate_response(query)
            
            # Apply the typewriter effect for response and suggestions
            typewriter_effect(response)  # Simulating typewriter effect for response
            st.markdown("### Şu soruları sorabilirsiniz: ")
            typewriter_effect(suggestions)  # Simulating typewriter effect for suggestions

            # Store assistant's response in session state
            st.session_state["messages"].append({"role": "assistant", "content": response})

# Add robot avatar to the right of chat input with the name "Techie"
avatar_html = """
<style>
.robot-avatar {
    position: fixed;
    right: 30px;
    bottom: 50px;
    width: 80px;
    height: 80px;
    background: linear-gradient(45deg, #32CD32, #FFFFFF);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    animation: float 2s ease-in-out infinite;
    box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.3);
    text-align: center;
    font-family: Arial, sans-serif;
}

.robot-avatar img {
    width: 50px;
    height: 50px;
    border-radius: 50%;
}

.robot-name {
    position: absolute;
    top: -25px;
    left: 10px;
    font-size: 16px;
    font-weight: bold;
    color: #32CD32;
    background-color: white;
    padding: 2px 10px;
    border-radius: 50px;    
    text-align: center;
}

@keyframes float {
    0%, 100% {
        transform: translateY(-5px);
    }
    50% {
        transform: translateY(5px);
    }
}
</style>
<div class="robot-avatar">
    <div class="robot-name">Techie </div>
    <img src="https://cdn-icons-png.flaticon.com/512/4712/4712109.png" alt="Robot Avatar">
</div>
"""
html(avatar_html, height=200)
