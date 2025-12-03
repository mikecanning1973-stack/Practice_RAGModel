import streamlit as st
from query_db import query_rag
from dotenv import load_dotenv

# Load environment variables
load_dotenv('configuration.env')

# Set up page configuration
st.set_page_config(
    page_title="Engineering Calculation Search and Find",
    page_icon="🤖",
    layout="centered"
)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Add custom CSS
st.markdown("""
<style>
    .user-message {
        background-color: #e6f3ff;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .bot-message {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Display header
st.title("🔩 Engineering Concept Search ⚙")
st.write("Hello, I am Aero Space. I am your guide to finding and explaining calculations that engineers need to know and understand. Just ask me a question and I will do my best to find you he answer!")

# Display chat history
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"""<div class="user-message">👤 You: {message["content"]}</div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class="bot-message">🤖 Bot: {message["content"]}</div>""", unsafe_allow_html=True)

# Get user input
if prompt := st.chat_input("Ask a question about engineering calculations and concepts..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Get bot response
    with st.spinner("Thinking... 🤔"):
        try:
            response = query_rag(prompt)
            # Add bot response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Force a rerun to update the chat display
            st.rerun()
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            if "model" in str(e).lower():
                st.info("💡 Tip: Make sure your Ollama service is running!")

# Add sidebar with information
with st.sidebar:
    st.title("ℹ️ About")
    st.markdown("""
    This chatbot uses RAG (Retrieval Augmented Generation) to answer questions about your documents.
    
    The bot will:
    1. Search through your loaded documents
    2. Find all the relevant information
    3. Generate an answer based on the the question that was asked.
    
    To get started:
    1. Make sure your documents are loaded
    2. Ensure Ollama is running (ollama serve in the command prompt)
    3. Ask your questions!
    """)
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()