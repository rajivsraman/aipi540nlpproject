import streamlit as st
from tfidfchatbot import TFIDFChatbot  # Import the TF-IDF chatbot class

# Initialize TF-IDF chatbot instance
folder_path = "data/processed/tfidf"  # Ensure this folder exists with preprocessed text files
tfidf_chatbot = TFIDFChatbot(folder_path)

# Placeholder chatbot functions for now
def chatbot_2(prompt):
    return "🚧 Word2Vec chatbot is not implemented yet."

def chatbot_3(prompt):
    return "🚧 OpenAI Embeddings chatbot is not implemented yet."

# **Streamlit UI Setup**
st.title("NLP Chatbot Approach Comparison")

# Sidebar chatbot selection
st.sidebar.header("Select a Chatbot")
if "selected_bot" not in st.session_state:
    st.session_state.selected_bot = "TF-IDF"  # Default

selected_bot = st.sidebar.radio("Choose a chatbot approach:", ["TF-IDF", "Word2Vec", "OpenAI Embeddings"])

st.session_state.selected_bot = selected_bot  # Store selection

st.write(f"### Currently Using: {st.session_state.selected_bot}")

# **User Query Input**
user_input = st.text_input("Enter your prompt:", "")

# **Process and Display Response**
if user_input:
    if st.session_state.selected_bot == "TF-IDF":
        response = tfidf_chatbot.chatbot(user_input)
    elif st.session_state.selected_bot == "Word2Vec":
        response = chatbot_2(user_input)
    else:
        response = chatbot_3(user_input)
    
    st.write("### Chatbot Response:")
    st.write(response)
