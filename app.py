import streamlit as st
from tfidfchatbot import TFIDFChatbot
from w2vchatbot import Word2VecChatbot

# Initialize chatbot instances
folder_tfidf = "data/processed/tfidf"
folder_word2vec = "data/processed/word2vec"

tfidf_chatbot = TFIDFChatbot(folder_tfidf)
word2vec_chatbot = Word2VecChatbot(folder_word2vec)

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
        response = word2vec_chatbot.chatbot(user_input)
    else:
        response = chatbot_3(user_input)
    
    st.write("### Chatbot Response:")
    st.write(response)
