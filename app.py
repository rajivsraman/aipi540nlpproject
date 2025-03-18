import streamlit as st

# chatbot functions
def chatbot_1(prompt):
    return f"TF-IDF Model response to: {prompt}"

def chatbot_2(prompt):
    return f"Word2Vec Model response to: {prompt}"

def chatbot_3(prompt):
    return f"OpenAI Embeddings Model response to: {prompt}"

# streamlit UI
st.title("NLP Chatbot Approach Comparison")

# sidebar for chatbot selection
st.sidebar.header("Select a Chatbot")
if "selected_bot" not in st.session_state:
    st.session_state.selected_bot = "TF-IDF"

if st.sidebar.button("Use TF-IDF"):
    st.session_state.selected_bot = "TF-IDF"
if st.sidebar.button("Use Word2Vec"):
    st.session_state.selected_bot = "Word2Vec"
if st.sidebar.button("Use OpenAI Embeddings"):
    st.session_state.selected_bot = "OpenAI Embeddings"

st.write(f"### Currently Using: {st.session_state.selected_bot}")

# user prompt
user_input = st.text_input("Enter your prompt:", "")

if user_input:
    if st.session_state.selected_bot == "TF-IDF":
        response = chatbot_1(user_input)
    elif st.session_state.selected_bot == "Word2Vec":
        response = chatbot_2(user_input)
    else:
        response = chatbot_3(user_input)
    
    st.write("### Chatbot Response:")
    st.write(response)
