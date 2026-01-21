import streamlit as st
import os
import json
from sentinel_rag.core.engine import SupportBot

# Set page config
st.set_page_config(page_title="ChatSolveAI Support Center", page_icon="🤖")

st.title("🤖 ChatSolveAI Support Center")

# Sidebar for configuration or debugging
with st.sidebar:
    st.header("Configuration")
    kb_path = st.text_input("Knowledge Base Path", value="chatbot_responses.json")
    if st.button("Reload Knowledge Base"):
        if "bot" in st.session_state:
            with st.spinner("Reloading Knowledge Base..."):
                st.session_state.bot.load_knowledge_base(kb_path)
            st.success("Reloaded!")

# Initialize bot only once
if 'bot' not in st.session_state:
    with st.spinner("Initializing Bot..."):
        # Check for API Key
        if not os.getenv("GEMINI_API_KEY"):
            st.error("GEMINI_API_KEY not found. Please set it in .env file.")
            st.stop()
            
        bot = SupportBot()
        # Try to load default KB if exists
        if os.path.exists(kb_path):
            bot.load_knowledge_base(kb_path)
        else:
            st.warning(f"Knowledge base not found at {kb_path}. Please check the path.")
            
        st.session_state.bot = bot

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("How can we help you today?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = st.session_state.bot.get_response(prompt)
            response_text = result['answer']
            st.markdown(response_text)
            
            # Show debug info in expander
            with st.expander("Debug Info"):
                st.json(result)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response_text})
