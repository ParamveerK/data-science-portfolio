import pandas as pd
import ollama
import chromadb
import streamlit as st

def create_prompt(query, chat_history = []):
    chat_context = ''
    if len(chat_history) > 0:
        for row in chat_history:
            for usr, msg in row.items():
                chat_context += f'{msg} | '

    contexts = collection.query(query_texts=query, n_results=10)
    context = ''
    for i, chunk in enumerate(contexts['metadatas'][0]):
        context = context +  f'{chunk["document"]}: {contexts["documents"][0][i]} \n'
    
    prompt = f"""
    Here is the chat context to work with: {chat_context}.
    You are chatting to someone looking to learn more about TfL consultations, you have access to their consultation website data and documents. Answer the following question, drawing on all context that has been provided that is relevant. If new context is not relevant to the question, don't mention it. If it is, give citations. Be direct in your answer.
    Question: {query} 
    Citations: {context}
    """
    return prompt

def run_prompt(prompt):
    response = ollama.chat(
        model='llama3.1:8b',
        messages=[{'role':'user', 'content': prompt}])

    return response['message']['content']

chroma_client = chromadb.PersistentClient("/Users/paramveerkumar/Documents/GitHub/data-science-portfolio/tfl-rag-pipeline/chroma")
collection = chroma_client.get_collection(name='consultations')

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Let's start chatting! 👇"}]

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything (about TfL)!"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    final_prompt = create_prompt(prompt, st.session_state.chat_history)

    response = run_prompt(final_prompt)

    st.session_state.chat_history.append({'query':prompt,'response': response})

    with st.chat_message('assistant'):
        st.markdown(response)
    
    st.session_state.messages.append({'role':'assistant','content':response})