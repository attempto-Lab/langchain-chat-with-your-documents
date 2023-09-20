import os
import streamlit as st

from docchat import DocChat, Document, ChatResponse

# please note that the location of this function has changed multiple times in the last versions of streamlit
from streamlit.runtime.scriptrunner import get_script_run_ctx
# if run directly, print a warning
ctx = get_script_run_ctx()
if ctx is None:
    print("************")
    print("PLEASE NOTE: run this app with `streamlit run docchat_streamlit_ui.py`")
    print("************")
    exit(1)

print("Starting")

# we only want to create this doc_chat once, so we keep it in the session state
if st.session_state.get("doc_chat", None) is None:
    print("Creating DocChat instance")
    st.session_state["doc_chat"] = DocChat()

doc_chat = st.session_state.get("doc_chat", None)


st.set_page_config(page_title="attempto Lab Chat - An LLM-powered Streamlit app")
st.title('💬 attempto Lab Chat App')

with st.sidebar:
    st.header('Document')

    uploaded_file = st.file_uploader("Upload your Document", type=["pdf"])
    # be aware, that this is not reset when you reload the page, so you will always see the last uploaded file
    if uploaded_file is not None and (doc_chat.doc is None or not uploaded_file.name == doc_chat.doc.name):
        local_file = os.path.join("docs", uploaded_file.name)
        with open(local_file, "wb") as f:
            f.write(uploaded_file.getbuffer())
        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                try:
                    embeddings_count = doc_chat.process_doc(Document(uploaded_file.name, local_file))
                    st.success(f"Using document {uploaded_file.name}")
                    st.success(f"Created {embeddings_count} chunks/embeddings")
                except Exception as e:
                    st.error(f"Error: {e}")

# Store AI generated responses
if "messages" not in st.session_state.keys():
    st.session_state["messages"] = [{"role": "assistant", "content": "I'm attempto Lab Chat, How may I help you?"}]

# Display existing chat messages
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# Function for generating LLM response
def generate_response(question: str) -> ChatResponse:
    print(f"generate_response: {question}")
    try:
        return doc_chat.get_response(question)
    except Exception as ex:
        error_msg = f"Sorry, an error occurred: {ex}"
        return ChatResponse(answer=error_msg)


# Prompt for user input and save
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# If last message is not from assistant, we need to generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    # Call LLM and process response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(prompt)
            st.write(response.answer)
            if response.db_source_chunks not in [None, []]:
                sources = [x.metadata for x in response.db_source_chunks]
                # unique sources
                unique_sources = [dict(s) for s in set(frozenset(d.items()) for d in sources)]
                # sorted by page
                sources = sorted(unique_sources, key=lambda s: s['page'])
                st.dataframe(data=sources, column_config={"source": {"label": "Document"}, "page": {"label": "Page"}})
                st.caption('Sources for the answer above :sunglasses:')

    message = {"role": "assistant", "content": response.answer}
    st.session_state.messages.append(message)
