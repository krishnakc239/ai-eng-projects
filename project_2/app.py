import streamlit as st

# Minimal Streamlit app that wires up a RAG pipeline using code from the
# notebook `rag_chatbot.ipynb`. This app is intentionally defensive and
# prints clear messages if dependencies or environment pieces are missing.

SYSTEM_TEMPLATE = """
You are a **Customer Support Chatbot**. Use only the information in CONTEXT to answer.
If the answer is not in CONTEXT, respond with "I'm not sure from the docs."

Rules:
1) Use ONLY the provided <context> to answer.
2) If the answer is not in the context, say: "I don't know based on the retrieved documents."
3) Be concise and accurate. Prefer quoting key phrases from the context.
4) When possible, cite sources as [source: source] using the metadata.

CONTEXT:
{context}

USER:
{question}
"""

# --- Utility: attempt imports and show friendly error messages ---
try:
    import os, glob
    from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.embeddings import SentenceTransformerEmbeddings
    from langchain.vectorstores import FAISS
    from langchain.llms import Ollama
    from langchain.chains import ConversationalRetrievalChain
    from langchain.prompts import PromptTemplate
except Exception as e:
    st.error("Missing Python packages required to run the app.\n\n" +
             "Install the project dependencies and restart Streamlit.\n\n" +
             "Quick fix inside this environment: `pip install numpy sentence-transformers langchain langchain-community faiss-cpu unstructured`\n\n" +
             f"(details: {e})")
    raise

# Small helpers
INDEX_DIR = "faiss_index"

@st.cache_data(show_spinner=False)
def load_and_chunk_pdfs(pdf_pattern="data/Everstorm_*.pdf"):
    pdf_paths = glob.glob(pdf_pattern)
    docs = []
    for p in pdf_paths:
        loader = PyPDFLoader(p)
        docs.extend(loader.load())
    if not docs:
        return []
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    return splitter.split_documents(docs)

@st.cache_resource(show_spinner=False)
def build_vectorstore(chunks):
    embeddings = SentenceTransformerEmbeddings(model_name="thenlper/gte-small")
    vectordb = FAISS.from_documents(chunks, embeddings)
    # save locally for reuse
    try:
        vectordb.save_local(INDEX_DIR)
    except Exception:
        # saving is optional; some FAISS wrappers may not implement save_local
        pass
    return vectordb

@st.cache_resource(show_spinner=False)
def load_vectorstore():
    if not os.path.exists(INDEX_DIR):
        return None
    embeddings = SentenceTransformerEmbeddings(model_name="thenlper/gte-small")
    try:
        vectordb = FAISS.load_local(INDEX_DIR, embeddings)
        return vectordb
    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def init_llm():
    # Requires Ollama server running (ollama serve) and gemma3 pulled
    try:
        llm = Ollama(model="gemma3:1b", temperature=0.1)
        return llm
    except Exception as e:
        st.warning("Could not initialize Ollama LLM. Make sure `ollama serve` is running and the model is pulled.\n" + str(e))
        return None

# --- Streamlit UI ---
st.set_page_config(page_title="RAG Chatbot — Minimal", layout="wide")
st.title("Customer Support RAG Chatbot — Minimal Streamlit UI")

with st.sidebar:
    st.header("Setup")
    if st.button("Build vector index from PDFs"):
        with st.spinner("Loading PDFs and building index..."):
            chunks = load_and_chunk_pdfs()
            if not chunks:
                st.warning("No PDFs found under data/Everstorm_*.pdf — add source PDFs or change pattern.")
            else:
                vectordb = build_vectorstore(chunks)
                st.success("Built FAISS index with %d documents" % (len(chunks)))
    if st.button("Load existing index"):
        vectordb = load_vectorstore()
        if vectordb:
            st.success("Loaded index from disk")
        else:
            st.warning("No saved index found or failed to load")
    st.markdown("---")
    st.markdown("Run Ollama server before using the LLM: `ollama serve`\nPull model: `ollama pull gemma3:1b`")

# Load vectorstore if present
vectordb = load_vectorstore()
if not vectordb:
    st.info("No vector index loaded. Build the index from the sidebar or place a saved index in `faiss_index`.")

llm = init_llm()

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

st.subheader("Ask a question")
question = st.text_input("Your question", key="question_input")

if st.button("Send") and question:
    if not vectordb:
        st.error("Vector index missing — build or load it first from the sidebar.")
    elif not llm:
        st.error("LLM not initialized — ensure Ollama server is running and model is pulled.")
    else:
        retriever = vectordb.as_retriever(search_kwargs={"k": 4})
        prompt = PromptTemplate(template=SYSTEM_TEMPLATE, input_variables=["context", "question"])
        try:
            # Preferred constructor if available
            chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, return_source_documents=True, qa_prompt=prompt)
        except Exception:
            # Fallback to more explicit constructor
            try:
                chain = ConversationalRetrievalChain(llm=llm, retriever=retriever)
            except Exception as e:
                st.error(f"Failed to construct ConversationalRetrievalChain: {e}")
                chain = None

        if chain is not None:
            with st.spinner("Generating answer..."):
                try:
                    out = chain({"question": question, "chat_history": st.session_state['chat_history']})
                except Exception as e:
                    st.error(f"Chain invocation failed: {e}")
                    out = None

            if out:
                # Many ConversationalRetrievalChain implementations return a dict with 'answer'
                answer = out.get("answer") or out.get("result") or out
                if isinstance(answer, dict):
                    # guard: if chain returned a complex object
                    answer_text = str(answer)
                else:
                    answer_text = answer

                st.markdown("**Answer:**")
                st.write(answer_text)

                # show retrieved sources if available
                docs = out.get("source_documents") if isinstance(out, dict) else None
                if docs:
                    st.markdown("**Sources / Retrieved chunks**")
                    for d in docs:
                        meta = getattr(d, "metadata", {})
                        st.write(meta.get("source", "unknown source"))
                        st.write(d.page_content[:1000])

                # append to chat history
                hist_entry = (question, answer_text)
                st.session_state['chat_history'].append(hist_entry)


# Footer: quick run instructions
st.markdown("---")
st.markdown("**Run this app:**\n1) Ensure your conda env has dependencies installed.\n2) Start Ollama: `ollama serve` and pull `gemma3:1b`.\n3) Run: `streamlit run app.py` in this project folder.")
