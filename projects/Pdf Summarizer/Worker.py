from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
from dotenv import load_dotenv

class Worker:
    def __init__(self):
        # Load environment variables
        self.load_env()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.question_key = "question"
        self.chat_history_key = "chat_history"
        
        # Initialize the LLM with ChatOpenAI for the gpt-4o-mini model
        self.llm_hub = ChatOpenAI(
            model="gpt-4o-mini",  # Specify gpt-4o-mini model
            temperature=0.1,
            max_tokens=256,
            api_key=self.openai_api_key
        )

        # Initialize OpenAI embeddings
        self.embeddings = OpenAIEmbeddings(api_key=self.openai_api_key)

        # Initialize conversation retrieval chain (vector store using Chroma)
        self.db = None  # Placeholder, will be set when processing documents
        self.conversation_retrieval_chain = None  # Placeholder for the chain
        self.chat_history = []

    def load_env(self):
        """Load environment variables from .env file"""
        load_dotenv()

    def process_document(self, document_path):
        """Load, split, and index a document into a Chroma DB"""
        # Load the document
        loader = PyPDFLoader(document_path)
        documents = loader.load()

        # Split the document into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
        texts = text_splitter.split_documents(documents)

        # Create a Chroma database with the document chunks
        self.db = Chroma.from_documents(texts, embedding=self.embeddings)

        # Create a retrieval chain
        self.conversation_retrieval_chain = RetrievalQA.from_chain_type(
            llm=self.llm_hub,
            chain_type="stuff",
            retriever=self.db.as_retriever(search_type="mmr", search_kwargs={'k': 6, 'lambda_mult': 0.25}),
            return_source_documents=False,
            input_key=self.question_key
        )

    def process_prompt(self, prompt):
        """Process a user prompt using the retrieval chain"""
        
        if not self.conversation_retrieval_chain:
            return "No document has been processed yet. Please upload a document first."
        
        # Query the chain with the user prompt
        output = self.conversation_retrieval_chain.invoke({
            self.question_key: prompt,
            self.chat_history_key: self.chat_history
        })
        answer = output["result"]

        # Update chat history
        self.chat_history.append((prompt, answer))

        return answer

