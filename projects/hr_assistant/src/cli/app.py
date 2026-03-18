from dotenv import load_dotenv
from src.assistant import HRAssistant
from src.config import load_settings
from src.llm import get_llm
from src.stores import get_vector_store
from langchain_openai import OpenAIEmbeddings
from src.loader import Loader

# Load environment variables from .env 
load_dotenv()

settings = load_settings()

 
llm = get_llm(settings.provider_name, settings.provider_config.model, settings.provider_config.temperature, 
            settings.provider_config.max_tokens)

embeddings = OpenAIEmbeddings()

vector_store = get_vector_store(settings.store_name, settings.store_config.persist_directory, embeddings)

loader = Loader(file_path="./data")
documents = loader.load()
store = vector_store.get_store(f"{settings.store_name}_hr_policies", documents)

app = HRAssistant(store, llm, evidence=True)

# Scenario A: Valid prompt (Contracts)
print("Query 1: What is the core spirit of the Nestlé Human Resources Policy?")
response = app.run('What is the core spirit of the Nestlé Human Resources Policy?', [])
print(f"Response: {response}\n")