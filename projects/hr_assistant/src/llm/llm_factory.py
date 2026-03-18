from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.language_models.chat_models import BaseChatModel

def get_llm(provider: str, model: str, temperature: float = 0.2, max_tokens: int = 512) -> BaseChatModel:
    result = None
    if provider == "openai":
        result = ChatOpenAI(model=model, temperature=temperature, max_tokens=max_tokens)
    elif provider == "groq":
        result = ChatGroq(model=model, temperature=temperature, max_tokens=max_tokens)
    else:
        raise ValueError(f"Invalid provider: {provider}")

    return result