from langchain_openai import OpenAIEmbeddings
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools.retriever import create_retriever_tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from src.stores.vector_store_adapter import VectorStoreAdapter
from langchain_core.language_models.chat_models import BaseChatModel


class _CitationRetriever:
    """Wraps a retriever to prefix each chunk with document_name and chunk_number if enabled"""

    def __init__(self, retriever, include_citations: bool):
        self.retriever = retriever
        self.include_citations = include_citations

    def invoke(self, query: str, *args, **kwargs):
        if not self.include_citations:
            result = self.retriever.invoke(query, *args, **kwargs)
        else:
            docs = self.retriever.invoke(query, *args, **kwargs)
            result = [
                Document(
                    page_content=f"[Source: {d.metadata.get('document_name', 'unknown')}, Chunk {d.metadata.get('chunk_number', '?')}]\n{d.page_content}",
                    metadata=d.metadata,
                )
                for d in docs
            ]

        return result

class HRAssistant:
    def __init__(
        self,
        vector_store: VectorStoreAdapter,
        llm: BaseChatModel,
        verbose: bool = False,
        evidence: bool = False,
    ):
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = vector_store
        self.llm = llm

        # Enables lanchain verbosity for debugging prompts and response
        self.verbose = verbose

        # Displays evidence supporting the response if enabled
        self.evidence = evidence

        # Standard Guardrail: LLM-as-a-Judge
        self.guard_chain = self._setup_intent_guard()
        self.agent_executor = self._setup_agent()

    def _setup_intent_guard(self):
        """Standard internal guardrail to verify topic alignment """
        guard_prompt = ChatPromptTemplate.from_template("""
            Classify the user intent. 
            Allowed Topics: 
            1. HR policies.
            2. HR questions.
            3. Questions about Nestlé policies.

            If the intent is about these topics, reply 'SAFE'. 
            Otherwise, reply 'UNSAFE'.

            User Input: {user_input}
        """)
        return guard_prompt | self.llm | StrOutputParser()

    def _setup_agent(self):
        base_retriever = self.vectorstore.as_retriever()
        retriever = _CitationRetriever(base_retriever, include_citations=self.evidence)
        hr_search_tool = create_retriever_tool(
            retriever,
            "hr_policy_search",
            "Search for company HR policies, HR information, and Nestlé policies."
        )

        evidence_instruction = (
            " When citing evidence, include the source in parentheses: (Source: document_name, Chunk N)."
            if self.evidence
            else ""
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a senior HR assistant.
             Use the provided context to answer questions about HR policies, HR information, and Nestlé policies.
             If the information is not in the context, say you do not know.
             NEVER answer off-topic questions.""" + evidence_instruction),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])

        agent = create_tool_calling_agent(self.llm, [hr_search_tool], prompt)

        return AgentExecutor(agent=agent, tools=[hr_search_tool], verbose=self.verbose, 
                            early_stopping_method="generate", max_iterations=5)

    def run(self, user_input, history):
        # Intent Check (Standard Guardrail logic)
        intent_check = self.guard_chain.invoke({"user_input": user_input})
        
        if "UNSAFE" in intent_check.upper():
            result =  "I am sorry, but I can only answer HR questions reguarding Nestle."
        else:
            # Core Logic with high-precision settings
            try:
                response = self.agent_executor.invoke({"input": user_input, "chat_history": history})
                result = response['output']
            except Exception as e:
                result = f"An internal error occurred while processing your legal request. {e}"

        return result