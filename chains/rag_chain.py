from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def build_rag_chain(retriever):
    """
    Build a Retrieval-Augmented Generation (RAG) chain using ChatOpenAI and the provided retriever.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = ChatPromptTemplate.from_template(
        """
        You are an HR Assitant. Answer the question strictly using the context below. If the anwer is not present, say "I dont't know".
        context: {context}
        question: {question}
        """
    )

    chain = (
        {"context":retriever,
         "question":RunnablePassthrough()}
         | prompt | llm | StrOutputParser()
    ) 
    return chain
#LCEL - Langchain expression language
