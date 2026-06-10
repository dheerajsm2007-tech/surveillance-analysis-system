from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from part2.retriever import retrieve

SYSTEM_PROMPT = """You are a security analyst assistant.
Answer the user's question using ONLY the surveillance event context provided below.
Do not guess. Do not add any information that is not explicitly in the context.
If the context does not contain enough information to answer, respond with exactly:
"Not enough information in the footage data to answer this question."

Context:
{context}"""

HUMAN_PROMPT = "{question}"


def get_llm():
    return ChatOllama(
        model="gemma3:4b",
        temperature=0,
        num_ctx=4096,
    )


def build_context(docs) -> str:
    if not docs:
        return "No relevant events found."
    return "\n".join(f"- {doc.page_content}" for doc in docs)


def ask(question: str, video_filename: str) -> dict:
    """
    Full RAG pipeline:
    1. Retrieve relevant event docs via two-stage retrieval
    2. Build context string
    3. Send to Gemma 3 4B via Ollama
    4. Return answer + sources
    """
    docs    = retrieve(question, video_filename, top_k=5)
    context = build_context(docs)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human",  HUMAN_PROMPT),
    ])

    llm    = get_llm()
    parser = StrOutputParser()
    chain  = prompt | llm | parser

    answer = chain.invoke({"context": context, "question": question})

    return {
        "answer":  answer,
        "context": context,
        "sources": [
            {
                "track_id":   d.metadata["track_id"],
                "label":      d.metadata["label"],
                "similarity": d.metadata["similarity"],
                "text":       d.page_content,
            }
            for d in docs
        ],
    }
