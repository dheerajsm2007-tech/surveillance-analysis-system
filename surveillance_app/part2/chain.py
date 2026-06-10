from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from part2.retriever import retrieve, retrieve_summary

SYSTEM_PROMPT = """You are a security analyst assistant reviewing surveillance footage data.

Answer the user's question using ONLY the information in the context below.
Do not guess. Do not add any information not explicitly stated in the context.
If the context does not contain enough information to answer, respond with exactly:
"Not enough information in the footage data to answer this question."

When the question asks for a count, total, or "how many" — count the distinct items
listed in the context and state the number explicitly.
When the question asks for a maximum or minimum — compare the values in the context
and identify the correct one.
When the question asks about anomalies or suspicious activity — look for an
ANOMALY FLAGS section in the context and report it; if none exists say so.

Context:
{context}"""

HUMAN_PROMPT = "{question}"


def get_llm():
    return ChatOllama(
        model="gemma3:4b",
        temperature=0,
        num_ctx=4096,
    )


def build_context(docs, summary_text: str = "") -> str:
    parts = []
    if summary_text:
        parts.append(summary_text)
    if docs:
        parts.append("INDIVIDUAL EVENT RECORDS:")
        parts.extend(f"- {doc.page_content}" for doc in docs)
    return "\n".join(parts) if parts else "No relevant events found."


def ask(question: str, video_filename: str) -> dict:
    """
    Full RAG pipeline:
    1. Retrieve top-k matching event docs (increased to 8 for better coverage)
    2. Inject pre-built summary + anomaly flags as context header
    3. Send to Gemma 3 4B via Ollama
    4. Return answer + sources
    """
    docs         = retrieve(question, video_filename, top_k=8)
    summary_text = retrieve_summary(video_filename)
    context      = build_context(docs, summary_text)

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
