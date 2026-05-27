from google import genai

from app.config import GOOGLE_API_KEY, LLM_MODEL

client = genai.Client(api_key=GOOGLE_API_KEY)

def generate_answer(query, retrieved_chunks):

    context = "\n\n".join([
        f"Source: {chunk['source']} (Page {chunk['page_number']})\n{chunk['text']}"
        for chunk in retrieved_chunks
    ])

    prompt = f"""
    You are a helpful document question answering assistant.

    Your Job:
    - Answer the user's question using only the provided document content.
    - If the answer is not contained in the provided content, say "I don't know".
    - Do not use outside knowledge.
    - Keep answers concise.

    Context:
    {context}

    User Question:
    {query}

    Answer:
    """

    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt
    )

    return response.text