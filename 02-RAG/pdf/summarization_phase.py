from langchain_google_genai import ChatGoogleGenerativeAI

def run_summarization_phase(relavent_chunks, system_prompt, llm_model, user_question):
    chunks_text = "\n\n".join([
        chunk.page_content if hasattr(chunk, "page_content") else str(chunk)
        for chunk in relavent_chunks
    ])
    prompt = system_prompt.format(relevant_chunks=chunks_text)
    llm = ChatGoogleGenerativeAI(
        model=llm_model,
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    messages = [
        ("human", user_question),
        ("system", prompt)
    ]
    ai_msg = llm.invoke(messages)
    print("AI Response:", ai_msg.content)
    return ai_msg.content
