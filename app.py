import chainlit as cl
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from main import search_reddit_comments
from langchain import PromptTemplate, LLMChain
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory
import warnings
import numpy as np
warnings.filterwarnings("ignore")

text_splitter = None
embeddings = None
repo_id = None
llm = None
memory = None


def cosine_similarity(vec1, vec2):
    a = np.linalg.norm(vec1)
    b = np.linalg.norm(vec2)
    return np.dot(vec1, vec2)/(a*b)


@cl.on_chat_start
def on_chat_start():
    global text_splitter
    global embeddings
    global repo_id
    global llm
    global memory

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-mpnet-base-v2')
    repo_id = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
    llm = HuggingFaceEndpoint(
        repo_id=repo_id, max_length=4096, temperature=0.2)

    message_history = ChatMessageHistory()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )


@cl.on_message
async def main(message: cl.Message):

    query = message.content

    past_messages = memory.chat_memory.messages
    related_content = []
    for past_message in past_messages:
        past_query_embedding = embeddings.embed_query(
            past_message)
        current_query_embedding = embeddings.embed_query(query)
        similarity = cosine_similarity(
            past_query_embedding, current_query_embedding)
        if similarity > 0.8:
            related_content.append(past_message['answer'])

    if related_content:
        combined_related_content = " ".join(related_content)
    else:
        combined_related_content = ""
    try:
        text = await search_reddit_comments(query)
        text_chunks = text_splitter.split_text(text)
        if text_chunks:
            db = await cl.make_async(FAISS.from_texts)(text_chunks, embeddings)
            docs = db.similarity_search(query)
            # print(docs)
            if docs:
                template = """
                    Question: Based on the following question "{question}" summarize the best from the given answer and the related content and not frame your own answer. You can use examples from the question and frame some stories also. 
                    Related Content: {related_content}
                    Answer: {docs}
                """

                prompt = PromptTemplate(template=template, input_variables=[
                                        "question", "related_content", "docs"])

                chain = LLMChain(llm=llm, prompt=prompt)

                inputs = {
                    "question": query,
                    "related_content": combined_related_content,
                    "docs": docs
                }
                result = chain.run(inputs)
                if result:
                    await cl.Message(
                        content=result
                    ).send()

                    memory.chat_memory.add_message(result)
                else:
                    await cl.Message(content="Oops! No result found for your question").send()
            else:
                await cl.Message(content="I can't answer that question").send()
        else:
            await cl.Message(content="No matching result found for that query").send()
    except Exception as e:
        print("Error = ", e)
        await cl.Message(content="No matching result found for that query").send()
