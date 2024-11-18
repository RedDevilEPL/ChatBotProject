import os
from typing import List

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

from langchain.docstore.document import Document
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
import chainlit as cl
from chainlit.types import ThreadDict
from chainlit import AskUserMessage, Message, on_chat_start
from openai import AsyncOpenAI


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
client = AsyncOpenAI()

# Instrument the OpenAI client
cl.instrument_openai()
@cl.password_auth_callback
def auth_callback(username: str, password: str):
    if (username, password) == ("admin", "admin"):
        return cl.User(
            identifier="admin", metadata={"role": "admin", "provider": "credentials"}
        )
    else:
        return None
    
@cl.on_chat_start
async def on_chat_start():
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload a text file to begin!",
            accept=["text/plain"],
            max_size_mb=20,
            timeout=60,
        ).send()

    file = files[0]
    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()

    with open(file.path, "r", encoding="utf-8") as f:
        text = f.read()

    texts = text_splitter.split_text(text)
    metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]

    embeddings = OpenAIEmbeddings()
    docsearch = await cl.make_async(Chroma.from_texts)(texts, embeddings, metadatas=metadatas)

    message_history = ChatMessageHistory()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    chain = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(model_name="gpt-4o-mini", temperature=0, streaming=True),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )

    msg.content = f"Processing `{file.name}` done. You can now ask questions!"
    await msg.update()
    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler()

    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["source_documents"]

    text_elements = []
    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            text_elements.append(cl.Text(content=source_doc.page_content, name=f"source_{source_idx}", display="side"))
        source_names = [text_el.name for text_el in text_elements]
        answer += f"\\nSources: {', '.join(source_names)}" if source_names else "\\nNo sources found"

    await cl.Message(content=answer, elements=text_elements).send()
