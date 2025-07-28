from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl
import requests

DB_FAISS_PATH = "vectorstores/db_faiss"

custom_prompt_template = """
Use the following pieces of information to answer the user's question. If you don't know the answer, please say, "I don't have information." Don't make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])
    return prompt

def load_llm():
    llm = CTransformers(
        model="llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )
    return llm

def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain

def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

def format_response(res):
    answer = res["result"].strip()
    
    # Remove source information
    formatted_response = answer

    return formatted_response



###### Chainlit Functions
import requests
import chainlit as cl

# Function to fetch a random dog image
def get_random_dog_image():
    response = requests.get("https://random.dog/woof.json")
    if response.status_code == 200:
        data = response.json()
        return data['url']
    return None

###### Chainlit Functions
@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Finding a doctor for you....")
    await msg.send()
    await msg.update()
    cl.user_session.set("chain", chain)

    # Add avatars
    # await cl.Avatar(
    #     name="Dr Batra",
    #     url="https://avatars.githubusercontent.com/u/128686189?s=400&u=a1d1553023f8ea0921fba0debbe92a8c5f840dd9&v=4",
    # ).send()

    

    # await cl.Message(
    #     content="This message should have an avatar!", author="Dr Batra"
    # ).send()

    # Fetch a random dog image and send it
    image_url = get_random_dog_image()
    if image_url:
        elements = [
            cl.Image(name="Random Dog", url=image_url)
        ]
        await cl.Message(
            content="Hiiiiiiiii!",
            elements=elements
        ).send()
    else:
        await cl.Message(
            content="Couldn't fetch a random dog image at this moment."
        ).send()


@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True,
        answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reach = True
    res = await chain.acall(message.content, callbacks=[cb])
    formatted_response = format_response(res)

    image_url = get_random_dog_image()
    if image_url:
        elements = [
            cl.Image(name="Random Dog", url=image_url)
        ]
        await cl.Message(
            content="",
            elements=elements
        ).send()
    else:
        await cl.Message(
            content="Couldn't fetch a random dog image at this moment."
        ).send()
    
    await cl.Message(content=formatted_response).send()