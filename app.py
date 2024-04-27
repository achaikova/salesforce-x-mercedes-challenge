from openai import OpenAI
import os
import streamlit as st

#app = chat_agent_executor.create_function_calling_executor(model, tools)
from agents import chain, get_answer, enter_chain
from langchain_core.messages import HumanMessage

RECURSION_LIMIT = 25


def enter_chain(full_messages_history: list[str]):
    results = {
        "messages": [],
    }
    for i in range(len(full_messages_history)):
        if i % 2 == 0:
            results["messages"].append(HumanMessage(content=full_messages_history[i]))
        else:
            results["messages"].append(HumanMessage(content=full_messages_history[i], is_user=False))
    return results

def get_bot_answer(full_messages_history):
    print('START NEW ITERATION')
    print(full_messages_history)
    ideas = []
    for s in research_chain.stream(
        full_messages_history, {"recursion_limit": RECURSION_LIMIT}
    ):
        if "__end__" not in s:
            print(s)
            ideas.append(s)
            print("---")

    j = 0
    while 'conversation' not in ideas[-j]:
        j += 1
    bot_answer = ideas[-j]['conversation']['messages'][0].content
    print("ANSWER:", bot_answer)
    return bot_answer


research_chain = enter_chain | chain


full_messages_history = []

st.title("MASTER")
with open("key.txt", 'r') as f:
    os.environ["OPENAI_API_KEY"] = f.read()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# this just needs to be here
if user_message := st.chat_input("What is up?"):
    print('STREAMLIT SESSION STATE:', st.session_state)
    st.session_state.messages.append({"role": "user", "content": user_message})
    with st.chat_message("user"):
        st.markdown(user_message)
    full_messages_history.append(user_message)

    with st.chat_message("assistant"):
        bot_response = get_bot_answer(full_messages_history)
        full_messages_history.append(bot_response)
        response = st.write(bot_response)
        
    st.session_state.messages.append({"role": "assistant", "content": bot_response})