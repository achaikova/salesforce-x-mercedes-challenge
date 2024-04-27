from openai import OpenAI
import os
import streamlit as st

# app = chat_agent_executor.create_function_calling_executor(model, tools)
from agents import chain, get_answer, enter_chain
from langchain_core.messages import HumanMessage, AIMessage

RECURSION_LIMIT = 25
if 'full_messages_history' not in st.session_state:
    st.session_state['full_messages_history'] = []


def enter_chain(full_messages_history: list[str]):
    results = {
        "messages": [],
    }
    for i in range(len(full_messages_history)):
        if i % 2 == 0:
            results["messages"].append(HumanMessage(content=full_messages_history[i]))
        else:
            results["messages"].append(AIMessage(content=full_messages_history[i]))
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

    if len(ideas) > 1:
        last_index = -1 if "FINISH" not in str(ideas[-1]) else -2
    else:
        bot_answer = "I apologise, could you rephrase the question?"
        return bot_answer

    key1 = list(ideas[last_index].keys())[0]
    key2 = list(ideas[last_index][key1].keys())[0]
    try:
        bot_mesages = [i.content for i in ideas[last_index][key1][key2]]
        bot_answer = bot_mesages[-1]
    except:
        bot_answer = "I apologise, could you rephrase the question?"
    print("ANSWER:", bot_answer)
    return bot_answer


research_chain = enter_chain | chain

st.title("MASTER")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# this just needs to be here
if user_message := st.chat_input("Hello! MASTER is waiting for your message."):
    print('STREAMLIT SESSION STATE:', st.session_state)
    st.session_state.messages.append({"role": "user", "content": user_message})
    with st.chat_message("user"):
        st.markdown(user_message)
    st.session_state['full_messages_history'].append(user_message)

    with st.chat_message("assistant"):
        bot_response = get_bot_answer(st.session_state['full_messages_history'])
        st.session_state['full_messages_history'].append(bot_response)
        response = st.write(bot_response)

    st.session_state.messages.append({"role": "assistant", "content": bot_response})