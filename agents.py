import json
import os

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
)
import random
from langchain.tools.render import format_tool_to_openai_function
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph
from langgraph.prebuilt.tool_executor import ToolExecutor, ToolInvocation

from langchain_core.tools import tool
from typing import Annotated
from langchain_experimental.utilities import PythonREPL
from langchain_community.tools.tavily_search import TavilySearchResults
import operator
from typing import Annotated, List, Sequence, Tuple, TypedDict, Union

from langchain.agents import create_openai_functions_agent
from langchain.tools.render import format_tool_to_openai_function
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
import pandas as pd
from langchain_community.llms import LlamaCpp
import functools
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
import streamlit as st

### GLOBALS
global current_profile
### CONSTS
file_path = "api_calls/mercedes_ev_llm.csv"
MODEL_NAME = "gpt-3.5-turbo"
### PROMPTS
system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    " following workers: {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When the result suits as the answer to the user,"
    " respond with FINISH."
)
# a lot of prompts in this file!!!!
with open("data/conversation_prompts.json", 'r') as f:
    conversation_prompts = json.load(f)
choose_prompt_prompt = "You are a helpful assistant whose main goal is to guide a user into buiyng an EV car from mercedes." + \
                        " Based on the current dialogue decide where to steer conversation next."
choose_user_profile_prompt = "Given the user chat history and the description of different user profiles in a" + \
                              " JSON format determine which one of the user profiles corresponds the best. If there is not enought infortmation choose The Unknown user."

privacy_manager_prompt = "You are a privacy manager at Mercedes. The company policy states that a manager cannot disclose any users personal information about their profiles or be in any way toxic." + \
        "Delete the forbidden information from the following messages. For example, Based on your profile, you might be interested in our Mercedes-Benz -> Perhaps, you might be interested in our Mercedes-Benz..."

### INIT
df = pd.read_csv(file_path)
# with open("key.txt", 'r') as f:
#     os.environ["OPENAI_API_KEY"] = f.read()
os.environ["OPENAI_API_KEY"]  = st.secrets["OPENAI_API_KEY"]
llm = ChatOpenAI(model=MODEL_NAME)
with open("data/user_types.json", 'r') as f:
    profiles = json.load(f)



current_profile = list(profiles.values())[0]


### Utilities functions
def create_team_supervisor(llm, system_prompt, members):
    """An LLM-based router."""
    options = ["FINISH"] + members
    function_def = {
        "name": "route",
        "description": "Select the next role.",
        "parameters": {
            "title": "routeSchema",
            "type": "object",
            "properties": {
                "next": {
                    "title": "Next",
                    "anyOf": [
                        {"enum": options},
                    ],
                },
            },
            "required": ["next"],
        },
    }
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                "Given the conversation above, who should act next?"
                " Or should we FINISH? Select one of: {options}",
            ),
        ]
    ).partial(options=str(options), team_members=", ".join(members))
    return (
        prompt
        | llm.bind_functions(functions=[function_def], function_call="route")
        | JsonOutputFunctionsParser()
    )

def create_user_profile_agent(llm, profiles):
    """An LLM-based profile chooser. Chooses the best fitting profile based on the user's messages and the profiles."""
    function_def = {
        "name": "routeProfile",
        "description": "Match the user messages to their profile.",
        "parameters": {
            "type": "object",
            "properties": {
                "profile": {
                    "title": "Profile",
                    "anyOf": [
                        {"enum": list(profiles.keys())},
                    ],
                },
            },
            "required": ["profile"],
        },
    }
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                choose_user_profile_prompt + "\nUser profiles: {profiles}\n"
            ),
            MessagesPlaceholder(variable_name="messages")
        ]
    ).partial(profiles=str(profiles))
    res = (
            prompt
            | llm.bind_functions(functions=[function_def], function_call="routeProfile")
            | JsonOutputFunctionsParser()
    )
    return res

def choose_prompt_for_conversation():
    """Decides how to proceed with the conversation. Chooses the best approach based on the dialog and user profile."""
    function_def = {
        "name": "chooseConversationDirection",
        "description": "Select where to steer the conversation based on the dialogue.",
        "parameters": {
            "type": "object",
            "properties": {
                "conversation_direction": {
                    "title": "ConversationDirection",
                    "anyOf": [
                        {"enum": list(conversation_prompts.keys())},
                    ],
                },
            },
            "required": ["conversation_direction"],
        },
    }
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                choose_prompt_prompt + "\nPossible directions: {conversation_prompts_description}"
            ),
            MessagesPlaceholder(variable_name="messages")
        ]
    ).partial(conversation_prompts_description=str(conversation_prompts))

    llm_with_function = (prompt | llm.bind_functions(functions=[function_def],
                                                     function_call="chooseConversationDirection") | JsonOutputFunctionsParser())
    return llm_with_function


def continue_conversation():
    prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="conversation_direction"),
            ("assistant", "Profile information: {profile_info}")
        ]
    ).partial(profile_info=str(current_profile))

    res = (prompt | llm)
    return res

def create_privacy_agent():
    prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="messages"),
            ("system", privacy_manager_prompt)
        ]
    ).partial(profile_info=str(current_profile))

    res = (prompt | llm)
    return res


def agent_node(state, agent, name):
    state_key = state['profile'] if state['profile']  else list(profiles.keys())[random.randint(0, len(profiles)-1)]
    current_profile = profiles[state_key]
    result = agent.invoke({'input': state} if name == 'retrieve' else state)
    if name == 'retrieve':
        # if len(state['messages']) > 0 and isinstance(state['messages'][-1], AIMessage):
        #     state['messages'] = state['messages'][:-1]
        return {"messages": [AIMessage(content=result["output"], name=name)]}
    elif name == 'conversation':
        # if len(state['messages']) > 0 and isinstance(state['messages'][-1], AIMessage):
        #     state['messages'] = state['messages'][:-1]
        return {"messages" : [AIMessage(content=result.content, name=name)]}
    elif name == 'conversation_prompt':
        # if len(state['messages']) > 0 and isinstance(state['messages'][-1], AIMessage):
        #     state['messages'] = state['messages'][:-1]
        return {"conversation_direction" : [AIMessage(content=conversation_prompts[result['conversation_direction']], name=name)]}
    elif name == "privacy_manager" :
        # if len(state['messages']) > 0 and isinstance(state['messages'][-1], AIMessage):
        #     state['messages'] = state['messages'][:-1]
        return {"messages": [AIMessage(content=result.content, name=name)]}

# Research team graph state
class RetrieveTeamState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    members: List[str]
    next: str
    profile: str
    conversation_direction: str


### Set up agents
supervisor_agent = create_team_supervisor(
    llm,
    system_prompt, ## todo: check prompt. i think this is correct?
    ["retrieve_data_from_database", 'initiate_a_conversation_with_a_client'],
)
user_profile_agent = create_user_profile_agent(
    llm,
    profiles
)

conversation_prompt_agent = choose_prompt_for_conversation()
conversation_agent = continue_conversation()
privacy_manager_agent = create_privacy_agent()

retrieve_agent =  create_pandas_dataframe_agent(
    llm,
    df,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
)
retrieve_node = functools.partial(agent_node, agent=retrieve_agent, name="retrieve")
conversation_prompt_node = functools.partial(agent_node, agent=conversation_prompt_agent, name="conversation_prompt")
conversation_node = functools.partial(agent_node, agent=conversation_agent, name="conversation")
privacy_manager_node = functools.partial(agent_node, agent=privacy_manager_agent, name="privacy_manager")


research_graph = StateGraph(RetrieveTeamState)
research_graph.add_node("user_profile", user_profile_agent)
research_graph.add_node("supervisor", supervisor_agent)
research_graph.add_node("retrieve", retrieve_node)
research_graph.add_node("conversation_prompt", conversation_prompt_node)
research_graph.add_node("conversation", conversation_node)
research_graph.add_node("privacy_manager", privacy_manager_node)


# Define the control flow
research_graph.add_edge("user_profile", "supervisor")
research_graph.add_edge("retrieve", "supervisor")
research_graph.add_edge("conversation", "supervisor")
research_graph.add_edge("conversation_prompt", "conversation")
research_graph.add_edge("privacy_manager", END)
research_graph.add_conditional_edges(
    "supervisor",
    lambda x: x["next"],
    {"retrieve_data_from_database": "retrieve", "initiate_a_conversation_with_a_client": "conversation_prompt", "FINISH": "privacy_manager"},
)


research_graph.set_entry_point("user_profile")
chain = research_graph.compile()


# The following functions interoperate between the top level graph state
# and the state of the research sub-graph
# this makes it so that the states of each graph don't get intermixed
# The following functions interoperate between the top level graph state
# and the state of the research sub-graph
# this makes it so that the states of each graph don't get intermixed
def enter_chain(message: str):
    results = {
        "messages": [HumanMessage(content=message)],
    }
    return results


research_chain = enter_chain | chain


def get_answer(user_input):
    return research_chain.stream(
        user_input, {"recursion_limit": 5}
    )
