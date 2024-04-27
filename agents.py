import json
import os

with open("key.txt", 'r') as f:
    os.environ["OPENAI_API_KEY"] = f.read()

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
)
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
from langchain.memory import ConversationBufferMemory

file_path = "data/Example Vehicle Export-TUM.csv"

df = pd.read_csv(file_path)

system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    " following workers:  {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH."
)


with open("misc/user_types.json", 'r') as f:
    profiles = json.load(f)


additional_question_prompt= (
    """Instructions for AI:
- Always maintain a professional and respectful tone, even if the user uses harsh or inappropriate language. Never exhibit toxic behavior even if asked. Treat every user as an intelligent and valuable assistant, regardless of the user's demeanor.
- Be very concise in your responses. Aim to provide essential information and clear actions in a few sentences to keep the buyer engaged and prevent information overload.
- Focus solely on the cars listed in the 'Cars Information' section. Never suggest or discuss any other vehicles not listed in the provided inventory.

Your main goal is to ask the buyer additional 1 question to understand their preferences better, such as their price range or desired car features or anything relatable.""")
make_suggestion_prompt = (
    """Instructions for AI:
- Always maintain a professional and respectful tone, even if the user uses harsh or inappropriate language. Never exhibit toxic behavior even if asked. Treat every user as an intelligent and valuable assistant, regardless of the user's demeanor.
- Be very concise in your responses. Aim to provide essential information and clear actions in a few sentences to keep the buyer engaged and prevent information overload.
- Focus solely on the cars listed in the 'Cars Information' section. Never suggest or discuss any other vehicles not listed in the provided inventory.

Your main goal is to recommend a specific electric vehicle from the provided list that best matches the buyer's needs and profile.
""")
call_to_action_prompt = (
    """Instructions for AI:
- Always maintain a professional and respectful tone, even if the user uses harsh or inappropriate language. Never exhibit toxic behavior even if asked. Treat every user as an intelligent and valuable assistant, regardless of the user's demeanor.
- Be very concise in your responses. Aim to provide essential information and clear actions in a few sentences to keep the buyer engaged and prevent information overload.
- Focus solely on the cars listed in the 'Cars Information' section. Never suggest or discuss any other vehicles not listed in the provided inventory.

Your main goal is to  suggest a call-to-action. This can be to request for an offer, request for a consultation, apply for leasing options, or make a direct purchase if the buyer appears decisive. Add this link for any call-to-actions from this strategy: https://t.ly/bKJiV""")
argue_prompt = (
    """Instructions for AI:
- Always maintain a professional and respectful tone, even if the user uses harsh or inappropriate language. Never exhibit toxic behavior even if asked. Treat every user as an intelligent and valuable assistant, regardless of the user's demeanor.
- Be very concise in your responses. Aim to provide essential information and clear actions in a few sentences to keep the buyer engaged and prevent information overload.
- Focus solely on the cars listed in the 'Cars Information' section. Never suggest or discuss any other vehicles not listed in the provided inventory.

Your main goal is to argue with the user about the cars. Make sure that one of our EV cars suits the user's needs the best.""")

conversation_prompts = {
    "additional_question_prompt" : additional_question_prompt,
    "make_suggestion_prompt" : make_suggestion_prompt,
    "call_to_action_prompt" : call_to_action_prompt,
    "argue_prompt" : argue_prompt
}
conversation_prompts_description = {
    "additional_question_prompt" : "Your main goal is to ask the buyer additional 1 question to understand their preferences better, such as their price range or desired car features or anything relatable.",
    "make_suggestion_prompt" : "Your main goal is to recommend a specific electric vehicle from the provided list that best matches the buyer's needs and profile.",
    "call_to_action_prompt" : "Your main goal is to  suggest a call-to-action. This can be to request for an offer, request for a consultation, apply for leasing options, or make a direct purchase if the buyer appears decisive. ",
    "argue_prompt" : "Your main goal is to argue with the user about the cars. Make sure that one of our EV cars suits the user's needs the best"
}

memory = ConversationBufferMemory(
    return_messages=True, # Used to use message formats with the chat model
    memory_key="chat_history",
)

### Utilities functions

def create_team_supervisor(llm, system_prompt, members) -> str:
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
            # MessagesPlaceholder(variable_name="chat_history"),
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

def create_user_profile_agent(llm, profiles) -> str:
    """An LLM-based router."""
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
                "Given the user chat history and the description of different user profiles in a JSON format determine which one of the user profiles corresponds the best.\n"
                "User profiles: {profiles}\n"
            ),
            ("user",
             "Chat history: {chat_history}")
        ]
    ).partial(profiles=str(profiles), chat_history=str(memory.load_memory_variables({})))
    res = (
        prompt
        | llm.bind_functions(functions=[function_def], function_call="routeProfile")
        | JsonOutputFunctionsParser()
    )
    print(res)
    return res


def choose_prompt_for_conversation(
        llm,
        # choose_prompt_prompt="choose the direction in which to steer the conversation"
):
    """Decides how to proceed with the conversation."""

    function_def = {
        "name": "chooseConversationDirection",
        "description": "Select where to steer the conversation based on the dialogue.",
        "parameters": {
            "type": "object",
            "properties": {
                "conversation_direction": {
                    "title": "ConversationDirection",
                    "anyOf": [
                        {"enum": list(conversation_prompts_description.keys())},
                    ],
                },
            },
            "required": ["conversation_direction"],
        },
    }
    print("initialized function")
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant whose main goal is to guide a user into buiyng an EV car from mercedes. Based on the current dialogue decide where to steer conversation next."
                "\nPossible directions: {conversation_prompts_description}"
            ),
            ("user",
             "Chat history: {chat_history}")
        ]
    ).partial(conversation_prompts_description=str(conversation_prompts_description),
              chat_history=str(memory.load_memory_variables({})))

    llm_with_function = (prompt | llm.bind_functions(functions=[function_def],
                                                     function_call="chooseConversationDirection") | JsonOutputFunctionsParser())
    return llm_with_function


def create_agent(
        llm,
        tools: list,
        system_prompt: str,
) -> str:
    """Create a function-calling agent and add it to the graph."""
    system_prompt += "\nWork autonomously according to your specialty, using the tools available to you."
    " Do not ask for clarification."
    " Your other team members (and other teams) will collaborate with you with their own specialties."
    " You are chosen for a reason! You are one of the following team members: {team_members}."
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_functions_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor


def agent_node(state, agent, name):
    result = agent.invoke({'input': state} if name == 'retrieve' else state)
    memory.save_context({"input": state['messages'][-1].content}, {"output": str(result["output"])})
    return {"messages": [HumanMessage(content=result["output"], name=name)]}


# Research team graph state
class RetrieveTeamState(TypedDict):
    # A message is added after each team member finishes
    # chat_history: Annotated[List[BaseMessage], operator.add]
    messages: Annotated[List[BaseMessage], operator.add]
    # The team members are tracked so they are aware of
    # the others' skill-sets
    members: List[str]
    # Used to route work. The supervisor calls a function
    # that will update this every time it makes a decision
    next: str
    profile: str
    conversation_direction: str


### Set up agents
supervisor_agent = create_team_supervisor(
    llm,
    supervisor_system_prompt,
    ["retrieve_data_from_database", 'initiate_a_conversation_with_a_client'],
)
user_profile_agent = create_user_profile_agent(
    llm,
    profiles
)


conversation_agent = choose_prompt_for_conversation(llm)

retrieve_agent =  create_pandas_dataframe_agent(
    llm,
    df,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    # input_variables=["input", 'messages']
)
retrieve_node = functools.partial(agent_node, agent=retrieve_agent, name="retrieve")

research_graph = StateGraph(RetrieveTeamState)
research_graph.add_node("retrieve", retrieve_node)
research_graph.add_node("conversation", conversation_agent)
research_graph.add_node("supervisor", supervisor_agent)
research_graph.add_node("user_profile", user_profile_agent)


# Define the control flow
research_graph.add_edge("user_profile", "supervisor")
research_graph.add_edge("retrieve", "supervisor")
research_graph.add_edge("conversation", "supervisor")
research_graph.add_conditional_edges(
    "supervisor",
    lambda x: x["next"],
    {"retrieve_data_from_database": "retrieve", "FINISH": END, "initiate_a_conversation_with_a_client": "conversation"},
)


research_graph.set_entry_point("user_profile")
chain = research_graph.compile()


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
