import json
import os

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

file_path = "api_calls/mercedes_ev_llm.csv"

df = pd.read_csv(file_path)


with open("key.txt", 'r') as f:
    os.environ["OPENAI_API_KEY"] = f.read()

system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    " following workers:  {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH."
)


with open("misc/user_types.json", 'r') as f:
    profiles = json.load(f)

with open("misc/conversation_prompts.json", 'r') as f:
    conversation_prompts = json.load(f)


current_profile = list(profiles.values())[0]

memory = ConversationBufferMemory(
    return_messages=True, # Used to use message formats with the chat model
    memory_key="chat_history",
)

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

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
            # ("assistant", f"Retrieval result: {mercedes_data}")
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
            MessagesPlaceholder(variable_name="messages")

        ]
    ).partial(profiles=str(profiles))
    # llm_with_function = llm.bind_functions(functions=[function_def], function_call="routeProfile")
    # profile = eval(llm_with_function.invoke(prompt).additional_kwargs['function_call']['arguments'])['profile']
    res = (
            prompt
            | llm.bind_functions(functions=[function_def], function_call="routeProfile")
            | JsonOutputFunctionsParser()
    )
    # res = {key.replace(' ', '_') : profiles[profile][key] for key in list(profiles[profile].keys())}
    # print(res)
    return res


def choose_prompt_for_conversation(
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
                "You are a helpful assistant whose main goal is to guide a user into buiyng an EV car from mercedes. Based on the current dialogue decide where to steer conversation next."
                "\nPossible directions: {conversation_prompts_description}"
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
            MessagesPlaceholder(variable_name="conversation_direction"),
            MessagesPlaceholder(variable_name="messages"),
            ("assistant", "Profile information: {profile_info}")
        ]
    ).partial(profile_info=str(current_profile))

    res = (prompt | llm)
    return res


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
    global current_profile, mercedes_data
    current_profile = profiles[state['profile']]
    result = agent.invoke({'input': state} if name == 'retrieve' else state)
    if name == 'retrieve':
      # memory.save_context({"input": state['messages'][-1].content}, {"output": str(result["output"])})
      mercedes_data = result["output"]
      if len(state['messages']) > 0 and isinstance(state['messages'][-1], AIMessage):
        state['messages'] = state['messages'][:-1]
      return {"messages": state['messages'] + [AIMessage(content=result["output"], name=name)]}
    elif name == 'conversation':
      # memory.save_context({"input": state['messages'][-1].content}, {"output": str(result.content)})
      if len(state['messages']) > 0 and isinstance(state['messages'][-1], AIMessage):
        state['messages'] = state['messages'][:-1]
      return {"messages" : [AIMessage(content=result.content, name=name)]}
    elif name == 'conversation_prompt':
      if len(state['messages']) > 0 and isinstance(state['messages'][-1], AIMessage):
        state['messages'] = state['messages'][:-1]
      return {"conversation_direction" : [AIMessage(content=conversation_prompts[result['conversation_direction']], name=name)]}

# Research team graph state
class RetrieveTeamState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    members: List[str]
    next: str
    profile: str
    conversation_direction: str
    mercedes_data : Annotated[List[BaseMessage], operator.add]


### Set up agents
supervisor_agent = create_team_supervisor(
    llm,
    system_prompt, ##todo: change prompt
    ["retrieve_data_from_database", 'initiate_a_conversation_with_a_client'],
)
user_profile_agent = create_user_profile_agent(
    llm,
    profiles
)

conversation_prompt_agent = choose_prompt_for_conversation()
conversation_agent = continue_conversation()

retrieve_agent =  create_pandas_dataframe_agent(
    llm,
    df,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
)
retrieve_node = functools.partial(agent_node, agent=retrieve_agent, name="retrieve")
conversation_node = functools.partial(agent_node, agent=conversation_agent, name="conversation")
conversation_prompt_node = functools.partial(agent_node, agent=conversation_prompt_agent, name="conversation_prompt")


research_graph = StateGraph(RetrieveTeamState)
research_graph.add_node("retrieve", retrieve_node)
research_graph.add_node("conversation_prompt", conversation_prompt_node)
research_graph.add_node("conversation", conversation_node)
research_graph.add_node("supervisor", supervisor_agent)
research_graph.add_node("user_profile", user_profile_agent)


# Define the control flow
research_graph.add_edge("user_profile", "supervisor")
research_graph.add_edge("retrieve", "supervisor")
research_graph.add_edge("conversation", "supervisor")
research_graph.add_edge("conversation_prompt", "conversation")
research_graph.add_conditional_edges(
    "supervisor",
    lambda x: x["next"],
    {"retrieve_data_from_database": "retrieve", "FINISH": END, "initiate_a_conversation_with_a_client": "conversation_prompt"},
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
