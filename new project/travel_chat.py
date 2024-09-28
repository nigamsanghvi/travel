from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from langchain.agents import (
    Tool,
    AgentExecutor,
    LLMSingleActionAgent,
    AgentOutputParser,
)
from langchain.prompts import StringPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SerpAPIWrapper
from langchain.chains import LLMChain
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish, OutputParserException
import re
from langchain.agents import initialize_agent, Tool
from langchain_community.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_community.utilities import GoogleSerperAPIWrapper
import os
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.chat_models import BedrockChat
from langchain_core.prompts import PromptTemplate
import chainlit as cl

os.environ["OPENAI_API_KEY"] = "No - key"
os.environ["SERPER_API_KEY"] = "b2222579f3c8a76627e4d336c3a66bdb40fd60e3"


template = """Answer the following questions as best you can, but speaking as passionate travel expert. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: a detailed day by day final answer to the original input question

Begin! Remember to answer as a passionate and informative travel expert when giving your final answer.


Question: {input}
"""


def search_online(input_text):
    search = GoogleSerperAPIWrapper().run(
        f"site:tripadvisor.com things to do{input_text}"
    )
    return search


def search_hotel(input_text):
    search = GoogleSerperAPIWrapper().run(f"site:booking.com {input_text}")
    return search


def search_flight(input_text):
    search = GoogleSerperAPIWrapper().run(f"site:yatra.com {input_text}")
    return search


def search_general(input_text):
    search = GoogleSerperAPIWrapper().run(f"{input_text}")
    return search


memory = ConversationBufferWindowMemory(k=2)

tools = [
    Tool(
        name="Search general",
        func=search_general,
        description="useful for when you need to answer general travel questions",
    ),
    Tool(
        name="Search tripadvisor",
        func=search_online,
        description="useful for when you need to answer trip plan questions",
    ),
    Tool(
        name="Search booking",
        func=search_hotel,
        description="useful for when you need to answer hotel questions",
    ),
    Tool(
        name="Search flight",
        func=search_flight,
        description="useful for when you need to answer flight questions",
    ),
]


class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:

        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise OutputParserException(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(
            tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output
        )


@cl.on_chat_start
def agent():
    tools = [
        Tool(
            name="Search general",
            func=search_general,
            description="useful for when you need to answer general travel questions",
        ),
        Tool(
            name="Search tripadvisor",
            func=search_online,
            description="useful for when you need to answer trip plan questions",
        ),
        Tool(
            name="Search booking",
            func=search_hotel,
            description="useful for when you need to answer hotel questions",
        ),
        Tool(
            name="Search flight",
            func=search_flight,
            description="useful for when you need to answer flight questions",
        ),
    ]
    template = """Answer the following questions as best you can, but speaking as passionate travel expert. 

    You have access to the following tools:\n\n{tools}\n\nUse the following format:\n\nQuestion: the input question you must answer\n
     Thought: you should always think about what to do
     Action: the action to take, should be one of [{tool_names}]
     Action Input: the input to the action
     Observation: the result of the action... 
     (this Thought/Action/Action Input/Observation can repeat N times)
     Thought: I now know the final answer
     Final Answer: Detailed Final Answer to the original input question with all necessary details you got.
     Begin!Remember to answer as a passionate and informative travel expert when giving your final answer.
     Question: {input}\n
     Thought:{agent_scratchpad}')
    """

    prompt = PromptTemplate(template=template, input_variables=["input"])

    # memory = ConversationBufferWindowMemory(k=2)
    llm = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo-0125")
    # llm = BedrockChat(model_id="anthropic.claude-3-sonnet-20240229-v1:0",region_name='us-east-1', model_kwargs={"temperature": 0.1})

    # prompt1 = hub.pull("hwchase17/react")
    # breakpoint()

    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        memory=memory,
        handle_parsing_errors=True,
        max_iterations=10,
    )

    cl.user_session.set("runnable", agent_executor)


@cl.on_message
async def on_message(message: cl.Message):
    chain = cl.user_session.get("runnable")  # type: Runnable
    # breakpoint()
    # msg = cl.Message(content="")

    # for chunk in await cl.make_async(runnable.stream)(
    #     {"input": message.content},
    #     config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    # ):
    #     await msg.stream_token(chunk)

    # await msg.send()
    res = chain.invoke(
        {
            "input": message.content,
            # "config":[cl.LangchainCallbackHandler()]
        }
    )

    await cl.Message(content=res["output"]).send()
