# LangAgent/logic_team/agents/reasoner.py


# * LIBRARIES

from langchain.agents import AgentExecutor, create_openai_tools_agent,load_tools
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool, Tool
from langchain.chains.llm import LLMChain
import os


def create_agent_with_tools(llm, tools, system_prompt):
    """
    Create an agent with a specific language model and tools.

    Parameters:
    llm (ChatOpenAI): Language model for the agent.
    tools (list): A list of tools the agent can use.
    system_prompt (str): The system instructions that guide the agent's behavior.

    Returns:
    AgentExecutor: An executor that manages the agent's responses and tool usage.
    """

    # Define the prompt template for system behavior
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    # Create the agent and link it with the defined tools and prompt
    agent = create_openai_tools_agent(llm, tools, prompt)
    
    # Use AgentExecutor to manage agent-tool interactions
    executor = AgentExecutor(agent=agent, tools=tools)
    
    return executor

def reasonser(llm, system_prompt = None):
    """
    Creates a reasoner agent using the provided LLM and search tool.
    
    Args:
    llm: The language model (e.g., ChatOpenAI).
    system_prompt: (Optional) A custom system prompt for the agent. If None, a default prompt is used.
    
    Returns:
    AgentExecutor: The reasoner agent ready to answer logic and reasoning questions.
    """

    # Default system prompt if none is provided
    if system_prompt is None:
        system_prompt = """
        You are a reasoning agent tasked with solving the user's logic-based questions and breaking down complex problems. 
        Logically arrive at the solution, and be factual. In your answers, clearly detail the steps involved and give the final answer. 
        Provide the response in bullet points.
        Example: "I have a 7 in the tens place. I have an even number in the ones place. I am lower than 74. What number am I?", 
        Example: A train leaves the station at 3:00 PM and travels at a constant speed of 60 miles per hour. 
        Another train leaves the same station at 4:00 PM and travels at a constant speed of 80 miles per hour on the same track.
        At what time will the second train catch up to the first train?
        """
    
    math_assistant_prompt = PromptTemplate(
    input_variables=[],  
    template=system_prompt
    )

    word_problem_chain = LLMChain(llm=llm, prompt=math_assistant_prompt)

    word_problem_tool = Tool.from_function(
        name="Reasoning_Tool",
        func=word_problem_chain.run,
        description="Useful for when you need to answer logic-based/reasoning questions."
    )

    reasoner_agent_executor = create_agent_with_tools(llm, [word_problem_tool], system_prompt)
    return reasoner_agent_executor