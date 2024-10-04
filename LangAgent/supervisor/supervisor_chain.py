# LangAgent/nodes/supervisor_chain.py

# * LIBRARIES

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
import os


def supervisor_chain(subagent_names: list, llm, 
                     subagent_descriptions: dict = None, 
                     completion_criteria: str = "FINISH", 
                     history_depth: int = None):
    
    # If agent descriptions are not provided, initialize them with a default message
    if subagent_descriptions is None:
        subagent_descriptions = {agent: "No description available" for agent in subagent_names}
    
    # Build the default system prompt with agent descriptions
    system_prompt = (
        f"""
        You are a supervisor tasked with managing a conversation between the following workers: {subagent_names}. 
        Each worker has a specific role: 
        """
        + "\n".join([f"- {agent}: {desc}" for agent, desc in subagent_descriptions.items()])
        + f"""
        Given the following user request, respond with the worker to act next.
        
        Each worker will perform a task and respond with their results and status. 
        When all tasks are complete, respond with {completion_criteria}.
        """
    )

    # Define the routing options (FINISH plus the names of the subagents)
    route_options = [completion_criteria] + subagent_names 

    # Define the function used for routing decisions
    function_def = {
        "name": "route",
        "description": "Select the next role.",
        "parameters": {
            "title": "route_schema",
            "type": "object",
            "properties": {
                "next": {
                    "title": "Next",
                    "anyOf": [
                        {"enum": route_options},
                    ],
                }
            },
            "required": ["next"],
        },
    }

    # Create the conversation prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            # Optionally limit the history depth
            MessagesPlaceholder(variable_name="messages", limit=history_depth),
            (
                "system",
                f"Given the conversation above, who should act next? "
                f"Or should we {completion_criteria}? Select one of: {route_options}",
            ),
        ]
    ).partial(route_options=str(route_options), subagent_names=", ".join(subagent_names))

    # Build the supervisor chain
    supervisor_chain = (
        prompt
        | llm.bind_functions(functions=[function_def], function_call="route")
        | JsonOutputFunctionsParser()
    )

    return supervisor_chain

    
