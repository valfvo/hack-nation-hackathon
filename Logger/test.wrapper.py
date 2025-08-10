from langgraph.prebuilt import create_react_agent
from wrapper import LoggingAgentWrapper
from langgraph.checkpoint.memory import InMemorySaver

def get_weather(city: str) -> str:  
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


checkpointer = InMemorySaver()

agent = create_react_agent(
    model="openai:gpt-4.1",  
    tools=[get_weather],  
    prompt="You are a helpful assistant"  ,
    checkpointer=checkpointer,
)

agent = LoggingAgentWrapper(agent)


config = {"configurable": {"thread_id": "1"}}

# Run the agent
res = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]},
    config)



