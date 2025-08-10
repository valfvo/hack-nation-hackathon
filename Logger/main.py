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


config = {"configurable": {"thread_id": "1"}}

# Run the agent
stream = agent.stream(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]},
    config)
for a in stream:
    print(a.keys())
    state = list(agent.get_state_history(config))[0]
    print(state.config["configurable"]["checkpoint_id"])


states = list(agent.get_state_history(config))
for state in states:
    print(state.next)
    print(state.config["configurable"]["checkpoint_id"])
    print(state.values)



# selected_state = states[1]
# values = selected_state.values
# values["messages"][-1].content = "It's always raining in sf"
# new_config = agent.update_state(selected_state.config, values=values)
# print("Updated state:")
# print(new_config)
# print()
# # Update the agent with the new state
# states = agent.stream(
#     {"messages": [{"role": "user", "content": "what is the weather in sf"}]},
#     new_config
# )
# print()
# for state in states:
#     print(state.keys())
#     print(state)

#     print()
# print()
# print(agent.invoke({"messages": [{"role": "user", "content": "what is the weather in sf"}] }, config ))



# # {'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f075714-b565-6ebd-8002-31bca9e04b68'}}
# # backconfig = checkpoint_list[-4]
# # messages = agent.invoke({"role": "tool", "content": "It's always raining in sf"}, backconfig.config)
# # print("ok")
# # for message in messages:
# #     print(message)  # Output: "It's always raining in sf"
# # # agent = LoggingAgentWrapper(agent)  # Wrap the agent with logging
