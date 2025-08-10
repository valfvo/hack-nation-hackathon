from langgraph.graph import Graph
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage

# Define your prompts
prompt_template = PromptTemplate(
    input_variables=["input"],
    template="You said: {input}. How can I assist you further?",
)

# Define a simple node function
def process_prompt(input):
    prompt = prompt_template.format(input=input)
    # Here you would typically call your LLM to process the prompt
    return prompt

# Create a graph
workflow = Graph()

# Add nodes to the graph
workflow.add_node("process_prompt", process_prompt)

# Define the entry point
workflow.set_entry_point("process_prompt")

# Compile the workflow
app = workflow.compile()

# Example usage
response = app.invoke(HumanMessage(content="Hello, how are you?"))
print(response)
