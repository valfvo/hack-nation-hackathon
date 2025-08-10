import requests
from datetime import datetime

API_URL = "https://api.example.com"  # Replace with your actual API URL


class LoggingAgentWrapper:
    def __init__(self, agent):
        self.agent = agent
        self.states = {}
    
    def process_created_at(self, created_at_str, last_timestamp_str):

        # Parse ISO 8601 with timezone
        current_ts = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
        last_ts = datetime.fromisoformat(last_timestamp_str.replace("Z", "+00:00")) if last_timestamp_str else None

        # Compute difference in seconds
        diff = (current_ts - last_ts).total_seconds()
        return diff

    def create_task(self, args):
        # Call the API task endpoint with args as part of the request
        return "id"
        response = requests.post(f"{API_URL}/task", json=args)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to create task: {response.text}")

    def upload_step(self, dict_step):
        # Call the API step endpoint with dict_step as part of the request
        return "ok"
        response = requests.post(f"{API_URL}/step", json=dict_step)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to upload step: {response.text}")

    def invoke(self, args, config=None):
        if config is None:
            config = {"configurable": {"thread_id": "1"}}
        # Call the API task endpoint with config as part of the request
        run_id = config["configurable"]["thread_id"]
        task_id = self.create_task({ "run_id": run_id })
        #response = requests.post(f"{API_URL}/task", json=args)

        result = self.agent.invoke(args, config)
        states = list(self.agent.get_state_history(config))

        last_tool_call = []
        last_timestamp = None
        for state in reversed(states[:-1]):
        
            checkpoint_id = state.config.get("configurable", {}).get("checkpoint_id", "N/A")
            thread_id = state.config.get("configurable", {}).get("thread_id", "N/A")
            values = state.values if state.values else {}
            created_at = state.created_at if hasattr(state, 'created_at') else "N/A"
            curr_messages = values.get("messages", [])
            content = ""
            if last_timestamp is None:
                last_timestamp = created_at
                duration = 0
            else: 
                duration = self.process_created_at(created_at, last_timestamp)
                last_timestamp = created_at
            

            self.states[checkpoint_id] = state
            
            message_type = "unknown"
            metadata = {}
            tool_calls = []
            if len(curr_messages) > 0:
                content = curr_messages[-1].content
                message_type = curr_messages[-1].type if hasattr(curr_messages[-1], 'type') else "unknown"
                metadata = curr_messages[-1].response_metadata
                tool_calls = curr_messages[-1].tool_calls if hasattr(curr_messages[-1], 'tool_calls') else []
            # print(f"[LOG] Checkpoint ID: {checkpoint_id}, Created At: {created_at}, Content: {content}, Thread ID: {thread_id}")
            
            step_dict = {
                "content": content,
                "tool_calls": last_tool_call,
                "message_type": message_type,
            }

            metrics_dict = {
                "created_at": created_at,
                "duration": duration,
                "metadata": metadata,
            }
            dict_state = {
                "task_id": task_id,
                "run_id": run_id,
                "step_id": checkpoint_id,
                "step": step_dict,
                "metrics": metrics_dict,
                # "values": values
            }
            print(f"[LOG] State: {dict_state}")
            #print("f[LOG] TRUE STATE: ", state)
            self.upload_step(dict_state)
            # print(f"[LOG] State: {dict_state}")
            # if "parsed_details" in values:
            #     print(values["parsed_details"])
            last_tool_call = tool_calls
        return result

    def stream(self, args, config):
        for output in self.agent.stream(args, config):
            print(f"[LOG] Streaming output: {output}")
        

    def replay(self, args, checkpoint_id=None, thread_id=None, patch=None):
        config = {"configurable": {"checkpoint_id": checkpoint_id, "thread_id": thread_id}}
        selected_state = self.states.get(checkpoint_id)
        # values["messages"[-1].content = path 
        # result = self.invoke(args, config)
        # print(f"[LOG] Replay output: {result}")
        # return result