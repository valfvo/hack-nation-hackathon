import requests

class LoggingAgentWrapper:
    def __init__(self, agent):
        self.agent = agent
        self.logging_service_url = "http://logging-service/log"
        self.states = {}

    def invoke(self, args, config):
        result = self.agent.invoke(args, config)
        states = list(self.agent.get_state_history(config))
    
        for state in states:
            checkpoint_id = state.config.get("configurable", {}).get("checkpoint_id", "N/A")
            thread_id = state.config.get("configurable", {}).get("thread_id", "N/A")
            values = state.values if state.values else {}
            created_at = state.created_at if hasattr(state, 'created_at') else "N/A"
            curr_messages = values["messages"]
            content = ""
            self.states[checkpoint_id] = state
            
            if len(curr_messages) > 0:
                content = curr_messages[-1].content
            print(f"[LOG] Checkpoint ID: {checkpoint_id}, Created At: {created_at}, Content: {content}, Thread ID: {thread_id}")
            # post to a logging service
            # requests.post(self.logging_service_url, json={
            #     "thread_id": thread_id,
            #     "checkpoint_id": checkpoint_id,
            #     "created_at": created_at,
            #     "content": content
            # })

        return result

    def stream(self, args, config):
        for output in self.agent.stream(args, config):
            print(f"[LOG] Streaming output: {output}")
        

    def replay(self, args, checkpoint_id=None, thread_id=None, patch=None):
        config = {"configurable": {"checkpoint_id": checkpoint_id, "thread_id": thread_id}}
        selected_state = self.states.get(checkpoint_id)
        values["messages"[-1].content = path 
        result = self.invoke(args, config)
        print(f"[LOG] Replay output: {result}")
        return result