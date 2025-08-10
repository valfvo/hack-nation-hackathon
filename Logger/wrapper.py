class LoggingAgentWrapper:
    def __init__(self, agent):
        self.agent = agent

    def invoke(self, args, config):
        result = self.agent.invoke(args, config)
        states = list(self.agent.get_state_history(config))
        for state in states:
            checkpoint_id = state.config.get("configurable", {}).get("checkpoint_id", "N/A")
            
            print(f"[LOG] Checkpoint ID: {state.config['configurable']['checkpoint_id']}")
            print(f"[LOG] Next state: {state.next}")

        return result

    def stream(self, args, config):
        for output in self.agent.stream(args, config):
            print(f"[LOG] Streaming output: {output}")
        

    def replay(self, *args, checkpoint_id=None, thread_id=None, **kwargs):
        print(f"[LOG] Replay input: {args if args else kwargs}")
        result = self.agent.replay(*args, **kwargs)
        print(f"[LOG] Replay output: {result}")
        return result