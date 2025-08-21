try:
    from crewai import Task
except Exception:
    # Minimal placeholder Task for tests
    class Task:
        def __init__(self, description=None, expected_output=None, agent=None):
            self.description = description
            self.expected_output = expected_output
            self.agent = agent

from agents import vision_agent, general_agent

class AgentTasks:
    def image_analysis_task(self, prompt):
        return Task(
            description=f"Analyze the current screen capture to help with the user's request: '{prompt}'. Use the Screen Capture Tool to see the screen, then describe what you see and how it relates to the user's query.",
            expected_output="A detailed description of the screen content relevant to the user's prompt, and a helpful answer to their question.",
            agent=vision_agent,
        )

    def general_research_task(self, prompt):
        return Task(
            description=f"Address the user's request: '{prompt}'. Provide a comprehensive answer, code snippet, or explanation as required.",
            expected_output="A helpful and accurate response to the user's query.",
            agent=general_agent,
        )