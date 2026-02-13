import langchain.agents
print("All in langchain.agents:")
print(dir(langchain.agents))
try:
    from langchain.agents import create_openai_tools_agent
    print("Successfully imported create_openai_tools_agent from langchain.agents")
except ImportError as e:
    print(f"Failed to import create_openai_tools_agent from langchain.agents: {e}")

try:
    from langchain.agents import create_tool_calling_agent
    print("Successfully imported create_tool_calling_agent from langchain.agents")
except ImportError as e:
    print(f"Failed to import create_tool_calling_agent from langchain.agents: {e}")
