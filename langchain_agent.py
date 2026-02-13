from typing import Dict, Any, Optional, Tuple
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.tools import StructuredTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from AgentClass import HFRouterCodeGenerator, run_python_chart, GenRequest, create_client, ChartResult

class LangChainVizAgent:
    def __init__(self, api_key: str, model: str = "moonshotai/Kimi-K2-Instruct-0905"):
        self.llm = ChatOpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=api_key,
            model=model,
            temperature=0.2
        )
        client = create_client(api_key)
        self.codegen = HFRouterCodeGenerator(client=client, model=model)
        self.memory = ConversationBufferMemory(memory_key="messages", return_messages=True)
        self.latest_result: Optional[ChartResult] = None
        self.datasets: Dict[str, Any] = {}

    def _generate_visualization_tool(self, question: str) -> str:
        req = GenRequest(question=question, datasets=self.datasets)
        gen = self.codegen.generate(req)
        self.latest_result = run_python_chart(gen.code, self.datasets)
        return self.latest_result.explanation

    def ask(self, question: str, datasets: Dict[str, Any]) -> Tuple[Optional[ChartResult], str]:
        self.datasets = datasets
        self.latest_result = None
        
        tools = [
            StructuredTool.from_function(
                func=self._generate_visualization_tool,
                name="data_assistant",
                description="Use this tool to answer questions about data by generating visualizations, charts, or performing data preparation/cleaning (renaming columns, removing nulls, filtering, etc.). Input should be the user's natural language question."
            )
        ]

        system_prompt = "You are an AI data assistant. When a user asks a question about their data or requests data preparation (like cleaning, renaming, or filtering), use the 'data_assistant' tool to perform the task and get an explanation. If you can answer without the tool, do so directly."

        # The new create_agent returns a compiled graph
        agent = create_agent(
            model=self.llm,
            tools=tools,
            system_prompt=system_prompt
        )
        
        # Get history from memory
        history = self.memory.load_memory_variables({})["messages"]
        
        # Add current user message
        messages = history + [("user", question)]
        
        # Invoke agent
        response = agent.invoke({"messages": messages})
        
        # Last message in the response is the assistant's output
        output = response["messages"][-1].content
        
        # Save to memory
        self.memory.save_context({"input": question}, {"output": output})
        
        return self.latest_result, output
