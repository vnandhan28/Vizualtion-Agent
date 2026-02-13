from typing import Dict, Any, Optional, Tuple
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.tools import StructuredTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
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
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
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
                name="generate_visualization",
                description="Use this tool to answer questions about data by generating visualizations, charts, or performing data analysis. Input should be the user's natural language question."
            )
        ]

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an AI data visualization assistant. When a user asks a question about their data, use the 'generate_visualization' tool to create a chart and get an explanation. If you can answer without the tool, do so directly."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = create_openai_tools_agent(self.llm, tools, prompt)
        executor = AgentExecutor(
            agent=agent, 
            tools=tools, 
            memory=self.memory,
            verbose=False,
            handle_parsing_errors=True
        )
        
        response = executor.invoke({"input": question})
        return self.latest_result, response["output"]
