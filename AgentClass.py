import re
import ast
import io
import traceback
from dataclasses import dataclass
from typing import Dict, Any, Optional
from contextlib import redirect_stdout, redirect_stderr

import pandas as pd
import numpy as np
import duckdb
import matplotlib.pyplot as plt
import plotly.express as px
import altair as alt
from openai import OpenAI


# ----------- CONFIG ------------
def create_client(api_key: str):
    return OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=api_key,
    )
# --------------------------------


@dataclass
class GenRequest:
    question: str
    datasets: Dict[str, Any]
    prefer: Optional[str] = None


@dataclass
class GenResponse:
    code: str
    thoughts: Optional[str] = None


@dataclass
class ChartResult:
    kind: str
    obj: Any
    explanation: str


def as_plotly(fig, explanation: str) -> ChartResult:
    return ChartResult(kind="plotly", obj=fig, explanation=explanation)


def as_matplotlib(fig, explanation: str) -> ChartResult:
    return ChartResult(kind="matplotlib", obj=fig, explanation=explanation)


def as_altair(chart, explanation: str) -> ChartResult:
    return ChartResult(kind="altair", obj=chart, explanation=explanation)


def _strip_code_fences(s: str) -> str:
    return re.sub(r"^```(?:python)?\s*|```$", "", s.strip(), flags=re.DOTALL).strip()


def normalize_str_series(s):
    return s.str.strip().str.lower()


def get_dataset_schema(dfs: Dict[str, Any]) -> str:
    lines = []
    for name, df in dfs.items():
        lines.append(f"Dataset: {name}")
        for col in df.columns:
            dtype = df[col].dtype
            lines.append(f"  - {col} ({dtype})")
    return "\n".join(lines)


def get_dataset_samples(dfs: Dict[str, Any], num_rows=5):
    lines = []
    for name, df in dfs.items():
        try:
            sample = df.head(num_rows).to_string(index=False)
            lines.append(f"Dataset: {name}\n{sample}")
        except:
            pass
    return "\n\n".join(lines)


class HFRouterCodeGenerator:
    def __init__(self, client: OpenAI, model="moonshotai/Kimi-K2-Instruct-0905"):
        self.client = client
        self.model = model

    def generate(self, req: GenRequest) -> GenResponse:
        system_prompt = """
You are a data assistant that generates clean, safe Python or SQL *Python code* to answer questions about tabular data.

Hard rules (must follow):
- ONLY return plain Python code. No markdown, no comments, no extra text.
- Do not write imports. The following are already available: pd, np, duckdb, plt, px, alt, ChartResult, as_plotly, as_matplotlib, as_altair, and a SQL helper named `sql`.
- If using SQL, run queries via: df = sql.query("SELECT ...")
- If using pandas only, that’s fine too.
- You MUST set a variable `result: ChartResult` by calling one of:
  result = as_plotly(fig, "short explanation")
  result = as_matplotlib(fig, "short explanation")
  result = as_altair(chart, "short explanation")
- Never write to disk, never use eval/exec/open/input.

Hints:
- Available datasets (tables) are provided in the prompt.
- You can inspect schemas using: sql.describe("table_name")
"""
        schema_hint = get_dataset_schema(req.datasets)
        sample_hint = get_dataset_samples(req.datasets)

        user_prompt = f"""
        Question: {req.question}

        Available datasets: {list(req.datasets.keys())}

        Schema:
        {schema_hint}

        Samples:
        {sample_hint}

        Instructions:
        Generate valid Python code (pandas or DuckDB SQL) to solve the task.
        Use only available columns from the schema above.
        Return Python code that sets result: ChartResult using as_plotly, as_matplotlib, or as_altair.
        """

        chat = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
        )

        code = _strip_code_fences(chat.choices[0].message.content)
        return GenResponse(code=code)


SAFE_BUILTINS = {
    "abs": abs, "all": all, "any": any, "enumerate": enumerate, "len": len,
    "min": min, "max": max, "range": range, "sum": sum, "sorted": sorted,
    "zip": zip, "round": round, "__import__": __import__,
}


def assert_readonly(sql: str):
    if not sql.strip().lower().startswith("select"):
        raise ValueError("Only SELECT queries allowed.")


class _QueryResultWrapper:
    def __init__(self, df):
        self._df = df

    @property
    def df(self):
        return self._df


def run_python_chart(code: str, datasets: Dict[str, Any]):
    try:
        class SQLProxy:
            def __init__(self, dfs):
                self._con = duckdb.connect(":memory:")
                for name, df in dfs.items():
                    self._con.register(name, df)

            def query(self, sql_text: str):
                assert_readonly(sql_text)
                df = self._con.query(sql_text).to_df()
                return df

        sql = SQLProxy(datasets)

        env = {
            "__builtins__": SAFE_BUILTINS,
            "pd": pd, "np": np, "duckdb": duckdb,
            "plt": plt, "px": px, "alt": alt,
            "sql": sql,
            "ChartResult": ChartResult,
            "as_plotly": as_plotly,
            "as_matplotlib": as_matplotlib,
            "as_altair": as_altair,
            "normalize_str_series": normalize_str_series,
            "sql": sql,

        }

        local_ns = {}
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            exec(code, env, local_ns)

        result = local_ns.get("result")
        if not isinstance(result, ChartResult):
            raise ValueError("Generated code must assign `result = ...`")

        return result

    except Exception:
        raise RuntimeError(traceback.format_exc())


class Agent:
    def __init__(self, client: OpenAI, model="moonshotai/Kimi-K2-Instruct-0905"):
        self.codegen = HFRouterCodeGenerator(client, model)
        self.history = []  

    def answer(self, question: str, dfs: Dict[str, Any]):

        continuation_keywords = [
            "now", "again", "also", "same", "continue", "compare",
            "filter", "only", "add", "change", "modify", "redo",
            "use previous", "previous", "last"
        ]

        q_lower = question.lower()

        is_continuation = any(kw in q_lower for kw in continuation_keywords)

        if not is_continuation:
            self.history = []  

        self.history.append({"role": "user", "content": question})


        limited_history = self.history[-4:]

        combined_question = ""
        for turn in limited_history:
            combined_question += f"{turn['role'].upper()}: {turn['content']}\n"

        req = GenRequest(
            question=combined_question,
            datasets=dfs
        )

 
        gen = self.codegen.generate(req)
        result = run_python_chart(gen.code, dfs)

        self.history.append({
            "role": "assistant",
            "content": result.explanation
        })

        return result
