from langchain.tools import BaseTool

class CalculatorTool(BaseTool):
    name = "Calculator"
    description = "Effectue des calculs mathématiques simples. Utilisez ce tool pour toute opération mathématique."

    def _run(self, query: str):
        try:
            result = eval(query, {"__builtins__": {}})
            return str(result)
        except Exception as e:
            return f"Erreur de calcul: {e}"

    async def _arun(self, query: str):
        return self._run(query)
