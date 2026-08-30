"""
app.generator — Default generator module for eval loop, delegating to eval_adapter.
"""
from eval_adapter import generate_answer, AnswerResult

__all__ = ["generate_answer", "AnswerResult"]
