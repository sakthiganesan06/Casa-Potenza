"""
app.embedder — Default embedder module for eval loop, delegating to eval_adapter.
"""
from eval_adapter import embed, embed_one, get_model

__all__ = ["embed", "embed_one", "get_model"]
