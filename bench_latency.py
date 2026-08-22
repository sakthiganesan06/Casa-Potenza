import time
import asyncio
import torch
import numpy as np
import config
from groq import AsyncGroq
from retrieval.embedder import get_embedder

async def main():
    print("=== 1. Groq Model Latency Benchmark ===")
    client = AsyncGroq(api_key=config.GROQ_API_KEY)
    models = ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "openai/gpt-oss-20b"]
    
    for m in models:
        t0 = time.perf_counter()
        try:
            stream = await client.chat.completions.create(
                model=m,
                messages=[{"role": "user", "content": "What is 2+2? Answer in JSON: {'result': 4}"}],
                temperature=0.1,
                max_tokens=60,
                stream=True,
                response_format={"type": "json_object"}
            )
            ttft = None
            async for chunk in stream:
                if ttft is None and chunk.choices[0].delta.content:
                    ttft = (time.perf_counter() - t0) * 1000
            total = (time.perf_counter() - t0) * 1000
            print(f"Model: {m:<25} | TTFT: {ttft:.1f}ms | Total: {total:.1f}ms")
        except Exception as e:
            print(f"Model: {m:<25} | Error: {e}")

    print("\n=== 2. Embedding Latency Benchmark ===")
    embedder = await get_embedder()
    # Test different thread counts
    for num_threads in [1, 2, 4, 8]:
        torch.set_num_threads(num_threads)
        t0 = time.perf_counter()
        for _ in range(5):
            await embedder.embed_one("What is the character of the university school?")
        avg_ms = ((time.perf_counter() - t0) / 5) * 1000
        print(f"Threads: {num_threads} | Avg Embed Time: {avg_ms:.1f}ms")

if __name__ == "__main__":
    asyncio.run(main())
