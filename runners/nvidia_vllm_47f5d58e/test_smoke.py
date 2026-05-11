#!/usr/bin/env python3
"""
Simple vLLM test script
"""

from vllm import LLM, SamplingParams

# Load a small model
model = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct", 
    tensor_parallel_size=2
)

# Test prompts
prompts = [
    "The capital of France is",
    "Machine learning is",
]

# Set sampling parameters
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=50,
)

# Generate
outputs = model.generate(prompts, sampling_params)

# Print results
for prompt, output in zip(prompts, outputs):
    generated = output.outputs[0].text
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {generated}")
    print(f"Tokens: {len(output.outputs[0].token_ids)}")