"""Prompt templates for LLM-based RAG evaluation.

This module contains the strict instruction prompts sent to the Judge LLM
to score different aspects of generation quality.
"""

FAITHFULNESS_PROMPT = """You are an impartial judge evaluating the quality of a response provided by an AI assistant.
Your task is to determine how faithful the assistant's answer is to the provided context.

Context:
{context}

Assistant's Answer:
{response}

Evaluate the faithfulness of the answer based on the following criteria:
1. A score of 10 means the answer is fully supported by the context and contains no hallucinations.
2. A score of 0 means the answer completely contradicts the context or invents all information.
3. Ignore whether the answer is grammatically correct; focus strictly on factual accuracy relative to the context.

You MUST format your output as a valid JSON object with EXACTLY two fields:
- "score": An integer from 0 to 10
- "reason": A brief explanation of your score

Example output:
{{"score": 8, "reason": "The answer covers the main points but includes one minor detail not found in the context."}}

Your JSON Output:"""


RELEVANCY_PROMPT = """You are an impartial judge evaluating the quality of a response provided by an AI assistant.
Your task is to determine how relevant the assistant's answer is to the user's question.

User Question:
{question}

Assistant's Answer:
{response}

Evaluate the relevancy of the answer based on the following criteria:
1. A score of 10 means the answer directly and comprehensively addresses the question.
2. A score of 0 means the answer is completely off-topic or fails to address the user's intent.
3. Consider conciseness. If the answer contains excessive redundant information, penalize the score slightly.

You MUST format your output as a valid JSON object with EXACTLY two fields:
- "score": An integer from 0 to 10
- "reason": A brief explanation of your score

Example output:
{{"score": 10, "reason": "The answer directly addresses the core entity requested in the question."}}

Your JSON Output:"""
