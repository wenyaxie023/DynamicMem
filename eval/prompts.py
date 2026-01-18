BASIC_JUDGE_PROMPT = """
You are an expert evaluator. Your task is to rate the quality of an AI-generated answer based on a standard reference.

[Question]: {query}
[Standard Reference]: {reference}
[AI Prediction]: {prediction}

Rate the prediction on a scale of 0 to 1.0 based on factual accuracy. 
0.0 means completely wrong or hallucinated.
1.0 means perfectly accurate.
Output ONLY a JSON object with two keys: "reason" (string) and "score" (float).
Example:
{{"reason": "The answer is correct but missing the specific storage capacity.", "score": 0.9}}
"""

QUERY_PROMPT = """
You are an AI assistant answering questions strictly based on the following user background.

[User Background]
{bg}

[Question]
{query}

Answer concisely and factually. Do not invent information that is not supported by the background.
Output ONLY a JSON object with two keys: "reason" (string) and "answer" (string). reason can show what you think about 
Example:
{{"reason": "I can see that the user has a 512GB SSD in Q3.", "answer": "The user has a 512GB SSD."}}
"""