# BASIC_JUDGE_PROMPT = """You are an expert evaluator. Your task is to rate the quality of an AI-generated answer based on a standard reference.

# [Question]: {query}
# [Standard Reference]: {reference}
# [AI Prediction]: {prediction}

# Rate the prediction on a scale of 1 to 10 based on factual accuracy and completeness.

# Scoring criteria:
# - 0-2: Mostly or completely incorrect. Contains major factual errors, hallucinations, or is irrelevant to the question.
# - 3-5: Partially correct but with significant mistakes, omissions, or misunderstandings. Core idea may be present but unreliable.
# - 6-8: Largely correct with minor inaccuracies or missing details. Overall understanding is good but not perfect.
# - 9-10: Fully accurate, complete, and consistent with the reference. No factual errors or meaningful omissions.

# Output ONLY a JSON object with two keys:
# - "reason": a brief explanation for the score
# - "score": a number from 1 to 10 (integer)

# Output format(JSON):
# {{"reason": "...", "score": ...}}
# """

BASIC_JUDGE_PROMPT = """You are an expert evaluator. Your task is to rate the quality of an AI-generated answer based on a standard reference.

[Question]: {query}
[Standard Reference]: {reference}
[AI Prediction]: {prediction}

Rate the prediction on a scale of 1 to 10 based on whether it correctly covers the KEY POINTS in the reference.

**Important Evaluation Principle:**
- Focus on whether the prediction captures the ESSENTIAL information from the reference.
- Additional details, elaborations, or supplementary information in the prediction should NOT be penalized, as long as they don't contradict the reference.
- Only deduct points for: missing key points, factual errors, or contradictions with the reference.

Scoring criteria:
- 0-2: Mostly or completely incorrect. Contains major factual errors, contradicts the reference, or misses almost all key points.
- 3-5: Partially correct but missing significant key points from the reference, or contains notable factual errors.
- 6-8: Covers most key points from the reference correctly. Minor omissions or inaccuracies may exist, but core information is accurate.
- 9-10: Fully covers all key points from the reference accurately. No factual errors or contradictions. (Extra details beyond the reference are acceptable.)

Output ONLY a JSON object with two keys:
- "reason": a brief explanation for the score (focus on which key points were hit or missed)
- "score": a number from 1 to 10 (integer)

Output format(JSON):
{{"reason": "...", "score": ...}}
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