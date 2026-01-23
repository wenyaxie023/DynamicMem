PROMPT = """# Question:
{{ question }}

# User App Logs:
{{ context }}

Return JSON only, with this schema:
{
  "evidence": [{"app_log_id": string, "supporting_content": string}],  # List of dicts; supporting_content should preserve key supporting content as faithfully as possible, and should be the exact content of the app log
  "answer": string
}
"""