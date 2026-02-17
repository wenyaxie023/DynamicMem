PROMPT = """# Task
Given a question and a list of user app logs, answer the question using ONLY the information in the logs.

# Input
Question:
{{ question }}

User App Logs (each log has: app_log_id, timestamp, app_name, api_name, content):
{{ context }}

# Output (JSON ONLY; no markdown, no extra text)
Return a single JSON object with this schema:
{
  "evidence": [
    {
      "app_log_id": "string",
      "timestamp": "string (as in the log; prefer ISO-8601 if present)",
      "app_name": "string",
      "api_name": "string",
      "anchor": "string" // [required, a short justification grounded in the context.]
    }
  ],
  "answer": "string"
}

# Missing-field policy
- If you don't know the app_log_id / timestamp / app_name / api_name, you should set them to null.
"""