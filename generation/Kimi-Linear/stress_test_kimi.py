import json
import argparse
import os
import time
from openai import OpenAI

def load_data(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def construct_prompt(history, current_event):
    messages = [
        {
            "role": "system", 
            "content": (
                "You are an intelligent operating system agent. "
                "You have access to the full history of user interactions across various applications. "
                "Your task is to predict the 'response' for the current user 'request' based on the history and context. "
                "Return ONLY the JSON object for the response, with no explanation or markdown."
            )
        },
        {
            "role": "user", 
            "content": f"History of events:\n{json.dumps(history, indent=2)}\n\n"
                       f"Current App: {current_event.get('app_name')}\n"
                       f"Current API: {current_event.get('api_name')}\n"
                       f"Current Request:\n{json.dumps(current_event.get('request'), indent=2)}\n\n"
                       "Generate the Response JSON:"
        }
    ]
    return messages

def stress_test(data_path, model_name, port):
    print(f"Loading stress data from {data_path}...")
    data = load_data(data_path)
    app_logs = data.get("app_logs", [])
    
    # We want the LAST event to maximize context
    current_event = app_logs[-1]
    history = app_logs[:-1] # All preceding events
    
    print(f"Total history length: {len(history)} events")
    
    # Construct prompt size estimation
    messages = construct_prompt(history, current_event)
    approx_chars = sum([len(m['content']) for m in messages])
    print(f"Approximate prompt length in chars: {approx_chars}")
    print(f"Approximate prompt tokens (char/4): {approx_chars / 4}")

    client = OpenAI(
        base_url=f"http://localhost:{port}/v1",
        api_key="EMPTY",
    )

    print("Sending request to vLLM...")
    start_time = time.time()
    try:
        chat_completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=100, # We don't need a long response, just the processing
            temperature=0.0,
        )
        print("Success!")
        print(f"Time taken: {time.time() - start_time:.2f}s")
        print("Response prefix:", chat_completion.choices[0].message.content[:200])
    except Exception as e:
        print(f"Error during stress test: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="../../app_log_stress_test.json")
    parser.add_argument("--model", type=str, default="moonshotai/Kimi-Linear-48B-A3B-Instruct")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    
    stress_test(args.data_path, args.model, args.port)
