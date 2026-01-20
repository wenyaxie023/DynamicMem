import json
import argparse
import os
from openai import OpenAI
from tqdm import tqdm

def load_data(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def construct_prompt(history, current_event):
    """
    Constructs the prompt for the model.
    History: List of previous app logs.
    Current Event: The event to predict response for.
    """
    # We instruct the model to act as the environment/app system.
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

def evaluate(data_path, output_path, model_name, port, limit=None, start_index=0):
    print(f"Loading data from {data_path}...")
    try:
        data = load_data(data_path)
    except FileNotFoundError:
        print(f"Error: File not found at {data_path}")
        return

    app_logs = data.get("app_logs", [])
    print(f"Total events: {len(app_logs)}")
    
    if limit:
        app_logs_to_process = app_logs[start_index : start_index + limit]
    else:
        app_logs_to_process = app_logs[start_index:]

    print(f"Processing {len(app_logs_to_process)} events starting from index {start_index}...")

    client = OpenAI(
        base_url=f"http://localhost:{port}/v1",
        api_key="EMPTY",
    )

    results = []
    # If output file exists, maybe load it? For now, we overwrite or start fresh.
    
    for i, current_event in enumerate(tqdm(app_logs_to_process)):
        global_index = start_index + i
        
        # History is everything before this event
        history = app_logs[:global_index]
        
        messages = construct_prompt(history, current_event)
        
        try:
            chat_completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=4096, 
                temperature=0.0,
            )
            prediction = chat_completion.choices[0].message.content
        except Exception as e:
            print(f"Error at index {global_index}: {e}")
            prediction = f"Error: {str(e)}"

        results.append({
            "event_id": current_event.get("event_id"),
            "global_index": global_index,
            "request_full": current_event,
            "ground_truth": current_event.get("response"),
            "prediction": prediction
        })
        
        # Save periodically
        if i % 10 == 0:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Kimi-Linear baseline")
    parser.add_argument("--data_path", type=str, default="../../app_log_518.json", help="Path to user data JSON")
    parser.add_argument("--output_path", type=str, default="kimi_results.json", help="Path to save results")
    parser.add_argument("--model", type=str, default="moonshotai/Kimi-Linear-48B-A3B-Instruct", help="Model name served by vLLM")
    parser.add_argument("--port", type=int, default=8000, help="vLLM server port")
    parser.add_argument("--limit", type=int, default=None, help="Number of events to evaluate")
    parser.add_argument("--start_index", type=int, default=0, help="Index to start evaluation from")
    
    args = parser.parse_args()
    
    evaluate(args.data_path, args.output_path, args.model, args.port, args.limit, args.start_index)
