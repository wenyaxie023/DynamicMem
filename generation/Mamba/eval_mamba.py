import json
import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
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
    # Mamba is a base model usually, but "state-spaces/mamba-2.8b-hf" or similar might be instruction tuned?
    # The user specified "state-spaces/mamba-2.8b". This is a base model.
    # It might not follow chat templates well. We will try a simple completion format.
    
    system_text = (
        "You are an intelligent operating system agent. "
        "You have access to the full history of user interactions across various applications. "
        "Your task is to predict the 'response' for the current user 'request' based on the history and context. "
        "Return ONLY the JSON object for the response, with no explanation or markdown.\n\n"
    )
    
    user_text = (
        f"History of events:\n{json.dumps(history, indent=2)}\n\n"
        f"Current App: {current_event.get('app_name')}\n"
        f"Current API: {current_event.get('api_name')}\n"
        f"Current Request:\n{json.dumps(current_event.get('request'), indent=2)}\n\n"
        "Generate the Response JSON:\n"
    )
    
    return system_text + user_text

def evaluate(data_path, output_path, model_name, limit=None, start_index=0, device="cuda"):
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

    print(f"Loading model {model_name}...")
    # Mamba uses GPT-NeoX tokenizer. Using the canonical one to avoid config issues.
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
    model.eval()

    results = []
    
    for i, current_event in enumerate(tqdm(app_logs_to_process)):
        global_index = start_index + i
        
        # History is everything before this event
        history = app_logs[:global_index]
        
        prompt = construct_prompt(history, current_event)
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_len = inputs.input_ids.shape[1]
        
        # Mamba context limit check? Mamba is technically infinite context but limited by VRAM/impl.
        # We'll just run it.
        
        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=500, # Enough for a JSON response
                    do_sample=False,
                    temperature=0.0
                )
            
            # Slice off the input
            prediction = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
            
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
    parser = argparse.ArgumentParser(description="Evaluate Mamba baseline")
    parser.add_argument("--data_path", type=str, default="../../app_log_518.json", help="Path to user data JSON")
    parser.add_argument("--output_path", type=str, default="mamba_results.json", help="Path to save results")
    parser.add_argument("--model", type=str, default="state-spaces/mamba-2.8b-hf", help="Mamba model name")
    parser.add_argument("--limit", type=int, default=None, help="Number of events to evaluate")
    parser.add_argument("--start_index", type=int, default=0, help="Index to start evaluation from")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on")
    
    args = parser.parse_args()
    
    evaluate(args.data_path, args.output_path, args.model, args.limit, args.start_index, args.device)
