# Function Calling

In the previous article, we built a simple AI Agent using **prompt engineering**.

The Agent was instructed to generate actions in a predefined format: `ACTION: WEATHER(Beijing)`, the runtime then parsed the text using regular expressions:

```python
match = re.search(
    r"ACTION:\s*WEATHER\((.*)\)",
    response
)
```

Although this approach works for simple demonstrations, it becomes difficult to maintain as the number of tools grows.

Modern AI Agents therefore rely on **Function Calling**, a structured protocol that allows LLMs to invoke tools in a machine-readable format.

The key idea is simple: **The LLM decides what to call, while the runtime performs the actual execution.**

## The Core Structure

Function Calling relies on a simple four-step workflow:

**1. The Definition (JSON Schema)**

You provide the AI with a strict "API document." This JSON schema defines the tool's **name**, **description** (so the AI knows when to use it), and the required **parameters**.

```json
{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the real-time weather of a specified city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name, e.g., Beijing, Shanghai"
                }
            },
            "required": ["city"]
        }
    }
}
```

**2. The Trigger (Structured Output)**

When a user asks a relevant question "**What is the weather today in Beijing?**", the LLM identifies the need for a tool and generates a structured function call with the required arguments "**Beijing**".

```html
<tool_call>
    <function=get_weather>
        <parameter=city>Beijing</parameter>
    </function>
</tool_call>
```

**3. The Execution (Local Code)**

The LLM itself does not execute code. Instead, the runtime intercepts the function call, extracts the parameter "**Beijing**", executes the corresponding local function, and returns the raw result "**sunny, 25°C**" back to the model.

```python
def get_weather(city):
    """Simple weather tool"""
    weather_data = {
        "beijing": "sunny, 25°C",
        "shanghai": "rainy, 22°C",
        "guangzhou": "cloudy, 28°C"
    }
    return weather_data.get(city.lower(), "Weather data not found")
```

**4. The Final Response (Observation)**

The LLM receives the returned result as an observation and converts it into a natural, human-friendly response for the user: "**Today in Beijing is sunny, with a temperature of 25°C.**".

## Practical Demo: Weather Agent with Function Calling

This example shows a simple Weather Agent built with Function Calling.

```python
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

# ----------------------
# 1. Load Model
# ----------------------
model_path = "./Qwen3.5-0.8B"
print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    local_files_only=True,
    device_map="cpu",
    torch_dtype="auto"
)

# ----------------------
# 2. Tool Definition (JSON Schema format)
# ----------------------
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the real-time weather of a specified city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name, e.g., Beijing, Shanghai"
                    }
                },
                "required": ["city"]
            }
        }
    }
]

# The actual execution function
def get_weather(city):
    """Simple weather tool"""
    weather_data = {
        "beijing": "sunny, 25°C",
        "shanghai": "rainy, 22°C",
        "guangzhou": "cloudy, 28°C"
    }
    return weather_data.get(city.lower(), "Weather data not found")

# ----------------------
# 3. Agent Execution Loop
# ----------------------
messages = [
    {"role": "system", "content": "You are a helpful AI Agent."},
    {"role": "user", "content": "What is the weather today in Beijing?"}
]

max_steps = 5

for step in range(max_steps):
    print(f"\n=== Agent Step {step + 1} ===")

    # Assemble prompt, pass the 'tools' parameter directly
    inputs = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        return_tensors="pt",
        add_generation_prompt=True,
        return_dict=True
    ).to(model.device)

    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    # Extract only the newly generated tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    print("Raw LLM Output:\n", response_text)

    # Check if the output contains the special tool call tag
    if "<tool_call>" in response_text:
        print("\n[Function Calling Triggered]")

        # Extract the function name using regex
        func_match = re.search(r"<function=(.*?)>", response_text)
        if func_match:
            func_name = func_match.group(1).strip()
            print(f"Target Function: {func_name}")

            # Extract parameter names and values using regex
            param_matches = re.findall(r"<parameter=(.*?)>\s*(.*?)\s*</parameter>", response_text, re.DOTALL)
            arguments = {p_name.strip(): p_val.strip() for p_name, p_val in param_matches}
            print(f"Extracted Parameters: {arguments}")

            # Execute local code
            if func_name == "get_weather" and "city" in arguments:
                city = arguments["city"]
                result = get_weather(city)
                print(f"Execution Result: {result}")

                # Append the assistant's action request and the tool's result to the conversation
                messages.append({"role": "assistant", "content": response_text})
                messages.append({"role": "tool", "name": "get_weather", "content": result})

                print("-> Passing the execution result back to LLM for final summarization...")
                # The loop continues to the next step, feeding the result back to the model
                continue
            else:
                print("Error: Function name mismatch or missing parameters.")
                break
        else:
            print("Error: Failed to parse function name.")
            break

    else:
        # If no <tool_call> tag is detected, it means the model has reached its final conclusion
        print("\n[Final Semantic Answer]:\n", response_text)
        break # Exit the loop since the task is completed
```

Run it.

```shell
$ python3 func_call.py
Loading model...
[transformers] The fast path is not available because one of the required library is not installed. Falling back to torch implementation. To install follow https://github.com/fla-org/flash-linear-attention#installation and https://github.com/Dao-AILab/causal-conv1d
Loading weights: 100%|██████████████████████| 320/320 [00:00<00:00, 1647.25it/s]

=== Agent Step 1 ===
Raw LLM Output:
 <tool_call>
<function=get_weather>
<parameter=city>
Beijing
</parameter>
</function>
</tool_call>


[Function Calling Triggered]
Target Function: get_weather
Extracted Parameters: {'city': 'Beijing'}
Execution Result: sunny, 25°C
-> Passing the execution result back to LLM for final summarization...

=== Agent Step 2 ===
Raw LLM Output:
 Today in Beijing is sunny, with a temperature of 25°C.


[Final Semantic Answer]:
 Today in Beijing is sunny, with a temperature of 25°C.
```