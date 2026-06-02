# Agent

A traditional LLM receives a prompt and generates a response. Once the response is produced, the interaction ends.

An AI Agent extends this capability. Instead of generating a single response, it can repeatedly reason, take actions, observe results, and continue working until a goal is achieved.

For example, when asked: `Find the latest Linux kernel release and summarize its new features.`

A normal LLM can only answer using its existing knowledge. An AI Agent can:

1. Search the web.
2. Read the results.
3. Extract relevant information.
4. Generate a summary.
5. Return the final answer.

In short, an AI Agent combines reasoning with actions.

## Core Components

Although implementations vary, most AI Agents contain four fundamental components.

### 1. LLM 

The LLM acts as the agent's brain. It interprets user requests, reasons about the next step, and decides which action to take.

### 2. Tools

Tools allow the agent to interact with external systems and obtain information beyond its training data.

Without tools, an LLM can only generate text. With tools, it can interact with the real world.

### 3. Memory

Memory stores information across multiple steps.

- **Short-term memory**: remembers the current conversation and the steps it has taken so far.

- **Long-term memory**: stores past experiences and user preferences for future use.


### 4. Loop

The loop is the execution mechanism of an Agent. It repeatedly reasons, selects actions, executes tools, and processes observations until the goal is achieved.

Unlike a traditional chatbot, an Agent may perform many reasoning and action cycles before producing a final answer.

## How Agent Think

When solving a complex task, an Agent typically follows one of two reasoning paradigms.

### ReAct (Reasoning + Acting)

**ReAct** is a dynamic, step-by-step approach. The Agent continuously alternates between reasoning and acting.

`Thought >> Action >> Observation`

The Agent determines the next action based on the latest observation. This makes ReAct well suited for tasks involving information retrieval, debugging, troubleshooting, and other unpredictable environments.

### Plan-and-Execute

**Plan-and-Execute** separates reasoning into two phases:
1. **Planning**: create a complete execution plan.
2. **Execution**: perform each step in the plan.

Compared with ReAct, Plan-and-Execute is generally more effective for long and complex tasks because the overall strategy is determined before execution begins.

## Practical Demo: Weather Agent

This lightweight demo builds a simplest weather Agent. It defines a local mock weather tool, lets LLM judge when to invoke the tool, then runs loop iteration to finish the weather inquiry automatically.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

# ----------------------
# Load Model
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
# Tool Definition
# ----------------------
def get_weather(city):
    """Simple weather tool"""
    weather_data = {
        "beijing": "sunny, 25°C",
        "shanghai": "rainy, 22°C",
        "guangzhou": "cloudy, 28°C"
    }
    return weather_data.get(city.lower(), "Weather data not found")

# ----------------------
# LLM Helper
# ----------------------
def ask_llm(messages):
    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        return_dict=True,
        add_generation_prompt=True
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)

# ----------------------
# Agent Loop
# ----------------------
system_prompt = """
You are an AI Agent.

You have one tool:
WEATHER(city)

Rules:
1. Use WEATHER tool to get real-time weather.
2. Put the city name inside WEATHER(...).
3. After getting Observation, give the final answer.

Example:
User: What's the weather in Shanghai?
Assistant: ACTION: WEATHER(Shanghai)
"""

messages = [{"role": "system", "content": system_prompt}]
user_question = "What is the weather today in Beijing?"
messages.append({"role": "user", "content": user_question})

# Run up to 5 steps
for step in range(5):
    print(f"\n=== Agent Step {step+1} ===")
    response = ask_llm(messages)
    print("LLM:\n", response)

    # Match tool call
    match = re.search(r"ACTION:\s*WEATHER\((.*)\)", response, re.DOTALL)
    if match:
        city = match.group(1).strip()
        print("\nTool Call:", city)
        result = get_weather(city)
        print("Tool Result:", result)

        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": f"Observation: weather returned {result}"})
    else:
        print("\nFinal Answer:\n", response)
        break
```

Execute the script directly. 
- **Step 1**: makes LLM output tool invocation command to query Beijing’s weather.
- **Step 2**: receives tool feedback and generates the final natural language reply, fully simulating standard Agent workflow.

```shell
$ python3 agent.py
Loading model...
[transformers] The fast path is not available because one of the required library is not installed. Falling back to torch implementation. To install follow https://github.com/fla-org/flash-linear-attention#installation and https://github.com/Dao-AILab/causal-conv1d
Loading weights: 100%|██████████████████████| 320/320 [00:00<00:00, 1695.48it/s]

=== Agent Step 1 ===
LLM:
 ACTION: WEATHER(Beijing)


Tool Call: Beijing
Tool Result: sunny, 25°C

=== Agent Step 2 ===
LLM:
 The weather today in Beijing is sunny, with a temperature of 25°C.


Final Answer:
 The weather today in Beijing is sunny, with a temperature of 25°C.
```