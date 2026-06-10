import torch
import re
import asyncio
from transformers import AutoTokenizer, AutoModelForCausalLM
from mcp.client.streamable_http import streamable_http_client
from mcp.client.session import ClientSession

# Load local LLM
model_path = "../llm/Qwen3.5-0.8B"
print("Loading LLM model...")

tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    local_files_only=True,
    device_map="cpu",
    torch_dtype="auto"
)

# MCP Server endpoint (RAG service deployed here)
MCP_SERVER_URL = "http://127.0.0.1:8889/mcp"

async def interactive_rag_mcp_client():
    print(f"=== RAG-MCP Interactive Client ===")
    print(f"MCP Server: {MCP_SERVER_URL}")
    print("Input your question, type exit to quit\n")

    # Persistent MCP connection
    async with streamable_http_client(MCP_SERVER_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            # Fetch all tools (RAG search tool is included)
            tools_res = await session.list_tools()
            tools = [{"type": "function", "function": t.model_dump()} for t in tools_res.tools]
            tool_names = [t["function"]["name"] for t in tools]
            print(f"Loaded MCP Tools: {tool_names}\n")

            while True:
                user_query = input("Your Question: ").strip()
                if user_query.lower() == "exit":
                    print("Client stopped.")
                    break
                if not user_query:
                    continue

                # Conversation history
                messages = [
                    {
                        "role": "system",
                        "content": "Answer user questions based on knowledge base. Use the RAG search tool first to get relevant context."
                    },
                    {"role": "user", "content": user_query}
                ]

                # Step 1: LLM decides to call RAG tool
                inputs = tokenizer.apply_chat_template(
                    messages,
                    tools=tools,
                    return_tensors="pt",
                    add_generation_prompt=True
                ).to(model.device)

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=256,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id
                    )

                llm_output = tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True
                )
                print(f"LLM Tool Call Instruction:\n{llm_output}\n")

                # Step 2: Parse RAG tool call
                if "<function=" not in llm_output:
                    print("No tool call triggered.\n" + "-" * 50 + "\n")
                    continue

                func_match = re.search(r"<function=(.*?)>", llm_output)
                if not func_match:
                    print("Invalid tool format.\n" + "-" * 50 + "\n")
                    continue

                tool_name = func_match.group(1).strip()
                param_pairs = re.findall(r"<parameter=(.*?)>\s*(.*?)\s*</parameter>", llm_output, re.DOTALL)
                params = {k.strip(): v.strip() for k, v in param_pairs}

                # Step 3: Call RAG tool via MCP (Core RAG retrieval)
                print(f"Calling MCP RAG Tool: {tool_name}")
                print(f"Search Params: {params}\n")
                rag_result = await session.call_tool(name=tool_name, arguments=params)
                context = rag_result.content[0].text if rag_result.content else "No relevant context found"
                print(f"RAG Retrieved Context:\n{context}\n")

                # Step 4: Append tool result to history, generate final answer
                messages.append({"role": "assistant", "content": llm_output})
                messages.append({"role": "tool", "name": tool_name, "content": context})

                # Generate final answer with retrieved context
                final_inputs = tokenizer.apply_chat_template(
                    messages,
                    tools=tools,
                    return_tensors="pt",
                    add_generation_prompt=True
                ).to(model.device)

                with torch.no_grad():
                    final_outputs = model.generate(
                        **final_inputs,
                        max_new_tokens=512,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id
                    )

                final_answer = tokenizer.decode(
                    final_outputs[0][final_inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True
                )
                print(f"Final Answer:\n{final_answer}")
                print("-" * 60 + "\n")

if __name__ == "__main__":
    asyncio.run(interactive_rag_mcp_client())