import json
import os
import requests
from dotenv import load_dotenv
from openai import OpenAI
from langsmith.wrappers import wrap_openai
from langsmith import traceable

# Load .env
load_dotenv()

# Initialize Gemini (OpenAI-compatible client)
client = wrap_openai(OpenAI(
    api_key=os.getenv("GOOGLE_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
))

# ------------------ TOOL DEFINITIONS ------------------


@traceable
def run_command(command):
    print("🔨 Tool Called: run_command", command)
    result = os.popen(command).read()
    return result


@traceable
def get_weather(city: str):
    print("🔨 Tool Called: get_weather", city)
    url = f"https://wttr.in/{city}?format=%C+%t"
    response = requests.get(url)
    if response.status_code == 200:
        return f"The weather in {city} is {response.text.strip()}."
    return "Something went wrong"


@traceable
def add(x, y):
    print("🔨 Tool Called: add", x, y)
    return x + y


available_tools = {
    "get_weather": {
        "fn": get_weather,
        "description": "Takes a city name as input and returns the current weather for the city."
    },
    "run_command": {
        "fn": run_command,
        "description": "Takes a shell command as input and executes it on the system."
    },
    "add": {
        "fn": add,
        "description": "Adds two numbers and returns the result."
    }
}

# ------------------ SYSTEM PROMPT ------------------
system_prompt = """
You are a helpful AI Assistant specialized in resolving user queries.
You operate in a cycle of: plan → action → observe → output.

For the given user query and available tools:
- Plan the step-by-step execution.
- Select the most relevant tool.
- Perform an action to call that tool.
- Wait for the observation (result).
- Then produce the final resolved answer.

Rules:
- Always follow the Output JSON Format.
- Always perform one step at a time.
- Carefully analyze the query before acting.

Output JSON Format:
{
    "step": "string",
    "content": "string",
    "function": "The function name if the step is 'action'",
    "input": "The input parameter for the function"
}

Available Tools:
- get_weather(city): Returns the current weather for the city.
- run_command(command): Executes a system command.
- add(x, y): Adds two numbers and returns the result.

Example:
User Query: What is the weather of New York?
Output: {"step": "plan", "content": "User wants weather info for New York."}
Output: {"step": "plan", "content": "I should call get_weather."}
Output: {"step": "action", "function": "get_weather", "input": "New York"}
Output: {"step": "observe", "output": "12°C, Sunny"}
Output: {"step": "output", "content": "The weather in New York is 12°C and sunny."}
"""

# ------------------ MAIN CHAT LOOP ------------------
messages = [{"role": "system", "content": system_prompt}]

while True:
    user_query = input("> ")
    if user_query.lower() in ["exit", "quit"]:
        print("👋 Exiting Gemini Agent.")
        break

    messages.append({"role": "user", "content": user_query})

    while True:
        response = client.chat.completions.create(
            model="gemini-2.0-flash-exp",  # use Gemini model
            response_format={"type": "json_object"},
            messages=messages
        )

        parsed_output = json.loads(response.choices[0].message.content)
        messages.append(
            {"role": "assistant", "content": json.dumps(parsed_output)})

        step = parsed_output.get("step")

        if step == "plan":
            print(f"🧠: {parsed_output.get('content')}")
            continue

        if step == "action":
            tool_name = parsed_output.get("function")
            tool_input = parsed_output.get("input")

            if tool_name in available_tools:
                try:
                    fn = available_tools[tool_name]["fn"]
                    if isinstance(tool_input, str):
                        output = fn(tool_input)
                    elif isinstance(tool_input, list):
                        output = fn(*tool_input)
                    else:
                        output = fn(tool_input)
                    messages.append({
                        "role": "assistant",
                        "content": json.dumps({"step": "observe", "output": output})
                    })
                except Exception as e:
                    messages.append({
                        "role": "assistant",
                        "content": json.dumps({"step": "observe", "output": f"Error: {str(e)}"})
                    })
                continue

        if step == "output":
            print(f"🤖: {parsed_output.get('content')}")
            break
