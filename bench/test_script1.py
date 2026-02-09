import os
import json
from openai import OpenAI
from datetime import datetime
from zoneinfo import ZoneInfo
from bench.backend.utils.clients import client_advanced, client_deepseek1


# Initialize the Azure OpenAI client
client = OpenAI(
    base_url="https://api.cometapi.com/v1",
    api_key="sk-A9mOxHmhV3ZRkr6Dchs86dStqQqTCWySgMdx91eK9hd2xr4x"
)

# Define the deployment you want to use for your chat completions API calls

deployment_name = "gpt-5"

# Simplified timezone data
TIMEZONE_DATA = {
    "tokyo": "Asia/Tokyo",
    "san francisco": "America/Los_Angeles",
    "paris": "Europe/Paris"
}


def get_current_time(location):
    """Get the current time for a given location"""
    print(f"get_current_time called with location: {location}")
    location_lower = location.lower()

    for key, timezone in TIMEZONE_DATA.items():
        if key in location_lower:
            print(f"Timezone found for {key}")
            current_time = datetime.now(ZoneInfo(timezone)).strftime("%I:%M %p")
            return json.dumps({
                "location": location,
                "current_time": current_time
            })

    print(f"No timezone data found for {location_lower}")
    return json.dumps({"location": location, "current_time": "unknown"})


def run_conversation():
    # Initial user message
    messages = [{"role": "user", "content": "What's the current time in San Francisco"}]  # Single function call
    # messages = [{"role": "user", "content": "What's the current time in San Francisco, Tokyo, and Paris?"}] # Parallel function call with a single tool/function defined

    # Define the function for the model
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "Get the current time in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city name, e.g. San Francisco",
                        },
                    },
                    "required": ["location"],
                },
            }
        }
    ]

    # First API call: Ask the model to use the function
    response = client_deepseek1.chat.completions.create(
        model="deepseek-reasoner",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    # Process the model's response
    response_message = response.choices[0].message
    messages.append(response_message)

    print("Model's response:")
    print(response_message)

    # Handle function calls
    if response_message.tool_calls:
        for tool_call in response_message.tool_calls:
            if tool_call.function.name == "get_current_time":
                function_args = json.loads(tool_call.function.arguments)
                print(f"Function arguments: {function_args}")
                time_response = get_current_time(
                    location=function_args.get("location")
                )
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": "get_current_time",
                    "content": time_response,
                })
    else:
        print("No tool calls were made by the model.")

        # Second API call: Get the final response from the model
    final_response = client_deepseek1.chat.completions.create(
        model="deepseek-reasoner",
        messages=messages,
    )

    return final_response.choices[0].message.content


# Run the conversation and print the result
print(run_conversation())