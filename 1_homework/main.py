import os
from dataclasses import dataclass
from pathlib import Path
from openai import OpenAI
import json
import yfinance as yf
from dotenv import load_dotenv
from typing import List, Dict, Any
import requests


# Load environment variables
if Path('.env.local').exists():
    load_dotenv('.env.local')
else:
    load_dotenv()

WEATHER_API_BASE_URL = "https://api.weatherapi.com/v1/"

# Initialize OpenAI client
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

@dataclass
class WeatherApiLocation:
    name: str
    region: str
    country: str
    lat: float
    lon: float
    tz_id: str
    localtime_epoch: int
    localtime: str

@dataclass
class WeatherApiCurrent:
    last_updated_epoch: int
    last_updated: str
    temp_c: float
    is_day: int
    humidity: int

@dataclass
class Day:
    maxtemp_c: float
    mintemp_c: float
    avgtemp_c: float

@dataclass
class WeatherApiForecastDay:
    date: str
    date_epoch: int
    day: Day

@dataclass
class WeatherApiForecast:
    forecastday: List[WeatherApiForecastDay]

@dataclass
class WeatherApiResponse:
    location: WeatherApiLocation
    current: WeatherApiCurrent
    forecast: WeatherApiForecast | None = None



# Function Implementations
def get_current_temperature_for_city(city_name: str):
    params = {
        "key": os.environ.get("WEATHER_API_KEY"),
        "q": city_name,
        "aqi": "no" # Disable air quality index data
    }

    url = WEATHER_API_BASE_URL + "current.json"

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise an error for bad responses

        data = response.json()
        # filter out the keys that are not in the dataclass
        location_data = {
            key: data["location"][key]
            for key in WeatherApiLocation.__annotations__.keys()
        }

        # filter out the keys that are not in the dataclass
        current_data = {
            key: data["current"][key]
            for key in WeatherApiCurrent.__annotations__.keys()
        }

        location = WeatherApiLocation(**location_data)
        current = WeatherApiCurrent(**current_data)

        return {'city': location.name, 'temperature_c': current.temp_c}
    except requests.exceptions.HTTPError as e:
        print(f"Chyba: {e}")
        return None

def get_temperature_forecast_for_city(city_name: str, days: int):
    if not 1 <= days <= 14:
        raise ValueError(f"The number of days must be between 1 and 14, but was entered: {days}")

    params = {
        "key": os.environ.get("WEATHER_API_KEY"),
        "q": city_name,
        'days': days,
        "aqi": "no" # Disable air quality index data
    }

    url = WEATHER_API_BASE_URL + "forecast.json"

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise an error for bad responses

        data = response.json()
        # filters out the keys that are not in the dataclass
        location_data = {
            key: data["location"][key]
            for key in WeatherApiLocation.__annotations__.keys()
        }

        # filters out the keys that are not in the dataclass
        current_data = {
            key: data["current"][key]
            for key in WeatherApiCurrent.__annotations__.keys()
        }

        location = WeatherApiLocation(**location_data)
        current = WeatherApiCurrent(**current_data)

        return {'city': location.name, 'temperature_c': current.temp_c}
    except requests.exceptions.HTTPError as e:
        print(f"Chyba: {e}")
        return None

# Define custom tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_temperature_for_city",
            "description": "Use this function to get the current temperature in city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city_name": {
                        "type": "string",
                        "description": "The city name, e.g. Praha",
                    }
                },
                "required": ["city_name"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_temperature_forecast_for_city",
            "description": "Use this function to get a temperature forecast for a city, with 1 to 14-day forecasts available.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city_name": {
                        "type": "string",
                        "description": "The city name, e.g. Praha",
                    },
                    "days": {
                        "type": "string",
                        "description": "Number of days for the forecast (1-14).",
                    },
                },
                "required": ["city_name", "days"],
            },
        }
    },
]

available_functions = {
    "get_current_temperature_for_city": get_current_temperature_for_city,
    "get_temperature_forecast_for_city": get_temperature_forecast_for_city,
}


class ReactAgent:
    """A ReAct (Reason and Act) agent that handles multiple tool calls."""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.max_iterations = 10  # Prevent infinite loops

    def run(self, messages: List[Dict[str, Any]]) -> str:
        """
        Run the ReAct loop until we get a final answer.

        The agent will:
        1. Call the LLM
        2. If tool calls are returned, execute them
        3. Add results to conversation and repeat
        4. Continue until LLM returns only text (no tool calls)
        """
        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1
            print(f"\n--- Iteration {iteration} ---")

            # Call the LLM
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                parallel_tool_calls=False
            )

            response_message = response.choices[0].message
            print(f"LLM Response: {response_message}")

            # Check if there are tool calls
            if response_message.tool_calls:
                # Add the assistant's message with tool calls to history
                messages.append({
                    "role": "assistant",
                    "content": response_message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            }
                        }
                        for tc in response_message.tool_calls
                    ]
                })

                # Process ALL tool calls (not just the first one)
                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    tool_id = tool_call.id

                    print(f"Executing tool: {function_name}({function_args})")

                    # Call the function
                    function_to_call = available_functions[function_name]
                    function_response = function_to_call(**function_args)

                    print(f"Tool result: {function_response}")

                    # Add tool response to messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "name": function_name,
                        "content": json.dumps(function_response),
                    })

                # Continue the loop to get the next response
                continue

            else:
                # No tool calls - we have our final answer
                final_content = response_message.content

                # Add the final assistant message to history
                messages.append({
                    "role": "assistant",
                    "content": final_content
                })

                print(f"\nFinal answer: {final_content}")
                return final_content

        # If we hit max iterations, return an error
        return "Error: Maximum iterations reached without getting a final answer."


def main():
    print(get_temperature_forecast_for_city('Kralupy nad Vltavou', 3))
    '''
    # Create a ReAct agent
    agent = ReactAgent()

    # Example 1: Simple query (single tool call)
    print("=== Example 1: Single Tool Call ===")
    messages1 = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "What is the current temperature in Kralupy nad Vltavou?"},
    ]

    result1 = agent.run(messages1.copy())
    '''


if __name__ == "__main__":
    main()
