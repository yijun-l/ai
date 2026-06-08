from mcp.server.fastmcp import FastMCP

# Initialize MCP server instance
mcp = FastMCP(
    "weather-service",
    host="127.0.0.1",
    port=8888,
    json_response=True,
    stateless_http=True
)

# Define exposed weather tool
@mcp.tool()
def get_weather(city: str) -> str:
    weather_data = {
        "beijing": "sunny, 25°C",
        "shanghai": "rainy, 22°C",
        "guangzhou": "cloudy, 28°C"
    }
    return weather_data.get(city.lower(), "Weather not found")

if __name__ == "__main__":
    # Start with official Streamable HTTP transport
    mcp.run(transport="streamable-http")