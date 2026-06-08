---
name: weather-intent-advisor
description: Converts raw weather API data into actionable human decisions (packing, transit, scheduling). Use when a user asks about the weather/climate, to prevent the agent from just dumping unhelpful raw numbers.
---

# Weather Intent Advisor

## Overview
People don't ask for weather to get numbers; they ask to make decisions (what to wear, whether to drive, should they cancel an event). This skill forces the agent to extract the context, query the MCP tool precisely, and translate meteorological data into human impact.

## When to Use
- **DO USE:** User asks for current weather, forecasts, or climate of a location.
- **DO NOT USE:** Generic scientific queries ("Why does it rain?"), or when the user explicitly demands raw API JSON.

## The Process

### 1. Adaptive Tool Execution
Parse the location and time dimension accurately. Call the MCP tool with precise parameters (e.g., use daily forecast for "this weekend", hourly for "this afternoon"). **Never hallucinate data if the tool fails.**

### 2. Translate Metrics to Impact (Mental Matrix)
Never copy-paste raw data. Translate numbers into constraints:
- **Temp swings > 10°C:** Advise layered clothing.
- **Wind > 15m/s:** Advise against umbrellas; warn about flight/transit delays.
- **Humidity > 80% & Temp > 30°C:** High Muggy Index; advise breathable clothes/hydration.
- **Precipitation > 5mm/h:** Traffic congestion expected; shift outdoor plans.

### 3. The Structured Deliverable
Format the final response strictly as follows. Omit the `⚠️ Alerts` section if conditions are normal.

```
### 🌤️ Weather Context: [City Name] ([Date/Time])
**The Numbers:** [1-line summary of temperature, conditions, and anomalies]

### 💡 What this means for you:
- **👕 Clothing:** [Specific guidance based on temp/wind]
- **🚶‍♂️ Activity/Travel:** [Impact on transit, outdoor plans]
- **⚠️ Alerts:** [Only if UV, Air Quality, or Storms cross dangerous thresholds]
```