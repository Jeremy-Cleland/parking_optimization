# API Setup Guide

## No Setup Required (Default)

The system works out-of-the-box with intelligent traffic simulation:

```python
# No APIs needed - uses realistic traffic patterns
from core.traffic_manager import TrafficManager
traffic_manager = TrafficManager()  # Uses fallback by default
```

## Optional: Real Traffic Data

For real-time traffic data, choose one provider:

### TomTom (Recommended)

- Free: 2,500 calls/day, no credit card
- Setup: [developer.tomtom.com](https://developer.tomtom.com/) → Get API key

### Google Maps

- Free: $200/month credit (40k requests)
- Setup: [console.cloud.google.com](https://console.cloud.google.com/) → Enable Directions API → Get API key
- ⚠️ Requires credit card

### Mapbox

- Free: 100k requests/month, no credit card
- Setup: [mapbox.com](https://www.mapbox.com/) → Get access token

## Configuration

Create `.env` file:

```bash
# Choose one or more
TOMTOM_API_KEY=your_key_here
GOOGLE_MAPS_API_KEY=your_key_here
MAPBOX_ACCESS_TOKEN=your_token_here

# Optional: Set preferred provider (default: tomtom)
MAP_PROVIDER=tomtom
```

## Test Setup

```bash
make test
```

That's it! The system automatically falls back to simulation if APIs are unavailable.
