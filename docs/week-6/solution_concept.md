# Solution Concept - Week 6

## Concept

Same core solution with performance optimizations for gaming use case.

## Gaming Mode Features

- **Low CPU Mode** — Reduce detection frequency during gaming (target < 10% CPU)
- **Subtle Alerts** — Edge-glow visual alerts instead of overlays
- **Performance Priority** — Background thread with low priority
- **Privacy** — Important for streamers and content creators

## Implementation

- Adaptive frame rate: reduce from 30fps to 10-12fps during gaming
- Simplified pose model for better performance
- Alerts between game rounds/matches instead of during gameplay
- Option to pause monitoring during competitive matches

## Benefits

- Serves both office and gaming use cases
- No performance impact on games
- Maintains privacy and local processing
- Addresses previously underserved market
