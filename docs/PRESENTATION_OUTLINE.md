# Video Presentation Outline (8-10 minutes)

## Real-Time Collaborative Parking Space Optimization

### Slide 1: Title & Team (30 seconds)

- Project title
- Team members
- CIS 505 - Summer 2025

### Slide 2: Problem & Motivation (1 minute)

- Urban parking challenges
  - 30% of traffic is parking search
  - 17 minutes average search time
  - Environmental impact
- Our solution: Algorithmic optimization
- Show city traffic image

### Slide 3: System Overview (1 minute)

- Four integrated components:
  1. Dynamic Pricing (Game Theory)
  2. Smart Routing (Modified A*)
  3. Demand Prediction (DP)
  4. City Coordination (D&C)
- System architecture diagram

### Slide 4: Dynamic Pricing Algorithm (1.5 minutes)

```
price = base × occupancy × competition × demand × time
```

- Game theory for zone competition
- Approximation algorithm (1.5x bound)
- Show pricing heat map
- Complexity: O(z²)

### Slide 5: Route Optimization (1.5 minutes)

- Modified A* with dynamic weights
- Real-time traffic updates
- Multi-objective: time + walk distance
- Show path visualization
- Complexity: O((V+E)log V)

### Slide 6: Demand Prediction (1 minute)

- Dynamic Programming approach
- State: (time, occupancy, weather)
- Historical pattern learning
- Show DP table visualization
- Complexity: O(t×s²×w)

### Slide 7: City Coordination (1 minute)

- Divide-and-conquer strategy
- Parallel district optimization
- Inter-district balancing
- Show district map
- Complexity: O(z²/d + d²)

### Slide 8: Results & Performance (1.5 minutes)

- Simulation: 20 zones, 500 drivers, 8 hours
- Key metrics:
  - Search time: 4.2 min (73% reduction)
  - Occupancy: 83.5% (near optimal)
  - Success rate: 94.3%
  - Revenue: +22% vs static
- Show dashboard screenshot

### Slide 9: Complexity Analysis (45 seconds)

- Show complexity comparison chart
- Overall: O(D×V log V + z²)
- Real-time performance: <100ms
- Scalability discussion

### Slide 10: Conclusions & Future Work (45 seconds)

- Successfully integrated 4 algorithms
- Significant improvements in all metrics
- Future: ML integration, mobile app
- Questions?

---

## Demo Script (if showing live)

1. **Start simulation** (30 sec)

   ```bash
   make run
   ```

2. **Show real-time updates** (30 sec)
   - Point out occupancy changes
   - Price adjustments
   - Driver routing

3. **Show visualizations** (30 sec)
   - Open dashboard image
   - Explain key metrics

---

## Key Speaking Points

### Technical Depth

- Emphasize algorithmic choices
- Explain complexity trade-offs
- Highlight innovations

### Practical Impact

- Real-world applicability
- Environmental benefits
- Economic advantages

### Team Collaboration

- Division of work
- Integration challenges
- Learning outcomes

---

## Visual Aids Needed

1. System architecture diagram
2. Algorithm flowcharts
3. Simulation screenshots
4. Performance graphs
5. Complexity analysis charts

---

## Backup Slides (if time permits)

- Detailed pseudocode
- Mathematical formulations
- Additional results
- Implementation challenges
