# Traffic Signal Optimization using PSO

## Overview
This project focuses on simulating a simplified traffic network and improving traffic signal timings using Particle Swarm Optimization (PSO).  
The goal is to reduce congestion by minimizing the total waiting time of vehicles at intersections.

---

## Project Idea
The system models a small traffic network consisting of intersections connected by roads.  
Each intersection has a traffic light, and vehicles arrive over time and wait in queues.  

The simulation runs step by step, where:
- Vehicles arrive at intersections
- Traffic lights control whether vehicles move or wait
- Vehicles move between intersections with a small delay

At the end, the system calculates the total waiting time, which is used later as a fitness value for optimization.

---

## System Components

### Traffic Simulation
- TrafficLight: controls the green/red signal timing  
- Intersection: manages vehicle queues and movement  
- Road: transfers vehicles between intersections with delay  
- TrafficNetwork: runs the full simulation and computes total waiting time  

### Optimization (PSO)
- Particle: represents a candidate solution (signal timings)  
- Swarm: group of particles  
- PSOOptimizer: improves solutions over iterations  

### Experiments
- Experiment: runs a configuration multiple times  
- ExperimentRunner: compares different PSO settings  

---

## How to Run

1. Install dependencies:
```bash
pip install -r requirements.txt

2. Run the project:
python main.py

Example Output
Timings: [10, 15, 12]
Total Waiting Time: 532

## Notes
- The simulation is simplified and does not represent real-world traffic perfectly.
- The focus of the project is on optimization rather than detailed traffic modeling.