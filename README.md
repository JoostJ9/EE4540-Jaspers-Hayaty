# EE4540 – Consensus Algorithms Project  
*By Jaspers & Hayaty*

## Overview

This repository contains our implementation for the EE4540 project, focused on distributed consensus algorithms. The project is divided into three main assignments, each exploring a different aspect of consensus in multi-agent systems.

## Project Structure

### 📍 Assignment 1: Sensor Field Generation  
This script generates a spatial field with randomly placed sensors.  
- Can be easily re-run to regenerate the field.  
- Outputs form the basis for all subsequent simulations.

### 🔄 Assignment 2: Randomized Gossip vs. PDMM (Average Consensus)  
This notebook compares the convergence behavior of the **Randomized Gossip** algorithm and the **PDMM Average Consensus** algorithm.  
- Convergence is visualized via animated `.gif` files stored in the `animations/` folder.  
- Note: GIF generation may take significant time.  
- Final report plots are generated at the end of the file.

### 📊 Assignment 3: PDMM Median Consensus  
This notebook contains an implementation of the **PDMM Median Consensus** algorithm, suitable for robust consensus under outlier presence.

## Codebase Notes

- All core algorithms are implemented in separate Python files named accordingly (e.g., `pdmm_average.py`, `randomized_gossip.py`, etc.).
- Utility functions (e.g., plotting, initialization, data saving) are defined in `helper.py` to improve notebook readability and modularity.

## Run Instructions

Each notebook can be run independently. Ensure all dependencies are installed and that the working directory includes the required `.py` files.
