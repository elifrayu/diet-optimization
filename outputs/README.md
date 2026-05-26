# Multi-Objective Diet Optimization using NSGA-II and SPEA2

This project implements a **multi-objective diet optimization system** using evolutionary algorithms.

The system generates daily meal plans while optimizing multiple objectives simultaneously:

- Maximizing user preference
- Minimizing meal cost
- Minimizing CO2 emission
- Satisfying nutritional constraints (DRI ranges)

Two multi-objective evolutionary algorithms are compared:

- NSGA-II
- SPEA2

---

# Project Structure

```text
diet-optimization/
│
├── compare_algorithms.py
├── visualizations.py
├── nsga2.py
├── spea2.py
├── chromosome.py
├── decoder.py
├── fitness.py
├── metrics.py
├── load_data.py
├── db.py
│
├── outputs/
│   ├── pareto_user1.png
│   ├── pareto_user2.png
│   ├── convergence_user1.png
│   ├── convergence_user2.png
│   ├── hv_bar.png
│   ├── user_comparison.png
│   ├── menu_nsga2_user1.png
│   ├── menu_spea2_user1.png
│   └── ...
│
└── tests/
    └── adim_adim_test.py
```

---

# Features

## Multi-objective optimization

The system optimizes:

1. Preference score (maximize)
2. Meal cost (minimize)
3. CO2 emission (minimize)

---

## Nutritional constraints

The generated menus are checked according to DRI ranges for:

- Energy
- Protein
- Carbohydrate
- Fiber
- Sodium

Penalty-based constraint handling is used.

---

## Diversity mechanism

Two diversity modes are implemented:

- `none`
- `penalty`

The diversity mechanism penalizes menus with low food-group variety.

---

## Algorithms

### NSGA-II

Implemented with:

- Fast non-dominated sorting
- Crowding distance
- Tournament selection
- Permutation-preserving crossover
- Mutation operator

---

### SPEA2

Implemented with:

- Strength assignment
- Raw fitness
- Density estimation (k-NN)
- External archive mechanism

---

# Database

The project uses a MySQL database containing:

- Food information
- Nutritional values
- User preferences
- DRI ranges

Required tables:

- foods
- food_group
- food_nutrients
- nutrients
- dri
- user
- user_foods

---

# Installation

## 1. Clone the repository

```bash
git clone <repository_url>
cd diet-optimization
```

---

## 2. Create virtual environment

```bash
python -m venv venv
```

Activate:

### macOS/Linux

```bash
source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Configure database

Update database credentials inside:

```python
db.py
```

Example:

```python
host="localhost"
user="root"
password="your_password"
database="diet"
```

---

# Running the Project

## Step-by-step test

```bash
python tests/adim_adim_test.py
```

This checks:

- Database connection
- Tables
- DRI loading
- Food pools
- Chromosome generation
- Decoder
- Fitness evaluation
- NSGA-II
- SPEA2

---

## Algorithm comparison

```bash
python compare_algorithms.py
```

Outputs:

- Average preference
- Cost
- CO2
- Constraint compliance
- Diversity
- Hypervolume comparison

---

## Generate visualizations

```bash
python visualizations.py
```

Generated figures are saved into:

```text
outputs/
```

---

# Visualizations

The project generates:

- Pareto front comparisons
- Convergence curves
- Hypervolume comparison charts
- Diversity impact analysis
- Example menu visualizations
- User comparison plots

---

# Experimental Setup

Parameters used in experiments:

```python
population_size = 50
generations = 50
lambda_penalty = 10.0
seed = 42
```

Objectives:

```python
("preference", "cost", "co2")
```

---

# Evaluation Metrics

## Hypervolume

Monte Carlo approximation is used to evaluate:

- Solution quality
- Pareto front spread

---

## Constraint compliance

Measures how many nutrients satisfy:

```text
RLL <= nutrient <= RUL
```

Maximum compliance score:

```text
5 / 5
```

---

# Example Outputs

The algorithms produce:

- Multiple Pareto-optimal menus
- Different trade-offs between preference, cost, and CO2
- Nutritionally feasible meal plans

---

# Technologies Used

- Python
- MySQL
- Matplotlib
- NumPy

---

# Authors

Developed as a university project for multi-objective optimization and evolutionary algorithms.

Algorithms implemented from scratch without external optimization libraries.