"""
Metrics for Multi-Objective Optimization

Implements hypervolume approximation using Monte Carlo sampling.
"""
#Hypervolume Monte Carlo ile hesaplar, referans noktası bulur (worst + %10), constraint compliance sayar (5 besin'den kaçı DRI içinde).
import random


def get_reference_point(all_objectives, margin=0.10):
    """
    Get reference point (worst observed objectives + margin).

    Args:
        all_objectives: List of objective tuples from both algorithms
        margin: Percentage margin to add (0.10 = 10%)

    Returns:
        Tuple: reference point for hypervolume calculation
    """
    if not all_objectives or len(all_objectives[0]) == 0:
        return None

    num_objectives = len(all_objectives[0])

    # Find worst value for each objective (remember: minimization)
    worst_values = []
    for obj_idx in range(num_objectives):
        obj_values = [objs[obj_idx] for objs in all_objectives]
        worst = max(obj_values)  # Worst for minimization is max
        worst_values.append(worst)

    # Add margin: reference must be strictly WORSE (higher) than all observed values.
    # Using an absolute margin avoids the pitfall of multiplying a negative worst value
    # by (1 + margin), which would move it toward zero (i.e., make it BETTER).
    # Formula: ref_i = worst_i + margin * max(|worst_i|, 1.0)
    reference_point = []
    for worst in worst_values:
        ref = worst + margin * max(abs(worst), 1.0)
        reference_point.append(ref)

    return tuple(reference_point)


def dominates(obj1, obj2):
    """
    Check if obj1 dominates obj2 (minimization).
    obj1 dominates obj2 if:
    - obj1 <= obj2 in all dimensions
    - obj1 < obj2 in at least one dimension
    """
    at_least_one_better = False

    for i in range(len(obj1)):
        if obj1[i] > obj2[i]:
            return False
        if obj1[i] < obj2[i]:
            at_least_one_better = True

    return at_least_one_better


def approximate_hypervolume(objectives, reference_point, samples=20000, seed=42):
    """
    Approximate hypervolume using Monte Carlo sampling.

    Args:
        objectives: List of objective vectors (minimization format)
        reference_point: Tuple of reference values (worst + margin)
        samples: Number of random samples to generate
        seed: Random seed for reproducibility

    Returns:
        float: Approximate hypervolume
    """
    if not objectives or not reference_point:
        return 0.0

    random.seed(seed)

    num_objectives = len(reference_point)

    # Find bounds of the objective space
    # Lower bounds: worst case 0 (ideally 0 for all minimize problems)
    # Upper bounds: reference point
    lower_bounds = [0.0] * num_objectives
    upper_bounds = list(reference_point)

    # Try to find better lower bounds from objectives
    for obj_idx in range(num_objectives):
        obj_values = [objs[obj_idx] for objs in objectives]
        best = min(obj_values)
        if best > 0:
            lower_bounds[obj_idx] = 0.0
        else:
            lower_bounds[obj_idx] = best * 0.9  # 10% margin below

    # Calculate box volume
    box_volume = 1.0
    for i in range(num_objectives):
        box_volume *= (upper_bounds[i] - lower_bounds[i])

    if box_volume <= 0:
        return 0.0

    # Monte Carlo sampling
    dominated_count = 0

    for _ in range(samples):
        # Generate random point in objective space
        sample_point = [
            random.uniform(lower_bounds[i], upper_bounds[i])
            for i in range(num_objectives)
        ]

        # Check if any solution dominates this sample point
        for obj in objectives:
            if dominates(obj, sample_point):
                dominated_count += 1
                break

    dominated_ratio = dominated_count / samples if samples > 0 else 0
    hypervolume = dominated_ratio * box_volume

    return hypervolume


def calculate_constraint_compliance(totals, dri):
    """
    Check constraint compliance for each nutrient.
    Returns: (compliant_count, total_nutrients)

    A nutrient is compliant if: RLL <= total <= RUL
    """
    nutrients = ["energy", "protein", "carbohydrate", "fiber", "sodium"]
    compliant = 0

    for nutrient in nutrients:
        if nutrient not in dri or nutrient not in totals:
            continue

        rll = dri[nutrient]["RLL"]
        rul = dri[nutrient]["RUL"]
        total = totals[nutrient]

        if rll <= total <= rul:
            compliant += 1

    return compliant

