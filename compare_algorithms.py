"""
NSGA-II vs SPEA2 Algorithm Comparison

Runs both algorithms with identical parameters and compares results.
"""

from nsga2 import nsga2
from spea2 import spea2
from metrics import (
    approximate_hypervolume,
    get_reference_point,
    calculate_constraint_compliance
)


def deduplicate_solutions(individuals):
    """Remove duplicate solutions (works for both NSGA-II and SPEA2)"""
    seen = {}
    unique = []

    for ind in individuals:
        menu_tuple = tuple(sorted(ind.fitness_result["menu"]))
        pref = round(ind.fitness_result["preference"], 4)
        cost = round(ind.fitness_result["cost"], 4)
        co2 = round(ind.fitness_result["co2"], 4)

        key = (menu_tuple, pref, cost, co2)

        if key not in seen:
            seen[key] = True
            unique.append(ind)

    return unique


def analyze_solutions(individuals, algorithm_name="Algorithm"):
    """Analyze front/archive solutions"""
    if not individuals:
        return None

    preferences = [ind.fitness_result['preference'] for ind in individuals]
    costs = [ind.fitness_result['cost'] for ind in individuals]
    co2s = [ind.fitness_result['co2'] for ind in individuals]
    nutrient_penalties = [ind.fitness_result['nutrient_penalty'] for ind in individuals]
    total_penalties = [ind.fitness_result['total_penalty'] for ind in individuals]
    diversities = [ind.fitness_result['diversity_count'] for ind in individuals]

    # Constraint compliance
    compliances = []
    for ind in individuals:
        compliance = calculate_constraint_compliance(
            ind.fitness_result['totals'],
            ind.fitness_result['dri']
        )
        compliances.append(compliance)

    avg_compliance = sum(compliances) / len(compliances) if compliances else 0

    return {
        "count": len(individuals),
        "objectives": [ind.objectives for ind in individuals],
        "avg_preference": sum(preferences) / len(preferences),
        "avg_cost": sum(costs) / len(costs),
        "avg_co2": sum(co2s) / len(co2s),
        "avg_nutrient_penalty": sum(nutrient_penalties) / len(nutrient_penalties),
        "avg_total_penalty": sum(total_penalties) / len(total_penalties),
        "avg_constraint_compliance": avg_compliance,
        "avg_diversity": sum(diversities) / len(diversities),
        "preference_range": (min(preferences), max(preferences)),
        "cost_range": (min(costs), max(costs)),
        "co2_range": (min(co2s), max(co2s)),
    }


print("\n" + "="*80)
print("NSGA-II vs SPEA2 - COMPREHENSIVE COMPARISON")
print("="*80)
print("\nConfiguration:")
print("  population_size: 50")
print("  generations: 50")
print("  objective_names: ('preference', 'cost', 'co2')")
print("  lambda_penalty: 10.0")
print("  diversity_mode: 'none'")
print("  seed: 42")

all_objectives_by_user = {}

for user_id in [1, 2]:
    print(f"\n{'='*80}")
    print(f"USER {user_id}")
    print(f"{'='*80}")

    # Run NSGA-II
    print("\n[Running NSGA-II...]")
    nsga2_result = nsga2(
        user_id=user_id,
        population_size=50,
        generations=50,
        objective_names=("preference", "cost", "co2"),
        lambda_penalty=10.0,
        diversity_mode="none",
        random_seed=42
    )

    # Run SPEA2
    print("[Running SPEA2...]")
    spea2_result = spea2(
        user_id=user_id,
        population_size=50,
        generations=50,
        objective_names=("preference", "cost", "co2"),
        lambda_penalty=10.0,
        diversity_mode="none",
        random_seed=42
    )

    # Deduplicate
    nsga2_unique = deduplicate_solutions(nsga2_result['best_front'])
    spea2_unique = deduplicate_solutions(spea2_result['best_individuals'])

    # Analyze
    nsga2_stats = analyze_solutions(nsga2_unique, "NSGA-II")
    spea2_stats = analyze_solutions(spea2_unique, "SPEA2")

    # Store for hypervolume calculation
    all_objectives_by_user[user_id] = {
        "nsga2": nsga2_stats["objectives"],
        "spea2": spea2_stats["objectives"]
    }

    # Print NSGA-II results
    print(f"\nNSGA-II Results:")
    print(f"  Total Final Solutions: {len(nsga2_result['best_front'])}")
    print(f"  Unique Solutions: {nsga2_stats['count']}")
    print(f"  Duplicates Removed: {len(nsga2_result['best_front']) - nsga2_stats['count']}")
    print(f"\n  Averages:")
    print(f"    Preference:             {nsga2_stats['avg_preference']:8.2f}")
    print(f"    Cost:                   {nsga2_stats['avg_cost']:8.2f}")
    print(f"    CO2:                    {nsga2_stats['avg_co2']:8.2f}")
    print(f"    Nutrient Penalty:       {nsga2_stats['avg_nutrient_penalty']:8.4f}")
    print(f"    Total Penalty:          {nsga2_stats['avg_total_penalty']:8.4f}")
    print(f"    Constraint Compliance:  {nsga2_stats['avg_constraint_compliance']:5.2f}/5")
    print(f"    Diversity:              {nsga2_stats['avg_diversity']:5.2f} groups")
    print(f"\n  Ranges:")
    print(f"    Preference: [{nsga2_stats['preference_range'][0]:7.2f}, {nsga2_stats['preference_range'][1]:7.2f}]")
    print(f"    Cost:       [{nsga2_stats['cost_range'][0]:7.2f}, {nsga2_stats['cost_range'][1]:7.2f}]")
    print(f"    CO2:        [{nsga2_stats['co2_range'][0]:7.2f}, {nsga2_stats['co2_range'][1]:7.2f}]")

    # Print SPEA2 results
    print(f"\nSPEA2 Results:")
    print(f"  Total Final Solutions: {len(spea2_result['best_individuals'])}")
    print(f"  Unique Solutions: {spea2_stats['count']}")
    print(f"  Duplicates Removed: {len(spea2_result['best_individuals']) - spea2_stats['count']}")
    print(f"\n  Averages:")
    print(f"    Preference:             {spea2_stats['avg_preference']:8.2f}")
    print(f"    Cost:                   {spea2_stats['avg_cost']:8.2f}")
    print(f"    CO2:                    {spea2_stats['avg_co2']:8.2f}")
    print(f"    Nutrient Penalty:       {spea2_stats['avg_nutrient_penalty']:8.4f}")
    print(f"    Total Penalty:          {spea2_stats['avg_total_penalty']:8.4f}")
    print(f"    Constraint Compliance:  {spea2_stats['avg_constraint_compliance']:5.2f}/5")
    print(f"    Diversity:              {spea2_stats['avg_diversity']:5.2f} groups")
    print(f"\n  Ranges:")
    print(f"    Preference: [{spea2_stats['preference_range'][0]:7.2f}, {spea2_stats['preference_range'][1]:7.2f}]")
    print(f"    Cost:       [{spea2_stats['cost_range'][0]:7.2f}, {spea2_stats['cost_range'][1]:7.2f}]")
    print(f"    CO2:        [{spea2_stats['co2_range'][0]:7.2f}, {spea2_stats['co2_range'][1]:7.2f}]")

# Hypervolume calculation
print(f"\n{'='*80}")
print("HYPERVOLUME ANALYSIS")
print(f"{'='*80}\n")

for user_id in [1, 2]:
    all_objectives = (
        all_objectives_by_user[user_id]["nsga2"] +
        all_objectives_by_user[user_id]["spea2"]
    )

    ref_point = get_reference_point(all_objectives, margin=0.10)

    nsga2_hv = approximate_hypervolume(
        all_objectives_by_user[user_id]["nsga2"],
        ref_point,
        samples=20000,
        seed=42
    )
    spea2_hv = approximate_hypervolume(
        all_objectives_by_user[user_id]["spea2"],
        ref_point,
        samples=20000,
        seed=42
    )

    print(f"User {user_id}:")
    print(f"  Reference Point: {tuple(f'{x:.2f}' for x in ref_point)}")
    print(f"  NSGA-II Hypervolume: {nsga2_hv:.2f}")
    print(f"  SPEA2 Hypervolume:   {spea2_hv:.2f}")
    print(f"  Winner: {'NSGA-II' if nsga2_hv > spea2_hv else 'SPEA2'} ({abs(nsga2_hv - spea2_hv):.2f} difference)")
    print()

print("="*80)
print("SUMMARY")
print("="*80)
print("""
Both NSGA-II and SPEA2 have been compared with:
- Identical parameters (pop=50, gen=50, lambda=10.0)
- Deduplication applied to both
- Hypervolume calculated using Monte Carlo approximation
- Constraint compliance measured (0-5 nutrients)
- Diversity metrics compared

Key metrics:
• Front sizes show algorithm-specific solution distribution
• Constraint compliance indicates feasibility of solutions
• Hypervolume measures quality and spread of approximation
• Diversity shows food group variety in solutions
""")


