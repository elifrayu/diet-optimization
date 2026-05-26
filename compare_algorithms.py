"""
NSGA-II vs SPEA2 Algorithm Comparison

Runs both algorithms with identical parameters and compares results.
Also compares diversity_mode="none" and diversity_mode="penalty".
"""

import csv

from nsga2 import nsga2
from spea2 import spea2
from metrics import (
    approximate_hypervolume,
    get_reference_point,
    calculate_constraint_compliance
)


def deduplicate_solutions(individuals):
    """Remove duplicate solutions."""
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


def analyze_solutions(individuals):
    """Analyze front/archive solutions."""
    if not individuals:
        return None

    preferences = [ind.fitness_result["preference"] for ind in individuals]
    costs = [ind.fitness_result["cost"] for ind in individuals]
    co2s = [ind.fitness_result["co2"] for ind in individuals]
    nutrient_penalties = [ind.fitness_result["nutrient_penalty"] for ind in individuals]
    total_penalties = [ind.fitness_result["total_penalty"] for ind in individuals]
    diversities = [ind.fitness_result["diversity_count"] for ind in individuals]

    compliances = []
    for ind in individuals:
        compliance = calculate_constraint_compliance(
            ind.fitness_result["totals"],
            ind.fitness_result["dri"]
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


def print_stats(title, result_count, unique_stats, original_count):
    """Print algorithm statistics."""

    print(f"\n{title} Results:")
    print(f"  Total Final Solutions: {result_count}")
    print(f"  Unique Solutions: {unique_stats['count']}")
    print(f"  Duplicates Removed: {original_count - unique_stats['count']}")

    print(f"\n  Averages:")
    print(f"    Preference:             {unique_stats['avg_preference']:8.2f}")
    print(f"    Cost:                   {unique_stats['avg_cost']:8.2f}")
    print(f"    CO2:                    {unique_stats['avg_co2']:8.2f}")
    print(f"    Nutrient Penalty:       {unique_stats['avg_nutrient_penalty']:8.4f}")
    print(f"    Total Penalty:          {unique_stats['avg_total_penalty']:8.4f}")
    print(f"    Constraint Compliance:  {unique_stats['avg_constraint_compliance']:5.2f}/5")
    print(f"    Diversity:              {unique_stats['avg_diversity']:5.2f} groups")

    print(f"\n  Ranges:")
    print(f"    Preference: [{unique_stats['preference_range'][0]:7.2f}, {unique_stats['preference_range'][1]:7.2f}]")
    print(f"    Cost:       [{unique_stats['cost_range'][0]:7.2f}, {unique_stats['cost_range'][1]:7.2f}]")
    print(f"    CO2:        [{unique_stats['co2_range'][0]:7.2f}, {unique_stats['co2_range'][1]:7.2f}]")


def run_experiment():

    print("\n" + "=" * 80)
    print("NSGA-II vs SPEA2 - COMPREHENSIVE COMPARISON")
    print("=" * 80)

    print("\nConfiguration:")
    print("  population_size: 50")
    print("  generations: 50")
    print("  objective_names: ('preference', 'cost', 'co2')")
    print("  lambda_penalty: 10.0")
    print("  diversity_modes: ['none', 'penalty']")
    print("  seed: 42")

    all_objectives_by_case = {}
    summary_rows = []

    for diversity_mode in ["none", "penalty"]:

        print(f"\n{'#' * 80}")
        print(f"DIVERSITY MODE: {diversity_mode}")
        print(f"{'#' * 80}")

        for user_id in [1, 2]:

            print(f"\n{'=' * 80}")
            print(f"USER {user_id}")
            print(f"{'=' * 80}")

            # NSGA-II
            print("\n[Running NSGA-II...]")

            nsga2_result = nsga2(
                user_id=user_id,
                population_size=50,
                generations=50,
                objective_names=("preference", "cost", "co2"),
                lambda_penalty=10.0,
                diversity_mode=diversity_mode,
                random_seed=42
            )

            # SPEA2
            print("[Running SPEA2...]")

            spea2_result = spea2(
                user_id=user_id,
                population_size=50,
                generations=50,
                objective_names=("preference", "cost", "co2"),
                lambda_penalty=10.0,
                diversity_mode=diversity_mode,
                random_seed=42
            )

            # Deduplicate
            nsga2_unique = deduplicate_solutions(nsga2_result["best_front"])
            spea2_unique = deduplicate_solutions(spea2_result["best_individuals"])

            # Analyze
            nsga2_stats = analyze_solutions(nsga2_unique)
            spea2_stats = analyze_solutions(spea2_unique)

            # Store objectives
            all_objectives_by_case[(diversity_mode, user_id)] = {
                "nsga2": nsga2_stats["objectives"],
                "spea2": spea2_stats["objectives"]
            }

            # Print results
            print_stats(
                "NSGA-II",
                len(nsga2_result["best_front"]),
                nsga2_stats,
                len(nsga2_result["best_front"])
            )

            print_stats(
                "SPEA2",
                len(spea2_result["best_individuals"]),
                spea2_stats,
                len(spea2_result["best_individuals"])
            )

            # Save summary rows
            summary_rows.append({
                "diversity_mode": diversity_mode,
                "user_id": user_id,
                "algorithm": "NSGA-II",
                "unique_solutions": nsga2_stats["count"],
                "avg_preference": nsga2_stats["avg_preference"],
                "avg_cost": nsga2_stats["avg_cost"],
                "avg_co2": nsga2_stats["avg_co2"],
                "avg_nutrient_penalty": nsga2_stats["avg_nutrient_penalty"],
                "avg_total_penalty": nsga2_stats["avg_total_penalty"],
                "avg_constraint_compliance": nsga2_stats["avg_constraint_compliance"],
                "avg_diversity": nsga2_stats["avg_diversity"],
                "hypervolume": ""
            })

            summary_rows.append({
                "diversity_mode": diversity_mode,
                "user_id": user_id,
                "algorithm": "SPEA2",
                "unique_solutions": spea2_stats["count"],
                "avg_preference": spea2_stats["avg_preference"],
                "avg_cost": spea2_stats["avg_cost"],
                "avg_co2": spea2_stats["avg_co2"],
                "avg_nutrient_penalty": spea2_stats["avg_nutrient_penalty"],
                "avg_total_penalty": spea2_stats["avg_total_penalty"],
                "avg_constraint_compliance": spea2_stats["avg_constraint_compliance"],
                "avg_diversity": spea2_stats["avg_diversity"],
                "hypervolume": ""
            })

    # Hypervolume
    print(f"\n{'=' * 80}")
    print("HYPERVOLUME ANALYSIS")
    print(f"{'=' * 80}\n")

    for diversity_mode in ["none", "penalty"]:

        print(f"\nDiversity mode: {diversity_mode}")

        for user_id in [1, 2]:

            all_objectives = (
                all_objectives_by_case[(diversity_mode, user_id)]["nsga2"] +
                all_objectives_by_case[(diversity_mode, user_id)]["spea2"]
            )

            ref_point = get_reference_point(
                all_objectives,
                margin=0.10
            )

            nsga2_hv = approximate_hypervolume(
                all_objectives_by_case[(diversity_mode, user_id)]["nsga2"],
                ref_point,
                samples=20000,
                seed=42
            )

            spea2_hv = approximate_hypervolume(
                all_objectives_by_case[(diversity_mode, user_id)]["spea2"],
                ref_point,
                samples=20000,
                seed=42
            )

            winner = "NSGA-II" if nsga2_hv > spea2_hv else "SPEA2"

            print(f"User {user_id}:")
            print(f"  Reference Point: {tuple(f'{x:.2f}' for x in ref_point)}")
            print(f"  NSGA-II Hypervolume: {nsga2_hv:.2f}")
            print(f"  SPEA2 Hypervolume:   {spea2_hv:.2f}")
            print(f"  Winner: {winner} ({abs(nsga2_hv - spea2_hv):.2f} difference)")
            print()

            # Save hypervolume to rows
            for row in summary_rows:
                if (
                    row["diversity_mode"] == diversity_mode and
                    row["user_id"] == user_id
                ):
                    if row["algorithm"] == "NSGA-II":
                        row["hypervolume"] = nsga2_hv

                    elif row["algorithm"] == "SPEA2":
                        row["hypervolume"] = spea2_hv

    # Save CSV
    with open("results_summary.csv", "w", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=summary_rows[0].keys()
        )

        writer.writeheader()
        writer.writerows(summary_rows)

    print("\nResults saved to results_summary.csv")

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print("""
Both NSGA-II and SPEA2 have been compared with:
- Identical parameters (pop=50, gen=50, lambda=10.0)
- Two diversity modes: none and penalty
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


if __name__ == "__main__":
    run_experiment()