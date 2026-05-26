"""
TASK 3: Penalty Effect Analysis for NSGA-II

Analyzes how different lambda_penalty values affect:
- Pareto front size
- Menu diversity
- Constraint compliance
- Solution characteristics
"""

from nsga2 import nsga2
from metrics import calculate_constraint_compliance



def deduplicate_front(individuals):
    """Remove duplicate solutions from front"""
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


def analyze_penalty_effect(user_id, lambda_penalties):
    """
    Run NSGA-II with different penalty values and analyze results.
    """

    print(f"\n{'='*80}")
    print(f"PENALTY EFFECT ANALYSIS - USER {user_id}")
    print(f"{'='*80}")

    print(f"\nConfiguration:")
    print(f"  population_size: 20")
    print(f"  generations: 10")
    print(f"  objective_names: ('preference', 'cost', 'co2')")
    print(f"  diversity_mode: 'none'")
    print(f"  lambda_penalty values: {lambda_penalties}")

    results_summary = []

    for lambda_val in lambda_penalties:
        print(f"\n{'-'*80}")
        print(f"Running NSGA-II with lambda_penalty={lambda_val}...")

        result = nsga2(
            user_id=user_id,
            population_size=20,
            generations=10,
            objective_names=("preference", "cost", "co2"),
            diversity_mode="none",
            lambda_penalty=lambda_val,
            random_seed=42
        )

        # Deduplicate front
        unique_front = deduplicate_front(result['best_front'])

        # Calculate statistics
        menu_sizes = [len(ind.fitness_result['menu']) for ind in unique_front]
        nutrient_penalties = [ind.fitness_result['nutrient_penalty'] for ind in unique_front]
        total_penalties = [ind.fitness_result['total_penalty'] for ind in unique_front]
        preferences = [ind.fitness_result['preference'] for ind in unique_front]
        costs = [ind.fitness_result['cost'] for ind in unique_front]
        co2s = [ind.fitness_result['co2'] for ind in unique_front]
        diversities = [ind.fitness_result['diversity_count'] for ind in unique_front]

        # Calculate constraint compliance
        compliances = []
        for ind in unique_front:
            compliance = calculate_constraint_compliance(
                ind.fitness_result['totals'],
                ind.fitness_result['dri']
            )
            compliances.append(compliance)

        avg_compliance = sum(compliances) / len(compliances) if compliances else 0

        # Summary statistics
        summary = {
            "lambda_penalty": lambda_val,
            "unique_solutions": len(unique_front),
            "total_solutions": len(result['best_front']),
            "duplicates": len(result['best_front']) - len(unique_front),
            "avg_menu_size": sum(menu_sizes) / len(menu_sizes) if menu_sizes else 0,
            "nutrient_penalty_avg": sum(nutrient_penalties) / len(nutrient_penalties) if nutrient_penalties else 0,
            "total_penalty_avg": sum(total_penalties) / len(total_penalties) if total_penalties else 0,
            "avg_constraint_compliance": avg_compliance,
            "avg_preference": sum(preferences) / len(preferences) if preferences else 0,
            "avg_cost": sum(costs) / len(costs) if costs else 0,
            "avg_co2": sum(co2s) / len(co2s) if co2s else 0,
            "avg_diversity": sum(diversities) / len(diversities) if diversities else 0,
        }

        results_summary.append(summary)

        # Print statistics
        print(f"\nResults for lambda_penalty={lambda_val}:")
        print(f"  Unique Solutions:          {summary['unique_solutions']:2d}")
        print(f"  Total Solutions (before dedup): {summary['total_solutions']:2d}")
        print(f"  Duplicates Removed:        {summary['duplicates']:2d}")
        print(f"  Avg Menu Size:             {summary['avg_menu_size']:6.2f} items")
        print(f"  Avg Nutrient Penalty:      {summary['nutrient_penalty_avg']:8.4f}")
        print(f"  Avg Total Penalty:         {summary['total_penalty_avg']:8.4f}")
        print(f"  Avg Constraint Compliance: {summary['avg_constraint_compliance']:4.1f}/5")
        print(f"  Avg Preference:            {summary['avg_preference']:8.2f}")
        print(f"  Avg Cost:                  {summary['avg_cost']:8.2f}")
        print(f"  Avg CO2:                   {summary['avg_co2']:8.2f}")
        print(f"  Avg Diversity:             {summary['avg_diversity']:6.2f} groups")

    # Comparative analysis
    print(f"\n{'='*80}")
    print(f"COMPARATIVE ANALYSIS")
    print(f"{'='*80}\n")

    print(f"{'Lambda':<10} {'Unique':<8} {'Menu':<7} {'Nutr.P':<10} {'Total.P':<10} {'Comply':<8} {'Pref':<8} {'Cost':<8} {'CO2':<8} {'Div':<7}")
    print(f"{'Penalty':<10} {'Sol':<8} {'Size':<7} {'Penalty':<10} {'Penalty':<10} {'(avg)':<8} {'(avg)':<8} {'(avg)':<8} {'(avg)':<8} {'(avg)':<7}")
    print(f"{'-'*92}")

    for summary in results_summary:
        print(f"{summary['lambda_penalty']:<10.1f} "
              f"{summary['unique_solutions']:<8d} "
              f"{summary['avg_menu_size']:<7.2f} "
              f"{summary['nutrient_penalty_avg']:<10.4f} "
              f"{summary['total_penalty_avg']:<10.4f} "
              f"{summary['avg_constraint_compliance']:<8.2f} "
              f"{summary['avg_preference']:<8.2f} "
              f"{summary['avg_cost']:<8.2f} "
              f"{summary['avg_co2']:<8.2f} "
              f"{summary['avg_diversity']:<7.2f}")

    # Key observations
    print(f"\nKey Observations:")

    if results_summary:
        min_solutions = min(s['unique_solutions'] for s in results_summary)
        max_solutions = max(s['unique_solutions'] for s in results_summary)

        max_compliance_idx = max(range(len(results_summary)),
                                 key=lambda i: results_summary[i]['avg_constraint_compliance'])
        min_compliance_idx = min(range(len(results_summary)),
                                 key=lambda i: results_summary[i]['avg_constraint_compliance'])

        print(f"  • Solution count range: {min_solutions} - {max_solutions} unique solutions")
        print(f"  • Best constraint compliance: λ={results_summary[max_compliance_idx]['lambda_penalty']:.1f} "
              f"({results_summary[max_compliance_idx]['avg_constraint_compliance']:.2f}/5)")
        print(f"  • Worst constraint compliance: λ={results_summary[min_compliance_idx]['lambda_penalty']:.1f} "
              f"({results_summary[min_compliance_idx]['avg_constraint_compliance']:.2f}/5)")

        # Preference vs penalty trade-off
        print(f"\n  Trade-offs:")
        for summary in results_summary:
            print(f"    λ={summary['lambda_penalty']:.1f}: "
                  f"Pref={summary['avg_preference']:.2f}, "
                  f"Penalty={summary['total_penalty_avg']:.4f}, "
                  f"Diversity={summary['avg_diversity']:.2f} groups")


if __name__ == "__main__":
    # User 2 with different penalty values
    lambda_penalties = [1.0, 5.0, 10.0]
    analyze_penalty_effect(user_id=2, lambda_penalties=lambda_penalties)

