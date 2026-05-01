from spea2 import spea2
from metrics import calculate_constraint_compliance



def deduplicate_spea2_solutions(individuals):
    """Remove duplicate solutions from SPEA2 archive"""
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


for user_id in [1, 2]:
    print(f"\n{'='*80}")
    print(f"SPEA2 - USER {user_id}")
    print(f"{'='*80}")

    result = spea2(
        user_id=user_id,
        population_size=20,
        generations=10,
        random_seed=42
    )

    print(f"\nEvolution History (Archive Size & Best Fitness per Generation):")
    for h in result['history']:
        print(f"  Gen {h['generation']:2d}: Archive={h['archive_size']:2d}, Best Fitness={h['best_fitness']:.4f}")

    # Deduplicate archive
    unique_solutions = deduplicate_spea2_solutions(result['best_individuals'])

    print(f"\nArchive Analysis:")
    print(f"  Initial archive size: {len(result['best_individuals'])}")
    print(f"  Unique solutions: {len(unique_solutions)}")
    print(f"  Duplicates removed: {len(result['best_individuals']) - len(unique_solutions)}")

    if unique_solutions:
        print(f"\nNon-dominated Solutions (R(i) = 0):")
        for i, ind in enumerate(unique_solutions):
            totals = ind.fitness_result["totals"]
            dri = ind.fitness_result["dri"]

            compliance_count = calculate_constraint_compliance(totals, dri)

            print(f"\n  Solution {i+1}:")
            print(f"    Objective Vector: ({ind.objectives[0]:8.2f}, {ind.objectives[1]:8.2f}, {ind.objectives[2]:8.2f})")
            print(f"    Preference:       {ind.fitness_result['preference']:8.2f}")
            print(f"    Cost:             {ind.fitness_result['cost']:8.2f}")
            print(f"    CO2:              {ind.fitness_result['co2']:8.2f}")
            print(f"    Time:             {ind.fitness_result['time']:8.2f} min")
            print(f"    Diversity:        {ind.fitness_result['diversity_count']:2d} food groups")
            print(f"    Menu Size:        {len(ind.fitness_result['menu']):2d} items")
            print(f"    Nutrient Penalty: {ind.fitness_result['nutrient_penalty']:8.4f}")
            print(f"    Total Penalty:    {ind.fitness_result['total_penalty']:8.4f}")
            print(f"    SPEA2 Fitness:    {ind.spea2_fitness:8.4f}")
            print(f"    Nutrient Totals:")
            print(f"      Energy:        {totals['energy']:8.2f}")
            print(f"      Protein:       {totals['protein']:8.2f}")
            print(f"      Carbohydrate:  {totals['carbohydrate']:8.2f}")
            print(f"      Fiber:         {totals['fiber']:8.2f}")
            print(f"      Sodium:        {totals['sodium']:8.2f}")
            print(f"    Constraint Compliance: {compliance_count}/5")
