from nsga2 import nsga2
from metrics import calculate_constraint_compliance



def deduplicate_evaluations(evaluations):
    """
    Remove duplicate evaluations from Pareto front.

    Key: (menu_tuple, rounded_preference, rounded_cost, rounded_co2)
    """
    seen = {}
    unique = []

    for eval_dict in evaluations:
        menu_tuple = tuple(sorted(eval_dict["decoded"]["menu"]))
        pref = round(eval_dict["raw"]["preference"], 4)
        cost = round(eval_dict["raw"]["cost"], 4)
        co2 = round(eval_dict["raw"]["co2"], 4)

        key = (menu_tuple, pref, cost, co2)

        if key not in seen:
            seen[key] = True
            unique.append(eval_dict)

    return unique


# Prepare evaluation data for better analysis
def prepare_evaluation_data(individuals):
    """Convert Individual objects to evaluation dictionaries for analysis"""
    evaluations = []

    for ind in individuals:
        eval_dict = {
            "objectives": ind.objectives,
            "raw": {
                "preference": ind.fitness_result["preference"],
                "cost": ind.fitness_result["cost"],
                "co2": ind.fitness_result["co2"],
                "time": ind.fitness_result["time"],
                "diversity": ind.fitness_result["diversity_count"],
            },
            "decoded": {
                "menu": ind.fitness_result["menu"],
                "totals": ind.fitness_result["totals"],
                "dri": ind.fitness_result["dri"],
                "nutrient_penalty": ind.fitness_result["nutrient_penalty"],
                "total_penalty": ind.fitness_result["total_penalty"],
            }
        }
        evaluations.append(eval_dict)

    return evaluations


for user_id in [1, 2]:
    print(f"\n{'='*80}")
    print(f"NSGA-II - USER {user_id}")
    print(f"{'='*80}")

    result = nsga2(
        user_id=user_id,
        population_size=20,
        generations=10,
        random_seed=42
    )

    print(f"\nEvolution History (First Front Size per Generation):")
    for h in result['history']:
        print(f"  Gen {h['generation']:2d}: {h['first_front_size']:2d} solutions in Pareto front")

    # Prepare evaluation data
    evals = prepare_evaluation_data(result['best_front'])
    unique_evals = deduplicate_evaluations(evals)

    print(f"\nPareto Front Analysis:")
    print(f"  Initial front size: {len(result['best_front'])}")
    print(f"  Unique solutions: {len(unique_evals)}")
    print(f"  Duplicates removed: {len(result['best_front']) - len(unique_evals)}")

    print(f"\nUnique Pareto Optimal Solutions:")
    for i, eval_dict in enumerate(unique_evals):
        objs = eval_dict["objectives"]
        raw = eval_dict["raw"]
        decoded = eval_dict["decoded"]
        totals = decoded["totals"]
        dri = decoded["dri"]

        compliance_count = calculate_constraint_compliance(totals, dri)

        print(f"\n  Solution {i+1}:")
        print(f"    Objective Vector: ({objs[0]:8.2f}, {objs[1]:8.2f}, {objs[2]:8.2f})")
        print(f"    Preference:       {raw['preference']:8.2f}")
        print(f"    Cost:             {raw['cost']:8.2f}")
        print(f"    CO2:              {raw['co2']:8.2f}")
        print(f"    Time:             {raw['time']:8.2f} min")
        print(f"    Diversity:        {raw['diversity']:2d} food groups")
        print(f"    Menu Size:        {len(decoded['menu']):2d} items")
        print(f"    Nutrient Penalty: {decoded['nutrient_penalty']:8.4f}")
        print(f"    Total Penalty:    {decoded['total_penalty']:8.4f}")
        print(f"    Nutrient Totals:")
        print(f"      Energy:        {totals['energy']:8.2f}")
        print(f"      Protein:       {totals['protein']:8.2f}")
        print(f"      Carbohydrate:  {totals['carbohydrate']:8.2f}")
        print(f"      Fiber:         {totals['fiber']:8.2f}")
        print(f"      Sodium:        {totals['sodium']:8.2f}")
        print(f"    Constraint Compliance: {compliance_count}/5")


