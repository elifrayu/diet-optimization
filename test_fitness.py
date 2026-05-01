from chromosome import create_individual
from fitness import evaluate_solution

for user_id in [1, 2]:
    print(f"\n{'='*60}")
    print(f"USER {user_id}")
    print(f"{'='*60}")

    chromosome, breakfast_size = create_individual(user_id)

    # Test with default objectives
    result = evaluate_solution(chromosome, breakfast_size, user_id)

    print(f"\nDefault objectives (preference, cost, co2):")
    print(f"  Objectives: {result['objectives']}")
    print(f"  Preference: {result['preference']:.2f}")
    print(f"  Cost: {result['cost']:.2f}")
    print(f"  CO2: {result['co2']:.2f}")
    print(f"  Time: {result['time']:.2f}")
    print(f"  Diversity Count: {result['diversity_count']}")
    print(f"  Nutrient Penalty: {result['nutrient_penalty']:.4f}")
    print(f"  Total Penalty: {result['total_penalty']:.4f}")
    print(f"  Menu size: {len(result['menu'])}")
    print(f"  Totals: {result['totals']}")

    # Test with diversity_mode="penalty"
    result_penalty = evaluate_solution(
        chromosome, breakfast_size, user_id,
        diversity_mode="penalty",
        alpha_diversity=1.0
    )
    print(f"\nWith diversity_mode='penalty':")
    print(f"  Objectives: {result_penalty['objectives']}")
    print(f"  Total Penalty: {result_penalty['total_penalty']:.4f}")

    # Test with diversity_mode="hard"
    result_hard = evaluate_solution(
        chromosome, breakfast_size, user_id,
        diversity_mode="hard"
    )
    print(f"\nWith diversity_mode='hard':")
    print(f"  Objectives: {result_hard['objectives']}")
    print(f"  Total Penalty: {result_hard['total_penalty']:.4f}")

