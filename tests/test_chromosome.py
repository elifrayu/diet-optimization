from chromosome import create_individual, crossover, mutate, check_individual

for user_id in [1, 2]:
    print(f"\n--- USER {user_id} ---")

    parent1, breakfast_size = create_individual(user_id)
    parent2, _ = create_individual(user_id)

    child1, child2 = crossover(parent1, parent2, breakfast_size)
    mutated = mutate(child1, breakfast_size)

    print("Parent kontrol:")
    check_individual(parent1, breakfast_size)

    print("Child kontrol:")
    check_individual(child1, breakfast_size)

    print("Mutated kontrol:")
    check_individual(mutated, breakfast_size)