import random
from load_data import get_pool

breakfast_pool, main_meal_pool = get_pool()

BREAKFAST_SIZE = len(breakfast_pool)


def create_individual():
    b_part = breakfast_pool.copy()
    m_part = main_meal_pool.copy()

    random.shuffle(b_part)
    random.shuffle(m_part)

    return b_part + m_part


def ox_crossover_part(part1, part2):
    size = len(part1)

    start, end = sorted(random.sample(range(size), 2))

    child = [None] * size

    # Parent1'den bir parçayı aynen al
    child[start:end] = part1[start:end]

    # Parent2'den eksik kalanları sırayla doldur
    p2_items = [item for item in part2 if item not in child]

    index = 0
    for i in range(size):
        if child[i] is None:
            child[i] = p2_items[index]
            index += 1

    return child


def crossover(parent1, parent2):
    p1_b = parent1[:BREAKFAST_SIZE]
    p1_m = parent1[BREAKFAST_SIZE:]

    p2_b = parent2[:BREAKFAST_SIZE]
    p2_m = parent2[BREAKFAST_SIZE:]

    child1_b = ox_crossover_part(p1_b, p2_b)
    child2_b = ox_crossover_part(p2_b, p1_b)

    child1_m = ox_crossover_part(p1_m, p2_m)
    child2_m = ox_crossover_part(p2_m, p1_m)

    child1 = child1_b + child1_m
    child2 = child2_b + child2_m

    return child1, child2


def mutate(chromosome, mutation_rate=0.01):
    mutated = chromosome.copy()

    # Kahvaltı kısmında swap mutation
    if random.random() < mutation_rate:
        i, j = random.sample(range(0, BREAKFAST_SIZE), 2)
        mutated[i], mutated[j] = mutated[j], mutated[i]

    # Ana yemek kısmında swap mutation
    if random.random() < mutation_rate:
        i, j = random.sample(range(BREAKFAST_SIZE, len(mutated)), 2)
        mutated[i], mutated[j] = mutated[j], mutated[i]

    return mutated


def check_individual(chromosome):
    breakfast_part = chromosome[:BREAKFAST_SIZE]
    main_part = chromosome[BREAKFAST_SIZE:]

    print("Toplam gen:", len(chromosome))
    print("Kahvaltı gen sayısı:", len(breakfast_part))
    print("Ana yemek gen sayısı:", len(main_part))
    print("Tekrar eden gen var mı?", len(chromosome) != len(set(chromosome)))


# TEST
if __name__ == "__main__":
    print("Kahvaltı havuzu:", len(breakfast_pool))
    print("Ana yemek havuzu:", len(main_meal_pool))

    parent1 = create_individual()
    parent2 = create_individual()

    child1, child2 = crossover(parent1, parent2)
    mutated_child = mutate(child1, mutation_rate=0.5)

    print("\nParent 1 kontrol:")
    check_individual(parent1)

    print("\nChild 1 kontrol:")
    check_individual(child1)

    print("\nMutated child kontrol:")
    check_individual(mutated_child)