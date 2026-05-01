import random
import math
from chromosome import create_individual, crossover, mutate
from fitness import evaluate_solution


class Individual:
    """NSGA-II için bir birey (chromosom + fitness bilgisi)"""

    def __init__(self, chromosome, breakfast_size, user_id,
                 objective_names=("preference", "cost", "co2"),
                 diversity_mode="none", alpha_diversity=1.0, lambda_penalty=10.0):
        self.chromosome = chromosome
        self.breakfast_size = breakfast_size
        self.user_id = user_id
        self.objective_names = objective_names
        self.diversity_mode = diversity_mode
        self.alpha_diversity = alpha_diversity
        self.lambda_penalty = lambda_penalty

        # Fitness bilgisi
        self.fitness_result = None
        self.objectives = None
        self.rank = None
        self.crowding_distance = 0.0

    def evaluate(self):
        """Bireyin fitness'ını hesapla"""
        self.fitness_result = evaluate_solution(
            self.chromosome,
            self.breakfast_size,
            self.user_id,
            objective_names=self.objective_names,
            diversity_mode=self.diversity_mode,
            alpha_diversity=self.alpha_diversity,
            lambda_penalty=self.lambda_penalty
        )
        self.objectives = self.fitness_result["objectives"]

    def __repr__(self):
        if self.objectives:
            return f"Ind(rank={self.rank}, objs={tuple(f'{x:.2f}' for x in self.objectives)}, cd={self.crowding_distance:.2f})"
        return "Ind(unevaluated)"


def non_dominated_sort(population):
    """
    Non-dominated sorting (Pareto ranking)
    Her bireye rank atanır (0 = Pareto optimal, 1, 2, ...)
    """
    n = len(population)

    # Initialization
    for p in population:
        p.rank = None

    fronts = []

    # Dominance counter ve dominated list
    domination_count = [0] * n
    dominated_solutions = [[] for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            if dominates(population[i], population[j]):
                dominated_solutions[i].append(j)
                domination_count[j] += 1
            elif dominates(population[j], population[i]):
                dominated_solutions[j].append(i)
                domination_count[i] += 1

    # First front: domination count = 0
    current_front = []
    for i in range(n):
        if domination_count[i] == 0:
            population[i].rank = 0
            current_front.append(i)

    fronts.append(current_front)

    # İteratif olarak diğer frontları bul
    rank = 1
    while len(current_front) > 0:
        next_front = []
        for i in current_front:
            for j in dominated_solutions[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    population[j].rank = rank
                    next_front.append(j)
        current_front = next_front
        if len(current_front) > 0:
            fronts.append(current_front)
            rank += 1

    return fronts


def dominates(ind1, ind2):
    """
    ind1 ind2'yi dominate ediyor mu?
    (Tüm objectives'te >= ve en az birinde > olmalı)

    Minimization problemi olduğundan:
    ind1 uchun lower objectives better.
    """
    at_least_one_better = False

    for i in range(len(ind1.objectives)):
        if ind1.objectives[i] > ind2.objectives[i]:  # ind1 worse
            return False
        if ind1.objectives[i] < ind2.objectives[i]:  # ind1 better
            at_least_one_better = True

    return at_least_one_better


def crowding_distance_assignment(population, fronts):
    """
    Crowding distance hesapla (çeşitlilik ölçümü)
    Her objectives dimension'ında bireyleri sırala ve sınırlara
    daha yakın olanları daha yüksek distance ver.
    """
    nobj = len(population[0].objectives) if population else 0

    for ind in population:
        ind.crowding_distance = 0.0

    for front in fronts:
        if len(front) <= 2:
            for i in front:
                population[i].crowding_distance = float('inf')
            continue

        # Her objective dimension'ı için
        for m in range(nobj):
            # Front'u m'inci objective'e göre sıra
            front_sorted = sorted(front, key=lambda i: population[i].objectives[m])

            # Uç noktalar (min ve max) infinite distance al
            population[front_sorted[0]].crowding_distance = float('inf')
            population[front_sorted[-1]].crowding_distance = float('inf')

            # Normalize distance
            obj_min = population[front_sorted[0]].objectives[m]
            obj_max = population[front_sorted[-1]].objectives[m]
            obj_range = obj_max - obj_min

            if obj_range == 0:
                continue

            for i in range(1, len(front_sorted) - 1):
                idx = front_sorted[i]
                distance = (population[front_sorted[i + 1]].objectives[m] -
                           population[front_sorted[i - 1]].objectives[m]) / obj_range
                population[idx].crowding_distance += distance


def tournament_selection(population, tournament_size=2):
    """
    Tournament selection: rank ve crowding distance'a göre
    """
    selected = random.sample(population, tournament_size)

    # Rank'e göre sırala (lower rank better)
    selected_by_rank = sorted(selected, key=lambda ind: ind.rank)

    if selected_by_rank[0].rank < selected_by_rank[1].rank:
        return selected_by_rank[0]
    elif selected_by_rank[0].rank > selected_by_rank[1].rank:
        return selected_by_rank[1]
    else:
        # Aynı rank'te: crowding distance'a göre seç
        if selected_by_rank[0].crowding_distance > selected_by_rank[1].crowding_distance:
            return selected_by_rank[0]
        else:
            return selected_by_rank[1]


def create_offspring(parent1, parent2, crossover_rate=0.9):
    """
    İki parentten bir offspring oluştur (crossover + mutation)
    """
    # Crossover (probability-based)
    if random.random() < crossover_rate:
        child_chromosome, _ = crossover(
            parent1.chromosome,
            parent2.chromosome,
            parent1.breakfast_size
        )
    else:
        child_chromosome = parent1.chromosome.copy()

    # Mutation (always applied)
    child_chromosome = mutate(child_chromosome, parent1.breakfast_size)

    # Yeni Individual oluştur
    child = Individual(
        child_chromosome,
        parent1.breakfast_size,
        parent1.user_id,
        objective_names=parent1.objective_names,
        diversity_mode=parent1.diversity_mode,
        alpha_diversity=parent1.alpha_diversity,
        lambda_penalty=parent1.lambda_penalty
    )

    return child


def nsga2(user_id, population_size=20, generations=10,
         objective_names=("preference", "cost", "co2"),
         diversity_mode="none", alpha_diversity=1.0, lambda_penalty=10.0,
         crossover_rate=0.9, random_seed=None):
    """
    NSGA-II algoritması

    Args:
        user_id: Kullanıcı ID (1 veya 2)
        population_size: Popülasyon boyutu (çift sayı)
        generations: Nesil sayısı
        objective_names: Optimize edilecek objectives
        diversity_mode: "none", "penalty", or "hard"
        alpha_diversity: Diversity penalty katsayısı
        lambda_penalty: Nutrient penalty ağırlığı
        random_seed: Tekrarlanabilirlik için

    Returns:
        Dict with:
        - population: Final population
        - best_front: Pareto optimal front (rank 0)
        - history: Her nesilde en iyi fitness'lar
    """

    if random_seed is not None:
        random.seed(random_seed)

    # 1. Initialize population
    population = []
    for _ in range(population_size):
        chromosome, breakfast_size = create_individual(user_id)
        ind = Individual(
            chromosome, breakfast_size, user_id,
            objective_names=objective_names,
            diversity_mode=diversity_mode,
            alpha_diversity=alpha_diversity,
            lambda_penalty=lambda_penalty
        )
        ind.evaluate()
        population.append(ind)

    history = []

    # 2. Ana loop
    for gen in range(generations):
        # Non-dominated sorting
        fronts = non_dominated_sort(population)

        # Crowding distance
        crowding_distance_assignment(population, fronts)

        # Tüm popülasyonu rank ve CD'ye göre sırala
        population.sort(key=lambda ind: (ind.rank, -ind.crowding_distance))

        # Iteration bilgisi kaydet
        first_front = [ind for ind in population if ind.rank == 0]
        if first_front:
            best_objs = [ind.objectives for ind in first_front]
            history.append({
                "generation": gen,
                "first_front_size": len(first_front),
                "first_front_objectives": best_objs
            })

        # Parent selection ve reproduction (offspring oluştur)
        offspring = []
        while len(offspring) < population_size:
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            child = create_offspring(parent1, parent2, crossover_rate=crossover_rate)
            child.evaluate()
            offspring.append(child)

        # Environmental selection (parent + offspring = 2N, seç N)
        combined = population + offspring
        fronts = non_dominated_sort(combined)
        crowding_distance_assignment(combined, fronts)

        # Rank ve CD'ye göre sırala
        combined.sort(key=lambda ind: (ind.rank, -ind.crowding_distance))

        # Yeni popülasyonun ilk population_size'ını al
        population = combined[:population_size]

    # Final ranking
    non_dominated_sort(population)
    best_front = [ind for ind in population if ind.rank == 0]

    return {
        "population": population,
        "best_front": best_front,
        "history": history
    }

