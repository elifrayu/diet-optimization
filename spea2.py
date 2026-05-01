import random
import math
from chromosome import create_individual, crossover, mutate
from fitness import evaluate_solution


class Individual:
    """SPEA2 için bir birey (chromosom + fitness bilgisi)"""
    
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
        
        # SPEA2 specific
        self.strength = 0.0      # Kaç bireyi dominate ettiği
        self.raw_fitness = 0.0   # Dominant olan bireylerin strength toplamı
        self.density = 0.0       # k-NN based density
        self.spea2_fitness = 0.0 # final fitness = raw_fitness + 1 / (density + 2)
    
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
            return f"Ind(objs={tuple(f'{x:.2f}' for x in self.objectives)}, fitness={self.spea2_fitness:.2f})"
        return "Ind(unevaluated)"


def dominates(ind1, ind2):
    """
    ind1 ind2'yi dominate ediyor mu?
    (Tüm objectives'te >= ve en az birinde > olmalı)
    Minimization problemi olduğundan lower objectives better.
    """
    at_least_one_better = False
    
    for i in range(len(ind1.objectives)):
        if ind1.objectives[i] > ind2.objectives[i]:  # ind1 worse
            return False
        if ind1.objectives[i] < ind2.objectives[i]:  # ind1 better
            at_least_one_better = True
    
    return at_least_one_better


def assignment_strength(population):
    """
    S(i) = |{j : j ∈ population and i dominate j}|
    Her bireyin kaç bireyi dominate ettiğini hesapla
    """
    for ind in population:
        ind.strength = 0.0
    
    n = len(population)
    for i in range(n):
        for j in range(n):
            if i != j and dominates(population[i], population[j]):
                population[i].strength += 1.0


def assignment_raw_fitness(population):
    """
    R(i) = sum{j : j dominate i and j ∈ population} S(j)
    Her bireyin raw fitness'ını hesapla (dominant olanların strength'inin toplamı)
    """
    for ind in population:
        ind.raw_fitness = 0.0
    
    n = len(population)
    for i in range(n):
        for j in range(n):
            if i != j and dominates(population[j], population[i]):
                population[i].raw_fitness += population[j].strength


def euclidean_distance(obj1, obj2):
    """İki objective vector arasında Euclidean distance"""
    return math.sqrt(sum((o1 - o2) ** 2 for o1, o2 in zip(obj1, obj2)))


def assignment_density(population, k=5):
    """
    D(i) = 1 / (σ_k(i) + 2)
    σ_k(i): i'nin k-inci en yakın komşusunun distance'ı
    
    k = min(sqrt(|pop|), 5) default
    """
    if k is None:
        k = max(int(math.sqrt(len(population))), 1)
        k = min(k, 5)
    
    if len(population) <= 1:
        for ind in population:
            ind.density = 1.0
        return
    
    for i, ind in enumerate(population):
        distances = []
        for j, other in enumerate(population):
            if i != j:
                dist = euclidean_distance(ind.objectives, other.objectives)
                distances.append(dist)
        
        if len(distances) == 0:
            ind.density = 1.0
        else:
            distances.sort()
            # k-inci en yakın komşu
            k_neighbor_distance = distances[min(k - 1, len(distances) - 1)]
            ind.density = 1.0 / (k_neighbor_distance + 2.0)


def assignment_fitness(population, k=5):
    """
    SPEA2 fitness değerlendirmesi:
    F(i) = R(i) + D(i)
    """
    assignment_strength(population)
    assignment_raw_fitness(population)
    assignment_density(population, k)
    
    for ind in population:
        ind.spea2_fitness = ind.raw_fitness + ind.density


def tournament_selection(population, tournament_size=2):
    """
    Tournament selection: SPEA2 fitness'a göre
    """
    selected = random.sample(population, tournament_size)
    best = min(selected, key=lambda ind: ind.spea2_fitness)
    return best


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


def environmental_selection(combined, population_size):
    """
    SPEA2 Environmental Selection:
    1. R(i) = 0 (non-dominated) olan bireyleri seç
    2. Yeterli değilse, R(i) > 0 olanları fitness'e göre sırala al
    3. Yeterli fazlaysa, fitness'e göre azaltarak non-dominated frontier'ı koru
    """
    # Non-dominated bireyleri bul
    non_dominated = [ind for ind in combined if ind.raw_fitness == 0]
    
    if len(non_dominated) < population_size:
        # Tümünü al, sonra geri kalanını fitness sırasına göre ekle
        remaining = [ind for ind in combined if ind.raw_fitness > 0]
        remaining.sort(key=lambda ind: ind.spea2_fitness)
        selected = non_dominated + remaining[:population_size - len(non_dominated)]
    elif len(non_dominated) > population_size:
        # Truncation: en iyi population_size kadarını seç
        # Clustering-based truncation (simple version: fitness'e göre)
        non_dominated.sort(key=lambda ind: ind.spea2_fitness)
        selected = non_dominated[:population_size]
    else:
        selected = non_dominated
    
    return selected


def spea2(user_id, population_size=20, generations=10,
         objective_names=("preference", "cost", "co2"),
         diversity_mode="none", alpha_diversity=1.0, lambda_penalty=10.0,
         crossover_rate=0.9, random_seed=None, k_neighbors=None,
         archive_size=None):
    """
    SPEA2 algoritması (Strength Pareto Evolutionary Algorithm 2)
    
    Args:
        user_id: Kullanıcı ID (1 veya 2)
        population_size: Popülasyon boyutu
        generations: Nesil sayısı
        objective_names: Optimize edilecek objectives
        diversity_mode: "none", "penalty", or "hard"
        alpha_diversity: Diversity penalty katsayısı
        lambda_penalty: Nutrient penalty ağırlığı
        random_seed: Tekrarlanabilirlik için
        k_neighbors: k-NN parametresi (None = auto)
    
    Returns:
        Dict with:
        - population: Final population
        - best_individuals: Optimal bireyleri (R(i) = 0)
        - history: Her nesilde non-dominated bireylerin sayısı
    """
    
    if random_seed is not None:
        random.seed(random_seed)

    if archive_size is None:
        archive_size = population_size

    # 1. Initialize population P0
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
    
    # Archive A0 (initially P0'dan non-dominated olanlar)
    assignment_fitness(population, k_neighbors)
    archive = [ind for ind in population if ind.raw_fitness == 0]
    
    history = []
    
    # 2. Ana loop
    for gen in range(generations):
        # Archive'dan tournament selection ile parent seç
        # Archive <N ise, P'nin non-dominated olanları da ekle (burada P=population)
        selection_pool = archive if archive else population
        
        # Offspring creation
        offspring = []
        while len(offspring) < population_size:
            parent1 = tournament_selection(selection_pool)
            parent2 = tournament_selection(selection_pool)
            child = create_offspring(parent1, parent2, crossover_rate=crossover_rate)
            child.evaluate()
            offspring.append(child)
        
        # Combined = P ∪ A
        combined = population + offspring
        
        # Fitness assignment
        assignment_fitness(combined, k_neighbors)
        
        # Environmental selection: R'ye göre (N bireyi seç)
        population = environmental_selection(combined, archive_size)
        
        # Archive update: R(i) = 0 olanlar
        archive = [ind for ind in population if ind.raw_fitness == 0]
        
        # Iteration bilgisi
        history.append({
            "generation": gen,
            "archive_size": len(archive),
            "best_fitness": min(population, key=lambda ind: ind.spea2_fitness).spea2_fitness
        })
    
    # Final fitness assignment
    assignment_fitness(population, k_neighbors)
    best_individuals = [ind for ind in population if ind.raw_fitness == 0]
    
    return {
        "population": population,
        "best_individuals": best_individuals,
        "history": history
    }

