from decoder import decode
from load_data import load_food_data, load_dri

# Nutrient violation hesaplaması
def calculate_nutrient_penalty(totals, dri):
    """
    Her nutrient için:
    viol_low_j = max(0, RLL_j - v_j) / (RUL_j - RLL_j)
    viol_high_j = max(0, v_j - RUL_j) / (RUL_j - RLL_j)

    Penalty: R = 0.7 * sum(viol_low) + 0.3 * sum(viol_high)
    """
    nutrients = ["energy", "protein", "carbohydrate", "fiber", "sodium"]

    viol_low_sum = 0.0
    viol_high_sum = 0.0

    for nutrient in nutrients:
        if nutrient not in dri:
            continue

        rll = dri[nutrient]["RLL"]
        rul = dri[nutrient]["RUL"]
        v = totals[nutrient]

        nutrient_range = rul - rll
        if nutrient_range <= 0:
            continue

        # Lower bound violation
        viol_low = max(0, rll - v) / nutrient_range
        viol_low_sum += viol_low

        # Upper bound violation
        viol_high = max(0, v - rul) / nutrient_range
        viol_high_sum += viol_high

    penalty = 0.7 * viol_low_sum + 0.3 * viol_high_sum
    return penalty


# Diversity hesaplaması (distinct foodGroupId sayısı)
def calculate_diversity_count(menu, foods_by_id):
    """
    Selected menu'deki distinct foodGroupId sayısını döndürür.
    """
    if not menu:
        return 0

    food_groups = set()
    for food_id in menu:
        if food_id in foods_by_id:
            food_groups.add(foods_by_id[food_id]["foodGroupId"])

    return len(food_groups)


# Total time hesaplaması
def calculate_total_time(menu, foods_by_id):
    """
    Selected menu'deki tüm yemeklerin toplam hazırlanma + pişirme süresi.
    """
    total_time = 0
    for food_id in menu:
        if food_id in foods_by_id:
            total_time += foods_by_id[food_id]["time"]

    return total_time


# Total preference hesaplaması
def calculate_total_preference(menu, foods_by_id):
    """
    Selected menu'deki tüm yemeklerin preference değerlerinin toplamı.
    """
    total_preference = 0
    for food_id in menu:
        if food_id in foods_by_id:
            total_preference += foods_by_id[food_id]["preference"]

    return total_preference


# Total cost hesaplaması
def calculate_total_cost(menu, foods_by_id):
    """
    Selected menu'deki tüm yemeklerin total cost'u.
    """
    total_cost = 0
    for food_id in menu:
        if food_id in foods_by_id:
            total_cost += foods_by_id[food_id]["cost"]

    return total_cost


# Total CO2 hesaplaması
def calculate_total_co2(menu, foods_by_id):
    """
    Selected menu'deki tüm yemeklerin total CO2 emissions'ı.
    """
    total_co2 = 0
    for food_id in menu:
        if food_id in foods_by_id:
            total_co2 += foods_by_id[food_id]["co2"]

    return total_co2


# Ana fitness/objective evaluasyon fonksiyonu
def evaluate_solution(chromosome, breakfast_size, user_id,
                     objective_names=("preference", "cost", "co2"),
                     diversity_mode="none",
                     alpha_diversity=1.0,
                     lambda_penalty=10.0):
    """
    Chromosom'u evaluate eder ve objective vector döndürür.

    Args:
        chromosome: Permutation chromosom (food IDs)
        breakfast_size: Kahvaltı parçasının boyutu
        user_id: Kullanıcı ID (1 veya 2)
        objective_names: Tuple of objectives to include in result
                        Default: ("preference", "cost", "co2")
        diversity_mode: "none", "penalty", or "hard"
        alpha_diversity: Diversity penalty katsayısı (penalty mode'da)
        lambda_penalty: Nutrient violation penalty ağırlığı

    Returns:
        Dict with:
        - objectives: tuple of float values (for MOEAs)
        - preference: float
        - cost: float
        - co2: float
        - time: float (raw metric, not in objectives)
        - diversity_count: int
        - nutrient_penalty: float
        - total_penalty: float (with diversity if applicable)
        - totals: dict of nutrient totals
        - dri: dict of DRI values
        - menu: list of selected food IDs
    """

    # 1. Decode chromosom
    decoded = decode(chromosome, breakfast_size, user_id)
    menu = decoded["menu"]
    totals = decoded["totals"]
    dri = decoded["dri"]

    # 2. Load food data
    foods_by_id, _ = load_food_data(user_id)

    # 3. Metrikleri hesapla
    nutrient_penalty = calculate_nutrient_penalty(totals, dri)
    diversity_count = calculate_diversity_count(menu, foods_by_id)
    total_time = calculate_total_time(menu, foods_by_id)
    total_preference = calculate_total_preference(menu, foods_by_id)
    total_cost = calculate_total_cost(menu, foods_by_id)
    total_co2 = calculate_total_co2(menu, foods_by_id)

    # 4. Diversity penalty hesapla
    diversity_penalty = 0.0
    if diversity_mode == "penalty":
        diversity_penalty = alpha_diversity / max(diversity_count, 1)
    elif diversity_mode == "hard":
        if diversity_count < 4:
            diversity_penalty = 10.0

    total_penalty = nutrient_penalty + diversity_penalty

    # 5. Objective vector hesapla (minimize format)
    objective_preference = -(total_preference - lambda_penalty * total_penalty)
    objective_cost = total_cost + lambda_penalty * total_penalty
    objective_co2 = total_co2 + lambda_penalty * total_penalty

    # Objective map
    objective_map = {
        "preference": objective_preference,
        "cost": objective_cost,
        "co2": objective_co2
    }

    # 6. İstenen objectives'i seç
    objectives = tuple(objective_map[name] for name in objective_names)

    return {
        "objectives": objectives,
        "preference": total_preference,
        "cost": total_cost,
        "co2": total_co2,
        "time": total_time,
        "diversity_count": diversity_count,
        "nutrient_penalty": nutrient_penalty,
        "total_penalty": total_penalty,
        "totals": totals,
        "dri": dri,
        "menu": menu,
        "breakfast": decoded["breakfast"],
        "main": decoded["main"]
    }

