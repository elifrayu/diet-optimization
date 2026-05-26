from load_data import load_food_data, load_dri

TARGET_NUTRIENTS = [
    "energy",
    "protein",
    "carbohydrate",
    "fiber",
    "sodium"
]

BREAKFAST_NUTRIENTS = [
    "energy",
    "protein"
]

# Aynı food group'tan çok fazla yemek seçilmesini engellemek için
MAX_SAME_GROUP_BREAKFAST = 2
MAX_SAME_GROUP_MAIN = 2


def get_food_nutrients(food_id, nutrients_by_food):
    return nutrients_by_food.get(food_id, {
        "energy": 0,
        "protein": 0,
        "carbohydrate": 0,
        "fiber": 0,
        "sodium": 0
    })


# Eklersem nutrient üst limitlerini aşar mı?
def can_add_food(food_id, totals, nutrients_by_food, upper_limits):
    food_nutrients = get_food_nutrients(food_id, nutrients_by_food)

    for nutrient in upper_limits:
        new_value = totals[nutrient] + food_nutrients[nutrient]

        if new_value > upper_limits[nutrient]:
            return False

    return True


def add_food(food_id, totals, nutrients_by_food):
    food_nutrients = get_food_nutrients(food_id, nutrients_by_food)

    for nutrient in TARGET_NUTRIENTS:
        totals[nutrient] += food_nutrients[nutrient]


def get_food_group_id(food_id, foods_by_id):
    """
    Food'un ait olduğu foodGroupId değerini döndürür.
    Eğer food bulunamazsa None döner.
    """
    if food_id not in foods_by_id:
        return None

    return foods_by_id[food_id].get("foodGroupId")


def can_add_by_group_limit(food_id, foods_by_id, group_counts, max_same_group):
    """
    Aynı foodGroupId'den çok fazla yemek seçilmesini engeller.
    Örneğin çorba grubundan 2 tane seçildiyse 3. çorbayı almaz.
    """
    group_id = get_food_group_id(food_id, foods_by_id)

    if group_id is None:
        return True

    current_count = group_counts.get(group_id, 0)

    if current_count >= max_same_group:
        return False

    return True


def add_group_count(food_id, foods_by_id, group_counts):
    """
    Seçilen yemeğin foodGroupId sayacını artırır.
    """
    group_id = get_food_group_id(food_id, foods_by_id)

    if group_id is None:
        return

    group_counts[group_id] = group_counts.get(group_id, 0) + 1


def decode(chromosome, breakfast_size, user_id):
    foods_by_id, nutrients_by_food = load_food_data(user_id)
    dri = load_dri(user_id)

    breakfast_part = chromosome[:breakfast_size]
    main_part = chromosome[breakfast_size:]

    selected_breakfast = []
    selected_main = []

    breakfast_group_counts = {}
    main_group_counts = {}

    totals = {
        "energy": 0,
        "protein": 0,
        "carbohydrate": 0,
        "fiber": 0,
        "sodium": 0
    }

    lower_factor = 0.90
    upper_factor = 1.15

    # Kahvaltı alt hedefleri: sadece energy + protein, günlük DRI'ın %35'i
    breakfast_lower_limits = {
        "energy": dri["energy"]["RLL"] * 0.35 * lower_factor,
        "protein": dri["protein"]["RLL"] * 0.35 * lower_factor
    }

    # Kahvaltı üst sınırı: sadece energy + protein, günlük DRI'ın %35'i
    breakfast_upper_limits = {
        "energy": dri["energy"]["RUL"] * 0.35 * upper_factor,
        "protein": dri["protein"]["RUL"] * 0.35 * upper_factor
    }

    # Günlük üst sınırlar: 5 nutrient için
    # Bunu kahvaltıda da kontrol ediyoruz ki kahvaltı carb/sodium gibi değerleri patlatmasın.
    daily_upper_limits = {
        nutrient: dri[nutrient]["RUL"] * upper_factor
        for nutrient in TARGET_NUTRIENTS
    }

    # Günlük alt sınırlar: 5 nutrient için
    daily_lower_limits = {
        nutrient: dri[nutrient]["RLL"] * lower_factor
        for nutrient in TARGET_NUTRIENTS
    }

    # 1) KAHVALTI DECODE
    for food_id in breakfast_part:

        # Aynı food group'tan çok fazla kahvaltılık seçilmesin
        if not can_add_by_group_limit(
            food_id,
            foods_by_id,
            breakfast_group_counts,
            MAX_SAME_GROUP_BREAKFAST
        ):
            continue

        # Kahvaltı energy/protein üst sınırını aşmasın
        if not can_add_food(food_id, totals, nutrients_by_food, breakfast_upper_limits):
            continue

        # Ayrıca günlük 5 nutrient üst sınırını da aşmasın
        if not can_add_food(food_id, totals, nutrients_by_food, daily_upper_limits):
            continue

        selected_breakfast.append(food_id)
        add_food(food_id, totals, nutrients_by_food)
        add_group_count(food_id, foods_by_id, breakfast_group_counts)

        # Kahvaltı için energy + protein alt hedefleri sağlandıysa dur
        if (
            totals["energy"] >= breakfast_lower_limits["energy"]
            and totals["protein"] >= breakfast_lower_limits["protein"]
        ):
            break

    # 2) ÖĞLE + AKŞAM DECODE
    for food_id in main_part:

        # Aynı food group'tan çok fazla ana öğün yemeği seçilmesin
        if not can_add_by_group_limit(
            food_id,
            foods_by_id,
            main_group_counts,
            MAX_SAME_GROUP_MAIN
        ):
            continue

        # 5 nutrient günlük üst sınırı aşılırsa yemeği alma
        if not can_add_food(food_id, totals, nutrients_by_food, daily_upper_limits):
            continue

        selected_main.append(food_id)
        add_food(food_id, totals, nutrients_by_food)
        add_group_count(food_id, foods_by_id, main_group_counts)

        all_lower_satisfied = all(
            totals[nutrient] >= daily_lower_limits[nutrient]
            for nutrient in TARGET_NUTRIENTS
        )

        if all_lower_satisfied:
            break

    menu = selected_breakfast + selected_main

    return {
        "breakfast": selected_breakfast,
        "main": selected_main,
        "menu": menu,
        "totals": totals,
        "dri": dri
    }