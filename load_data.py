#Veritabanından veri çeker, 4 fonksiyon: get_pool (hangi yiyecekler kahvaltıya ait, hangisi ana yemeğe), 
# load_food_data (her yiyeceğin maliyet/tercih/co2/süre bilgileri + besin değerleri), 
# load_dri (kullanıcının yaş+cinsiyetine göre günlük besin sınırları), 
# normalize_nutrient_name (DB'deki uzun nutrient isimlerini energy, protein gibi kısa anahtarlara çevirir).

from db import get_connection

# Kahvaltı grupları (food_group.name)
BREAKFAST_GROUP_NAMES = (
    'Dairy Products', 'Chicken Products', 'Meat Products',
    'Jams Syrups', 'Honey Products', 'Seed Bean Olive',
    'Salads', 'Bakery', 'Sweet Things marmelades',
    'Beverages', 'Cereals', 'Beverages 2', 'Bakery 2'
)

# Nutrient isimlerini normalize eder (DB → bizim sistem)
def normalize_nutrient_name(name):
    name = name.lower()

    if name == "energy":
        return "energy"
    elif name == "protein":
        return "protein"
    elif name == "carbohydrate, by difference":
        return "carbohydrate"
    elif name == "fiber, total dietary":
        return "fiber"
    elif name == "sodium, na":
        return "sodium"

    return None


# Kullanıcıya göre preference kolonu seçilir
def get_preference_column(user_id):
    if user_id == 1:
        return "preference"
    elif user_id == 2:
        return "preference2"
    else:
        raise ValueError("Bu projede sadece user_id 1 ve 2 kullanılacak.")


# Kahvaltı ve ana yemek havuzlarını döndürür
def get_pool(user_id):
    conn = get_connection()
    cursor = conn.cursor()

    pref_col = get_preference_column(user_id)
    placeholders = ",".join(["%s"] * len(BREAKFAST_GROUP_NAMES))

    # Kahvaltı
    query_breakfast = f"""
        SELECT f.id
        FROM foods f
        JOIN food_group fg ON f.foodGroupId = fg.id
        WHERE f.{pref_col} != -1
          AND fg.name IN ({placeholders})
    """
    cursor.execute(query_breakfast, BREAKFAST_GROUP_NAMES)
    breakfast_pool = [row[0] for row in cursor.fetchall()]

    # Öğle/Akşam
    query_main = f"""
        SELECT f.id
        FROM foods f
        JOIN food_group fg ON f.foodGroupId = fg.id
        WHERE f.{pref_col} != -1
          AND fg.name NOT IN ({placeholders})
    """
    cursor.execute(query_main, BREAKFAST_GROUP_NAMES)
    main_meal_pool = [row[0] for row in cursor.fetchall()]

    cursor.close()
    conn.close()

    return breakfast_pool, main_meal_pool


# Food bilgileri + nutrient map
def load_food_data(user_id):
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    pref_col = get_preference_column(user_id)

    cursor.execute(f"""
        SELECT 
            f.id,
            f.name,
            f.foodGroupId,
            f.cost,
            (f.preparingTime + f.cookingTime) AS time,
            f.co2,
            f.{pref_col} AS preference
        FROM foods f
        WHERE f.{pref_col} != -1
    """)

    food_rows = cursor.fetchall()

    # id → food bilgisi
    foods_by_id = {}
    for row in food_rows:
        foods_by_id[row["id"]] = row

    # nutrient bilgileri
    cursor.execute("""
        SELECT 
            fn.foodId,
            n.name,
            fn.quantity
        FROM food_nutrients fn
        JOIN nutrients n ON fn.nutrientId = n.id
    """)

    nutrient_rows = cursor.fetchall()

    nutrients_by_food = {}

    for row in nutrient_rows:
        food_id = row["foodId"]
        nutrient_name = normalize_nutrient_name(row["name"])

        if nutrient_name is None:
            continue

        if food_id not in nutrients_by_food:
            nutrients_by_food[food_id] = {
                "energy": 0,
                "protein": 0,
                "carbohydrate": 0,
                "fiber": 0,
                "sodium": 0
            }

        nutrients_by_food[food_id][nutrient_name] = row["quantity"]

    cursor.close()
    conn.close()

    return foods_by_id, nutrients_by_food


# Kullanıcıya göre DRI hesaplar (age + gender)
def load_dri(user_id):
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    # 1. Kullanıcı bilgisi
    cursor.execute("""
        SELECT age, gender
        FROM user
        WHERE id = %s
    """, (user_id,))
    user = cursor.fetchone()

    age = user["age"]
    gender = user["gender"].lower()

    # 2. DRI çek
    cursor.execute("""
        SELECT 
            n.name,
            d.RLL,
            d.RUL
        FROM dri d
        JOIN nutrients n ON d.nutrient_id = n.id
        WHERE d.low_age <= %s
          AND d.up_age >= %s
          AND LOWER(d.gender) = %s
    """, (age, age, gender))

    rows = cursor.fetchall()

    dri = {}

    for row in rows:
        nutrient_name = normalize_nutrient_name(row["name"])

        if nutrient_name is None:
            continue

        dri[nutrient_name] = {
            "RLL": row["RLL"],
            "RUL": row["RUL"]
        }

    cursor.close()
    conn.close()

    return dri