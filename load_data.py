from db import get_connection


BREAKFAST_GROUP_NAMES = (
    'Dairy Products', 'Chicken Products', 'Meat Products',
    'Jams Syrups', 'Honey Products', 'Seed Bean Olive',
    'Salads', 'Bakery', 'Sweet Things Marmelades',
    'Beverages', 'Cereals', 'Beverages 2', 'Bakery 2'
)


# Kullanıcıya göre kahvaltı ve ana yemek havuzlarını döndürür.
# preference = -1 olan yiyecekler o kullanıcı için yasak kabul edilir.
def get_pool(user_id):
    conn = get_connection()
    cursor = conn.cursor()

    placeholders = ",".join(["%s"] * len(BREAKFAST_GROUP_NAMES))

    # Kahvaltı havuzu
    query_breakfast = f"""
        SELECT f.id
        FROM foods f
        JOIN food_group fg ON f.foodGroupId = fg.id
        JOIN user_foods uf ON uf.foodId = f.id
        WHERE uf.userId = %s
          AND uf.preference != -1
          AND fg.name IN ({placeholders})
    """

    cursor.execute(query_breakfast, (user_id, *BREAKFAST_GROUP_NAMES))
    breakfast_pool = [row[0] for row in cursor.fetchall()]

    # Öğle/Akşam havuzu
    query_main = f"""
        SELECT f.id
        FROM foods f
        JOIN food_group fg ON f.foodGroupId = fg.id
        JOIN user_foods uf ON uf.foodId = f.id
        WHERE uf.userId = %s
          AND uf.preference != -1
          AND fg.name NOT IN ({placeholders})
    """

    cursor.execute(query_main, (user_id, *BREAKFAST_GROUP_NAMES))
    main_meal_pool = [row[0] for row in cursor.fetchall()]

    cursor.close()
    conn.close()

    return breakfast_pool, main_meal_pool