from db import get_connection

def load_foods():
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("""
        SELECT f.id, f.name, f.cost,
               (f.preparingTime + f.cookingTime) AS time,
               f.co2,
               uf.preference
        FROM foods f
        JOIN user_foods uf ON f.id = uf.foodId
        WHERE uf.userId = 1
    """)
    foods = cursor.fetchall()

    cursor.execute("""
        SELECT fn.foodId, n.name, fn.quantity
        FROM food_nutrients fn
        JOIN nutrients n ON fn.nutrientId = n.id
    """)
    nutrient_rows = cursor.fetchall()

    nutrient_map = {}
    for row in nutrient_rows:
        fid = row["foodId"]
        name = row["name"].lower()

        if fid not in nutrient_map:
            nutrient_map[fid] = {}

        nutrient_map[fid][name] = row["quantity"]

    cursor.close()
    conn.close()

    return foods, nutrient_map