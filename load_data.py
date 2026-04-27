from db import get_connection


def get_pool():
    conn = get_connection()
    if conn is None:
        print("Veritabanina bağlanilamadiği için işlem durduruldu.")
        return [], []
    
    # Bağlantı başarılıysa cursor'u oluşturuyoruz
    cursor = conn.cursor()

    breakfast_group_names = (
        'Dairy Products', 'Chicken Products', 'Meat Products', 
        'Jams Syrups', 'Honey Products', 'Seed Bean Olive',  
        'Salads', 'Bakery', 'Sweet Things Marmelades', 
        'Beverages', 'Cereals', 'Beverages 2', 'Bakery 2'
    )

    # 1. Kahvaltı Sorgusu
    query_breakfast = f"SELECT f.id FROM foods f JOIN food_group fg ON f.foodGroupId = fg.id WHERE fg.name IN {breakfast_group_names}"
    cursor.execute(query_breakfast)
    breakfast_pool = [row[0] for row in cursor.fetchall()]

    # 2. Ana Yemek Sorgusu
    query_main = f"SELECT f.id FROM foods f JOIN food_group fg ON f.foodGroupId = fg.id WHERE fg.name NOT IN {breakfast_group_names}"
    cursor.execute(query_main)
    main_meal_pool = [row[0] for row in cursor.fetchall()]

    # Bağlantıyı ve cursor'u kapatıyoruz
    cursor.close()
    conn.close()

    return breakfast_pool, main_meal_pool

# Kontrol
b_list, m_list = get_pool()
print(f"Kahvalti: {len(b_list)} | ogle/Aksam: {len(m_list)}")