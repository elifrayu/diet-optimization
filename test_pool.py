from load_data import get_pool

for user_id in [1, 2]:
    b, m = get_pool(user_id)

    print(f"\nUser {user_id}")
    print("Kahvaltı:", len(b))
    print("Öğle/Akşam:", len(m))
    print("Toplam:", len(b) + len(m))