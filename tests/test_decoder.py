#test_decoder.py
from chromosome import create_individual
from decoder import decode

for user_id in [1, 2]:
    print(f"\n--- USER {user_id} ---")

    chromosome, breakfast_size = create_individual(user_id)
    result = decode(chromosome, breakfast_size, user_id)

    print("Breakfast:", result["breakfast"])
    print("Main:", result["main"])
    print("Menu size:", len(result["menu"]))
    print("Totals:", result["totals"])
    print("DRI:", result["dri"])