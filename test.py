from load_data import load_foods

foods, nutrient_map = load_foods()

print("Toplam food:", len(foods))

# ilk 3 taneyi göster
for f in foods[:3]:
    print(f)

print("\nNutrient örnek:")
print(nutrient_map[foods[0]["id"]])