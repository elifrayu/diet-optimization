import random
from load_data import get_pool


# Bu fonksiyon bir kullanıcı için başlangıç kromozomu oluşturur.
# Kromozom iki parçadan oluşur:
# 1) Kahvaltı yemekleri
# 2) Öğle/Akşam yemekleri
def create_individual(user_id):

    # Kullanıcıya göre kahvaltı ve ana yemek havuzlarını veritabanından alıyoruz.
    # User 1 ve User 2 için havuzlar farklı olabilir.
    breakfast_pool, main_meal_pool = get_pool(user_id)

    # Orijinal havuzları bozmamak için kopyalarını alıyoruz.
    b_part = breakfast_pool.copy()
    m_part = main_meal_pool.copy()

    # Kahvaltı ve ana yemek kısımlarını ayrı ayrı karıştırıyoruz.
    # Çünkü kromozom bir permütasyon yapısıdır.
    random.shuffle(b_part)
    random.shuffle(m_part)

    # Kromozom: [kahvaltı kısmı | öğle/akşam kısmı]
    chromosome = b_part + m_part

    # Kahvaltı kısmının uzunluğunu tutuyoruz.
    # Böylece kromozomu sonradan iki parçaya ayırabiliriz.
    breakfast_size = len(b_part)

    return chromosome, breakfast_size


# Bu fonksiyon iki parent parçası arasında Order Crossover (OX) uygular.
# OX, permütasyon yapısını korumak için kullanılır.
# Yani aynı food id'nin iki kez gelmesini engeller.
def ox_crossover_part(part1, part2):

    # Parçanın uzunluğu alınır.
    size = len(part1)

    # Rastgele iki kesme noktası seçilir.
    start, end = sorted(random.sample(range(size), 2))

    # Çocuk kromozom başlangıçta boş oluşturulur.
    child = [None] * size

    # Parent1'den seçilen aralık çocuğa aynen aktarılır.
    child[start:end] = part1[start:end]

    # Parent2'de olup child içinde olmayan genler sırayla alınır.
    # Böylece eksik kalan yerler parent2 sırasına göre doldurulur.
    p2_items = [item for item in part2 if item not in child]

    index = 0

    # Child içindeki boş None alanları doldurulur.
    for i in range(size):
        if child[i] is None:
            child[i] = p2_items[index]
            index += 1

    return child


# Bu fonksiyon iki parent kromozomdan iki child üretir.
# Kahvaltı kısmı ve ana yemek kısmı ayrı ayrı crossover'a sokulur.
def crossover(parent1, parent2, breakfast_size):

    # Parent1'i kahvaltı ve ana yemek olarak ikiye ayırıyoruz.
    p1_b = parent1[:breakfast_size]
    p1_m = parent1[breakfast_size:]

    # Parent2'yi kahvaltı ve ana yemek olarak ikiye ayırıyoruz.
    p2_b = parent2[:breakfast_size]
    p2_m = parent2[breakfast_size:]

    # Kahvaltı parçaları arasında OX crossover uygulanır.
    child1_b = ox_crossover_part(p1_b, p2_b)
    child2_b = ox_crossover_part(p2_b, p1_b)

    # Ana yemek parçaları arasında da OX crossover uygulanır.
    child1_m = ox_crossover_part(p1_m, p2_m)
    child2_m = ox_crossover_part(p2_m, p1_m)

    # Kahvaltı ve ana yemek parçaları tekrar birleştirilir.
    child1 = child1_b + child1_m
    child2 = child2_b + child2_m

    return child1, child2


# Bu fonksiyon kromozoma swap mutation uygular.
# Swap mutation: iki genin yerini değiştirmek demektir.
def mutate(chromosome, breakfast_size):

    # Orijinal kromozomu bozmamak için kopyasını alıyoruz.
    mutated = chromosome.copy()

    # PDF'teki mantığa göre mutation rate = 1 / n
    # Kahvaltı ve ana yemek kısımları için ayrı ayrı hesaplanır.
    mutation_rate_b = 1 / breakfast_size
    mutation_rate_m = 1 / (len(chromosome) - breakfast_size)

    # Kahvaltı kısmında mutation uygulanıp uygulanmayacağına karar verilir.
    if random.random() < mutation_rate_b:

        # Kahvaltı kısmından rastgele iki indeks seçilir.
        i, j = random.sample(range(0, breakfast_size), 2)

        # Seçilen iki genin yeri değiştirilir.
        mutated[i], mutated[j] = mutated[j], mutated[i]

    # Ana yemek kısmında mutation uygulanıp uygulanmayacağına karar verilir.
    if random.random() < mutation_rate_m:

        # Ana yemek kısmından rastgele iki indeks seçilir.
        i, j = random.sample(range(breakfast_size, len(mutated)), 2)

        # Seçilen iki genin yeri değiştirilir.
        mutated[i], mutated[j] = mutated[j], mutated[i]

    return mutated


# Bu fonksiyon kromozomun doğru oluşup oluşmadığını kontrol eder.
# Test/debug amaçlı kullanılır.
def check_individual(chromosome, breakfast_size):

    # Kromozom iki parçaya ayrılır.
    breakfast_part = chromosome[:breakfast_size]
    main_part = chromosome[breakfast_size:]

    # Toplam gen sayısı yazdırılır.
    print("Toplam gen:", len(chromosome))

    # Kahvaltı kısmındaki gen sayısı yazdırılır.
    print("Kahvaltı gen sayısı:", len(breakfast_part))

    # Ana yemek kısmındaki gen sayısı yazdırılır.
    print("Ana yemek gen sayısı:", len(main_part))

    # Kromozomda tekrar eden food id var mı kontrol edilir.
    # Permütasyon yapısında tekrar olmamalıdır.
    print("Tekrar eden gen var mı?", len(chromosome) != len(set(chromosome)))