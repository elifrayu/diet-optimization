"""
adim_adim_test.py
=================
Her adımı tek tek çalıştır, sonucu kontrol et.
DB'nin kurulu olduğu makinede çalıştır.

Kullanım: python adim_adim_test.py
"""
import sys, traceback

PASS = 0
FAIL = 0
STEP = 0

def baslik(text):
    global STEP
    STEP += 1
    print(f"\n{'='*65}")
    print(f"  ADIM {STEP}: {text}")
    print(f"{'='*65}")

def ok(msg):
    global PASS; PASS += 1
    print(f"  [OK]   {msg}")

def hata(msg):
    global FAIL; FAIL += 1
    print(f"  [FAIL] {msg}")

def exc(e):
    global FAIL; FAIL += 1
    print(f"  [HATA] {type(e).__name__}: {e}")


# ══════════════════════════════════════════════════════════════
# ADIM 1 — DB Bağlantısı
# ══════════════════════════════════════════════════════════════
baslik("DB BAGLANTISI")
try:
    from db import get_connection
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT 1")
    cursor.fetchone()
    cursor.close()
    conn.close()
    ok("MySQL baglantisi basarili")
except Exception as e:
    exc(e)
    print("\n  DURDURULDU: DB baglantisi olmadan devam edilemez.")
    print("  db.py icindeki host/user/password/database degerlerini kontrol edin.")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════
# ADIM 2 — Tablolar var mı?
# ══════════════════════════════════════════════════════════════
baslik("TABLOLAR MEVCUT MU?")
from db import get_connection

beklenen_tablolar = ["foods","food_group","food_nutrients",
                     "nutrients","dri","user","user_foods"]
try:
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute("SHOW TABLES")
    mevcut = {row[0].lower() for row in cur.fetchall()}
    cur.close(); conn.close()

    for t in beklenen_tablolar:
        if t in mevcut:
            ok(f"Tablo mevcut: {t}")
        else:
            hata(f"Tablo EKSIK: {t}")
except Exception as e:
    exc(e)


# ══════════════════════════════════════════════════════════════
# ADIM 3 — User 1 ve User 2 var mı?
# ══════════════════════════════════════════════════════════════
baslik("USER 1 VE USER 2")
try:
    conn = get_connection()
    cur  = conn.cursor(dictionary=True)
    for uid in [1, 2]:
        cur.execute("SELECT id,name,surname,age,gender FROM user WHERE id=%s", (uid,))
        row = cur.fetchone()
        if row:
            ok(f"User {uid}: {row['name']} {row['surname']}, "
               f"yas={row['age']}, cinsiyet={row['gender']}")
        else:
            hata(f"User {uid} bulunamadi!")
    cur.close(); conn.close()
except Exception as e:
    exc(e)


# ══════════════════════════════════════════════════════════════
# ADIM 4 — DRI yükleniyor mu?
# ══════════════════════════════════════════════════════════════
baslik("DRI YUKLEMESI (load_dri)")
try:
    from load_data import load_dri
    beklenen_besinler = ["energy","protein","carbohydrate","fiber","sodium"]

    for uid in [1, 2]:
        dri = load_dri(uid)
        eksik = [b for b in beklenen_besinler if b not in dri]
        if eksik:
            hata(f"User {uid}: DRI'da eksik besinler: {eksik}")
        else:
            e = dri['energy']
            ok(f"User {uid}: DRI tamam | Energy: [{e['RLL']:.0f}-{e['RUL']:.0f}] kcal")
            for b in beklenen_besinler:
                d = dri[b]
                print(f"         {b:15}: RLL={d['RLL']}, RUL={d['RUL']}")
except Exception as e:
    exc(e)


# ══════════════════════════════════════════════════════════════
# ADIM 5 — Havuzlar (get_pool)
# ══════════════════════════════════════════════════════════════
baslik("HAVUZ BOYUTLARI (get_pool)")
try:
    from load_data import get_pool
    for uid in [1, 2]:
        b, m = get_pool(uid)
        label = "Et Yiyen" if uid==1 else "Vegetaryen"
        if len(b) == 0:
            hata(f"User {uid} ({label}): Kahvalti havuzu BOŞ!")
        elif len(m) == 0:
            hata(f"User {uid} ({label}): Ana yemek havuzu BOŞ!")
        else:
            ok(f"User {uid} ({label}): kahvalti={len(b)}, "
               f"ana yemek={len(m)}, toplam={len(b)+len(m)}")

        # Toplam 405 olmalı (veya User2 için daha az — yasak yiyecekler çıkarıldı)
        if uid == 1 and len(b)+len(m) != 405:
            hata(f"User 1: Toplam {len(b)+len(m)}, 405 bekleniyor")
        if uid == 2 and len(b)+len(m) >= 405:
            hata(f"User 2: Toplam {len(b)+len(m)}, 405'ten az olmalı (vegeteryan filtresi)")
        elif uid == 2 and len(b)+len(m) < 405:
            ok(f"User 2: Vegeteryan filtresi calisiyor ({405-(len(b)+len(m))} yiyecek elendi)")
except Exception as e:
    exc(e)


# ══════════════════════════════════════════════════════════════
# ADIM 6 — Food verisi (load_food_data)
# ══════════════════════════════════════════════════════════════
baslik("FOOD VERISI (load_food_data)")
try:
    from load_data import load_food_data
    for uid in [1, 2]:
        foods, nutrients = load_food_data(uid)
        ok(f"User {uid}: {len(foods)} yiyecek, "
           f"{len(nutrients)} besin kaydi yuklendi")

        # Preference değerleri makul mu? (0-10 arası)
        prefs = [f['preference'] for f in foods.values() if f['preference'] is not None]
        neg   = [p for p in prefs if p < 0]
        if neg:
            hata(f"User {uid}: {len(neg)} yiyecekte preference<0 var (filtre kaçırdı?)")
        else:
            avg = sum(prefs)/len(prefs)
            ok(f"User {uid}: Preference ortalama={avg:.2f}, "
               f"min={min(prefs):.1f}, max={max(prefs):.1f}")

        # Besin verisi eksik var mı?
        eksik_besin = [fid for fid in foods if fid not in nutrients]
        if eksik_besin:
            print(f"  UYARI: {len(eksik_besin)} yiyecegin besin verisi yok "
                  f"(ilk 5: {eksik_besin[:5]})")
        else:
            ok(f"User {uid}: Tum yiyeceklerin besin verisi mevcut")
except Exception as e:
    exc(e)


# ══════════════════════════════════════════════════════════════
# ADIM 7 — Kromozom oluşturma
# ══════════════════════════════════════════════════════════════
baslik("KROMOZOM OLUSTURMA (create_individual)")
try:
    from chromosome import create_individual, crossover, mutate
    for uid in [1, 2]:
        chrom, bsize = create_individual(uid)

        # Permütasyon kontrolü
        if len(set(chrom)) != len(chrom):
            hata(f"User {uid}: Tekrar eden gen var!")
        elif None in chrom:
            hata(f"User {uid}: None gen var!")
        else:
            ok(f"User {uid}: Kromozom={len(chrom)} gen, "
               f"kahvalti={bsize}, ana={len(chrom)-bsize}, tekrar=Yok")

    # Crossover testi
    p1, bs = create_individual(1)
    p2, _  = create_individual(1)
    c1, c2 = crossover(p1, p2, bs)
    for name, c in [("c1", c1), ("c2", c2)]:
        if len(set(c)) == len(c):
            ok(f"Crossover {name}: {len(c)} gen, tekrar=Yok")
        else:
            hata(f"Crossover {name}: Tekrar eden gen var!")

    # Mutation testi
    mut = mutate(c1, bs)
    if len(set(mut)) == len(mut):
        ok(f"Mutation: Permütasyon korunuyor")
    else:
        hata(f"Mutation: Tekrar eden gen var!")
except Exception as e:
    exc(e)


# ══════════════════════════════════════════════════════════════
# ADIM 8 — Decode (greedy menü seçimi)
# ══════════════════════════════════════════════════════════════
baslik("DECODE (chromosome -> menu)")
try:
    from chromosome import create_individual
    from decoder import decode

    for uid in [1, 2]:
        chrom, bsize = create_individual(uid)
        result = decode(chrom, bsize, uid)

        b_count = len(result['breakfast'])
        m_count = len(result['main'])
        totals  = result['totals']
        dri     = result['dri']

        ok(f"User {uid}: Kahvalti={b_count}, Ana yemek={m_count}, "
           f"Toplam menu={len(result['menu'])} item")

        # DRI karşılaştırması
        print(f"\n  User {uid} besin degerleri vs DRI:")
        print(f"  {'Besin':15} {'Toplam':>10} {'RLL':>8} {'RUL':>8} {'Durum':>10}")
        print(f"  {'-'*55}")
        for n in ['energy','protein','carbohydrate','fiber','sodium']:
            v   = totals.get(n, 0)
            rll = dri.get(n,{}).get('RLL', 0)
            rul = dri.get(n,{}).get('RUL', 0)
            if rll <= v <= rul:
                durum = "OK"
            elif v < rll:
                durum = "EKSIK"
            else:
                durum = "FAZLA"
            print(f"  {n:15} {v:>10.1f} {rll:>8.0f} {rul:>8.0f} {durum:>10}")
        print()
except Exception as e:
    exc(e)
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════
# ADIM 9 — Fitness değerlendirmesi
# ══════════════════════════════════════════════════════════════
baslik("FITNESS DEGERLENDIRMESI (evaluate_solution)")
try:
    from chromosome import create_individual
    from fitness import evaluate_solution
    from metrics import calculate_constraint_compliance
    import random; random.seed(42)

    for uid in [1, 2]:
        chrom, bsize = create_individual(uid)
        result = evaluate_solution(chrom, bsize, uid,
                                   objective_names=("preference","cost","co2"),
                                   lambda_penalty=10.0)
        comply = calculate_constraint_compliance(result['totals'], result['dri'])
        ok(f"User {uid}: Pref={result['preference']:.1f}, "
           f"Cost={result['cost']:.2f}, CO2={result['co2']:.2f}, "
           f"Penalty={result['nutrient_penalty']:.4f}, "
           f"Uyum={comply}/5")
        print(f"  Objective vector (minimize): {tuple(f'{x:.2f}' for x in result['objectives'])}")
except Exception as e:
    exc(e)
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════
# ADIM 10 — NSGA-II (küçük parametre)
# ══════════════════════════════════════════════════════════════
baslik("NSGA-II (pop=10, gen=5 — hizli test)")
try:
    from nsga2 import nsga2
    from metrics import calculate_constraint_compliance
    import random; random.seed(42)

    for uid in [1, 2]:
        result = nsga2(user_id=uid, population_size=10, generations=5,
                       random_seed=42)
        front = result['best_front']
        if not front:
            hata(f"User {uid}: Pareto front BOŞ!")
        else:
            prefs   = [i.fitness_result['preference'] for i in front]
            costs   = [i.fitness_result['cost'] for i in front]
            comply  = [calculate_constraint_compliance(
                           i.fitness_result['totals'],
                           i.fitness_result['dri']) for i in front]
            ok(f"User {uid}: {len(front)} cozum | "
               f"Pref=[{min(prefs):.1f}-{max(prefs):.1f}] | "
               f"Cost=[{min(costs):.1f}-{max(costs):.1f}] | "
               f"Avg uyum={sum(comply)/len(comply):.1f}/5")
except Exception as e:
    exc(e)
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════
# ADIM 11 — SPEA2 (küçük parametre)
# ══════════════════════════════════════════════════════════════
baslik("SPEA2 (pop=10, gen=5 — hizli test)")
try:
    from spea2 import spea2
    import random; random.seed(42)

    for uid in [1, 2]:
        result = spea2(user_id=uid, population_size=10, generations=5,
                       random_seed=42)
        archive = result['best_individuals']
        if not archive:
            hata(f"User {uid}: SPEA2 arsivi BOS!")
        else:
            prefs = [i.fitness_result['preference'] for i in archive]
            ok(f"User {uid}: {len(archive)} arsiv cozumu | "
               f"Pref=[{min(prefs):.1f}-{max(prefs):.1f}]")
except Exception as e:
    exc(e)
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════
# ADIM 12 — Grafikler (matplotlib test)
# ══════════════════════════════════════════════════════════════
baslik("GRAFIK KUTUPHANESI (matplotlib)")
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    ok(f"matplotlib {matplotlib.__version__} hazir")
    ok(f"numpy {np.__version__} hazir")

    # Basit bir grafik üret, kaydedebiliyor muyuz?
    fig, ax = plt.subplots()
    ax.plot([1,2,3],[1,4,9])
    fig.savefig("/tmp/test_plot.png")
    plt.close()

    import os
    if os.path.exists("/tmp/test_plot.png"):
        ok("Grafik dosyaya kaydedilebildi (/tmp/test_plot.png)")
    else:
        hata("Grafik kaydedilemedi")
except Exception as e:
    exc(e)


# ══════════════════════════════════════════════════════════════
# SONUÇ
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"  SONUC: {PASS} OK, {FAIL} FAIL")
print(f"{'='*65}")

if FAIL == 0:
    print("""
  TUM ADIMLAR BASARILI.
  Artik tam parametrelerle calistirabilirsiniz:

    python compare_algorithms.py   (pop=50, gen=50)
    python visualizations.py       (tum grafikler)
""")
else:
    print(f"""
  {FAIL} ADIM BASARISIZ.
  [FAIL] satirlarini inceleyin ve duzeltip tekrar calistirin.
""")