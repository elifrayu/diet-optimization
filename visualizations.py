"""
visualizations.py
=================
Multi-Objective Diet Optimization — Görselleştirmeler

Üretilen grafikler:
  1. pareto_front_user{1,2}.png   — Her kullanıcı için Pareto front scatter (overlay)
  2. convergence_user{1,2}.png    — Hypervolume vs. nesil (her iki algoritma)
  3. sample_menu_user{1,2}.png    — Örnek menü tablosu (3 Pareto çözümü)

Kullanım:
  python visualizations.py
"""

import sys
import os
import random
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np

# ─── Proje modülleri ─────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from nsga2  import nsga2
from spea2  import spea2
from metrics import (
    approximate_hypervolume,
    get_reference_point,
    calculate_constraint_compliance,
)
from load_data import load_food_data

# ─── Sabitler ────────────────────────────────────────────────────────────────
POP_SIZE   = 50
GENS       = 50
LAMBDA     = 10.0
SEED       = 42
OBJ_NAMES  = ("preference", "cost", "co2")
DIV_MODE   = "none"
HV_SAMPLES = 15_000

COLORS = {
    "nsga2": "#2E86AB",   # mavi
    "spea2": "#E84855",   # kırmızı
}
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


# ─── Yardımcı ────────────────────────────────────────────────────────────────
def deduplicate(individuals, key_fn):
    seen, unique = set(), []
    for ind in individuals:
        k = key_fn(ind)
        if k not in seen:
            seen.add(k)
            unique.append(ind)
    return unique


def ind_key(ind):
    fr = ind.fitness_result
    return (
        tuple(sorted(fr["menu"])),
        round(fr["preference"], 3),
        round(fr["cost"], 3),
        round(fr["co2"], 3),
    )


# ─── Nesil başına hypervolume hesabı ─────────────────────────────────────────
def compute_hv_history(history_objs_per_gen, ref_point, samples=HV_SAMPLES):
    """
    history_objs_per_gen: list of list of objective tuples (her nesil için)
    Döner: list of float (her nesil için HV)
    """
    hvs = []
    for objs in history_objs_per_gen:
        if not objs:
            hvs.append(0.0)
        else:
            hv = approximate_hypervolume(objs, ref_point, samples=samples, seed=SEED)
            hvs.append(hv)
    return hvs


# ─── Algoritmayı çalıştır, nesil HV geçmişini de hesapla ────────────────────
def run_and_collect(user_id):
    print(f"\n[User {user_id}] NSGA-II çalışıyor...")
    n_result = nsga2(
        user_id=user_id,
        population_size=POP_SIZE,
        generations=GENS,
        objective_names=OBJ_NAMES,
        lambda_penalty=LAMBDA,
        diversity_mode=DIV_MODE,
        random_seed=SEED,
    )

    print(f"[User {user_id}] SPEA2 çalışıyor...")
    s_result = spea2(
        user_id=user_id,
        population_size=POP_SIZE,
        generations=GENS,
        objective_names=OBJ_NAMES,
        lambda_penalty=LAMBDA,
        diversity_mode=DIV_MODE,
        random_seed=SEED,
    )

    n_front  = deduplicate(n_result["best_front"],       ind_key)
    s_front  = deduplicate(s_result["best_individuals"], ind_key)

    # Referans noktası (iki algoritmanın son frontlarından birlikte)
    all_objs = [ind.objectives for ind in n_front + s_front]
    ref      = get_reference_point(all_objs, margin=0.10)

    # NSGA-II: nesil başına first_front objectives'ten HV
    n_hist_objs = [h["first_front_objectives"] for h in n_result["history"]]
    n_hv_hist   = compute_hv_history(n_hist_objs, ref)

    # SPEA2: nesil başına arşiv objectives (history'de yok → her neslil population'dan çek)
    # SPEA2 history sadece archive_size & best_fitness tutuyor; objectives'e ulaşmak
    # için sonderçekimi son populasyondan yapıyoruz.
    # Alternatif: spea2'yi patch'leyip nesil bazı objeleri kaydet.
    # Burada basit yaklaşım: son HV'yi düz çiz (convergence eğrisi olmasa da gösterim için)
    # Gerçek convergence: spea2 history'sini extend et — aşağıda genişletilmiş wrapper kullanıyoruz.
    s_hv_hist = _spea2_hv_history(user_id, ref)

    return {
        "nsga2_front":   n_front,
        "spea2_front":   s_front,
        "ref":           ref,
        "n_hv_hist":     n_hv_hist,
        "s_hv_hist":     s_hv_hist,
        "n_result":      n_result,
        "s_result":      s_result,
    }


def _spea2_hv_history(user_id, ref_point):
    """
    SPEA2'yi yeniden çalıştırır fakat nesil başına arşiv objectives'i kaydeder.
    (spea2.py'yi değiştirmeden dışarıdan HV geçmişi üretir)
    """
    from spea2 import (Individual, create_individual, assignment_fitness,
                       environmental_selection, create_offspring, tournament_selection)
    from chromosome import create_individual as _ci

    random.seed(SEED)

    population = []
    for _ in range(POP_SIZE):
        chrom, bsize = _ci(user_id)
        ind = Individual(chrom, bsize, user_id,
                         objective_names=OBJ_NAMES,
                         lambda_penalty=LAMBDA,
                         diversity_mode=DIV_MODE)
        ind.evaluate()
        population.append(ind)

    assignment_fitness(population)
    archive = [ind for ind in population if ind.raw_fitness == 0]

    hv_hist = []

    for _ in range(GENS):
        sel_pool = archive if archive else population
        offspring = []
        while len(offspring) < POP_SIZE:
            p1 = tournament_selection(sel_pool)
            p2 = tournament_selection(sel_pool)
            child = create_offspring(p1, p2)
            child.evaluate()
            offspring.append(child)

        combined = population + offspring
        assignment_fitness(combined)
        population = environmental_selection(combined, POP_SIZE)
        archive = [ind for ind in population if ind.raw_fitness == 0]

        objs = [ind.objectives for ind in archive] if archive else []
        hv = approximate_hypervolume(objs, ref_point, samples=HV_SAMPLES, seed=SEED) if objs else 0.0
        hv_hist.append(hv)

    return hv_hist


# ═══════════════════════════════════════════════════════════════════════════════
# 1. PARETO FRONT SCATTER
# ═══════════════════════════════════════════════════════════════════════════════
def plot_pareto_front(user_id, data):
    n_front = data["nsga2_front"]
    s_front = data["spea2_front"]

    def objs(front): return [ind.objectives for ind in front]
    def raw(front, k): return [ind.fitness_result[k] for ind in front]

    n_objs = objs(n_front)
    s_objs = objs(s_front)

    # preference, cost, co2  →  objective vektörü minimize formunda
    # Gösterimde ham değerleri kullanıyoruz (preference: büyük iyi, cost/co2: küçük iyi)
    n_pref = raw(n_front, "preference"); n_cost = raw(n_front, "cost"); n_co2 = raw(n_front, "co2")
    s_pref = raw(s_front, "preference"); s_cost = raw(s_front, "cost"); s_co2 = raw(s_front, "co2")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Pareto Front — User {user_id}", fontsize=15, fontweight="bold", y=1.01)

    pairs = [
        (n_pref, s_pref, n_cost, s_cost, "Preference ↑", "Cost ↓",        "Preference vs. Cost"),
        (n_pref, s_pref, n_co2,  s_co2,  "Preference ↑", "CO₂ (kg) ↓",   "Preference vs. CO₂"),
        (n_cost, s_cost, n_co2,  s_co2,  "Cost (TL) ↓",  "CO₂ (kg) ↓",   "Cost vs. CO₂"),
    ]

    for ax, (nx, sx, ny, sy, xlabel, ylabel, title) in zip(axes, pairs):
        ax.scatter(nx, ny, c=COLORS["nsga2"], s=60, alpha=0.85,
                   edgecolors="white", linewidths=0.5, label="NSGA-II", zorder=3)
        ax.scatter(sx, sy, c=COLORS["spea2"], s=60, alpha=0.85,
                   edgecolors="white", linewidths=0.5, marker="^", label="SPEA2", zorder=3)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="semibold")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.spines[["top","right"]].set_visible(False)
        ax.legend(fontsize=9)

    plt.tight_layout()
    path = f"{OUT_DIR}/pareto_front_user{user_id}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# 2. CONVERGENCE CURVE (Hypervolume vs. Generation)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_convergence(user_id, data):
    n_hv = data["n_hv_hist"]
    s_hv = data["s_hv_hist"]
    gens = list(range(1, len(n_hv) + 1))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(gens, n_hv, color=COLORS["nsga2"], linewidth=2.2, label="NSGA-II", marker="o",
            markersize=4, markevery=5)
    ax.plot(gens[:len(s_hv)], s_hv, color=COLORS["spea2"], linewidth=2.2, label="SPEA2",
            marker="^", markersize=4, markevery=5, linestyle="--")

    ax.set_xlabel("Nesil (Generation)", fontsize=11)
    ax.set_ylabel("Hypervolume (Monte Carlo)", fontsize=11)
    ax.set_title(f"Convergence — Hypervolume vs. Generation  |  User {user_id}",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines[["top","right"]].set_visible(False)
    ax.legend(fontsize=10)
    plt.tight_layout()

    path = f"{OUT_DIR}/convergence_user{user_id}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# 3. SAMPLE MENU TABLE (3 Pareto çözümü)
# ═══════════════════════════════════════════════════════════════════════════════
def pick_diverse_solutions(front, n=3):
    """
    Front'tan n adet çeşitli çözüm seç:
    min_cost, max_preference, ve ortadaki (cost sıralamasında median).
    """
    if len(front) <= n:
        return front[:n]
    sorted_by_cost = sorted(front, key=lambda x: x.fitness_result["cost"])
    lo  = sorted_by_cost[0]
    hi  = max(front, key=lambda x: x.fitness_result["preference"])
    mid_idx = len(sorted_by_cost) // 2
    mid = sorted_by_cost[mid_idx]
    # deduplicate if lo/hi/mid overlap
    chosen = []
    seen   = set()
    for s in [lo, mid, hi]:
        k = id(s)
        if k not in seen:
            seen.add(k)
            chosen.append(s)
    # pad if needed
    for s in sorted_by_cost:
        if len(chosen) >= n:
            break
        if id(s) not in seen:
            seen.add(id(s))
            chosen.append(s)
    return chosen[:n]


def plot_sample_menu(user_id, front, algo_name="NSGA-II"):
    foods_by_id, _ = load_food_data(user_id)
    solutions = pick_diverse_solutions(front, n=3)

    sol_labels = ["Min. Cost", "Balanced", "Max. Pref."]

    fig = plt.figure(figsize=(18, 11))
    fig.patch.set_facecolor("#F7F9FC")
    fig.suptitle(
        f"Sample Pareto Solutions — {algo_name} | User {user_id}",
        fontsize=15, fontweight="bold", y=0.98
    )

    outer = gridspec.GridSpec(1, 3, figure=fig, wspace=0.06)

    nutrient_keys  = ["energy", "protein", "carbohydrate", "fiber", "sodium"]
    nutrient_units = ["kcal",   "g",       "g",            "g",    "mg"]
    nutrient_labels= ["Energy", "Protein", "Carbs",        "Fiber","Sodium"]

    for col_idx, (sol, sol_lbl) in enumerate(zip(solutions, sol_labels)):
        fr     = sol.fitness_result
        totals = fr["totals"]
        dri    = fr["dri"]
        menu   = fr["menu"]
        comply = calculate_constraint_compliance(totals, dri)

        inner = gridspec.GridSpecFromSubplotSpec(
            3, 1, subplot_spec=outer[col_idx],
            height_ratios=[1.2, 2.5, 3.2], hspace=0.05
        )

        # ── Başlık / özet kutusu ──────────────────────────────────────────
        ax_hdr = fig.add_subplot(inner[0])
        ax_hdr.set_facecolor(COLORS["nsga2"] if "NSGA" in algo_name else COLORS["spea2"])
        ax_hdr.axis("off")
        ax_hdr.text(0.5, 0.68, sol_lbl, ha="center", va="center",
                    fontsize=13, fontweight="bold", color="white",
                    transform=ax_hdr.transAxes)
        summary = (f"Pref: {fr['preference']:.1f}   Cost: {fr['cost']:.2f} TL   "
                   f"CO₂: {fr['co2']:.2f} kg   Compliance: {comply}/5")
        ax_hdr.text(0.5, 0.28, summary, ha="center", va="center",
                    fontsize=8.5, color="white", transform=ax_hdr.transAxes)

        # ── Besin değerleri çubuk grafiği ─────────────────────────────────
        ax_nut = fig.add_subplot(inner[1])
        ax_nut.set_facecolor("white")
        n_nuts   = len(nutrient_keys)
        x        = np.arange(n_nuts)
        ratios   = []
        bar_clrs = []
        for nk in nutrient_keys:
            rll = dri.get(nk, {}).get("RLL", 1)
            rul = dri.get(nk, {}).get("RUL", 1)
            val = totals.get(nk, 0)
            mid = (rll + rul) / 2
            r   = val / mid if mid > 0 else 0
            ratios.append(r)
            if rll <= val <= rul:
                bar_clrs.append("#4CAF50")    # yeşil — uyumlu
            elif val < rll:
                bar_clrs.append("#FF9800")    # turuncu — eksik
            else:
                bar_clrs.append("#F44336")    # kırmızı — fazla

        bars = ax_nut.bar(x, ratios, color=bar_clrs, width=0.55, zorder=3,
                          edgecolor="white", linewidth=0.8)
        ax_nut.axhline(1.0, color="#555555", linewidth=1.2, linestyle="--",
                       alpha=0.7, label="DRI midpoint", zorder=4)
        ax_nut.set_xticks(x)
        ax_nut.set_xticklabels(
            [f"{l}\n({u})" for l, u in zip(nutrient_labels, nutrient_units)],
            fontsize=8
        )
        ax_nut.set_ylabel("Actual / Midpoint", fontsize=8)
        ax_nut.set_title("Nutrient Compliance", fontsize=9, fontweight="semibold", pad=4)
        ax_nut.set_ylim(0, max(max(ratios)*1.15, 1.3))
        ax_nut.grid(axis="y", alpha=0.3, linestyle="--", zorder=0)
        ax_nut.spines[["top","right"]].set_visible(False)
        # Değer etiketleri
        for bar, nk, unit in zip(bars, nutrient_keys, nutrient_units):
            val = totals.get(nk, 0)
            ax_nut.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f"{val:.0f}", ha="center", va="bottom", fontsize=7)

        # Açıklama renk kutuları
        patches = [
            mpatches.Patch(color="#4CAF50", label="Within DRI"),
            mpatches.Patch(color="#FF9800", label="Below RLL"),
            mpatches.Patch(color="#F44336", label="Above RUL"),
        ]
        ax_nut.legend(handles=patches, fontsize=7, loc="upper right",
                      framealpha=0.7)

        # ── Menü öğeleri tablosu ──────────────────────────────────────────
        ax_tbl = fig.add_subplot(inner[2])
        ax_tbl.axis("off")

        # Sütun başlıkları
        col_hdrs = ["#", "Food Name", "Pref", "Cost", "CO₂"]
        col_w    = [0.05, 0.52, 0.14, 0.14, 0.14]

        y_start = 0.97
        row_h   = 0.078
        pad     = 0.01

        # Başlık satırı
        x_cursor = pad
        for hdr, w in zip(col_hdrs, col_w):
            ax_tbl.text(x_cursor + w/2, y_start, hdr,
                        ha="center", va="top", fontsize=8, fontweight="bold",
                        color="white",
                        bbox=dict(boxstyle="square,pad=0.3",
                                  facecolor="#2E4057", edgecolor="none"),
                        transform=ax_tbl.transAxes)
            x_cursor += w

        # Menü satırları (max 12)
        display_menu = menu[:12]
        for row_i, food_id in enumerate(display_menu):
            y_row = y_start - (row_i + 1) * row_h
            bg    = "#EEF2F7" if row_i % 2 == 0 else "white"
            food  = foods_by_id.get(food_id, {})
            name  = food.get("name", f"ID:{food_id}")
            # Uzun isimleri kırp
            if len(name) > 28:
                name = name[:26] + "…"
            pref = food.get("preference", 0)
            cost = food.get("cost", 0)
            co2  = food.get("co2", 0)
            vals = [str(row_i+1), name, f"{pref:.1f}", f"{cost:.2f}", f"{co2:.2f}"]

            x_cursor = pad
            for val, w, hdr in zip(vals, col_w, col_hdrs):
                ha = "left" if hdr == "Food Name" else "center"
                ax_tbl.text(x_cursor + (0 if ha=="left" else w/2), y_row,
                            val, ha=ha, va="top", fontsize=7.5,
                            bbox=dict(boxstyle="square,pad=0.25",
                                      facecolor=bg, edgecolor="none"),
                            transform=ax_tbl.transAxes)
                x_cursor += w

        if len(menu) > 12:
            ax_tbl.text(0.5, y_start - 13*row_h,
                        f"(+{len(menu)-12} more items)",
                        ha="center", va="top", fontsize=7.5,
                        color="gray", transform=ax_tbl.transAxes)

        ax_tbl.set_title(f"Menu ({len(menu)} items)", fontsize=9,
                         fontweight="semibold", pad=4)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = f"{OUT_DIR}/sample_menu_user{user_id}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓ {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    all_paths = []

    for user_id in [1, 2]:
        print(f"\n{'='*55}")
        print(f"  USER {user_id}")
        print(f"{'='*55}")

        data = run_and_collect(user_id)

        print(f"\n[User {user_id}] Grafikler üretiliyor...")

        p1 = plot_pareto_front(user_id, data)
        p2 = plot_convergence(user_id, data)
        p3 = plot_sample_menu(user_id, data["nsga2_front"], algo_name="NSGA-II")

        all_paths += [p1, p2, p3]

    print(f"\n{'='*55}")
    print(f"  Tüm grafikler hazır → {OUT_DIR}")
    print(f"{'='*55}")
    return all_paths


if __name__ == "__main__":
    main()