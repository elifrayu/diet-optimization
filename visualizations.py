"""
visualizations.py
=================
Multi-Objective Diet Optimization — Tüm Grafikler

PDF'in istediği 3 deney + zorunlu görseller:
  Deney 1 (User karşılaştırma):
    - user_comparison.png   → User 1 vs User 2 Pareto fronts overlay
  Deney 2 (Algoritma karşılaştırma):
    - pareto_user{1,2}.png  → NSGA-II vs SPEA2 scatter (3 eksen kombinasyonu)
    - convergence_user{1,2}.png → Hypervolume vs nesil
    - hv_bar.png            → Algoritma bazlı hypervolume bar
  Deney 3 (Diversity etkisi):
    - diversity_impact.png  → diversity_mode=none vs penalty vs hard
  Menü tabloları:
    - menu_nsga2_user{1,2}.png
    - menu_spea2_user{1,2}.png

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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nsga2  import nsga2
from spea2  import spea2
from metrics import (
    approximate_hypervolume,
    get_reference_point,
    calculate_constraint_compliance,
)
from load_data import load_food_data

# ─── Parametreler ─────────────────────────────────────────────────────────────
POP_SIZE   = 50
GENS       = 50
LAMBDA     = 10.0
SEED       = 42
OBJ_NAMES  = ("preference", "cost", "co2")
HV_SAMPLES = 15_000

COLORS = {
    "nsga2":  "#1B4F72",
    "spea2":  "#C0392B",
    "user1":  "#1B4F72",
    "user2":  "#C0392B",
    "green":  "#27AE60",
    "orange": "#E67E22",
    "red":    "#E74C3C",
}

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


# ─── Yardımcı ─────────────────────────────────────────────────────────────────

def deduplicate(individuals):
    seen, unique = set(), []
    for ind in individuals:
        fr = ind.fitness_result
        k = (tuple(sorted(fr["menu"])),
             round(fr["preference"], 3),
             round(fr["cost"], 3),
             round(fr["co2"], 3))
        if k not in seen:
            seen.add(k)
            unique.append(ind)
    return unique


def raw(front, key):
    return [ind.fitness_result[key] for ind in front]


def hv_from_objs(objs, ref):
    if not objs:
        return 0.0
    return approximate_hypervolume(objs, ref, samples=HV_SAMPLES, seed=SEED)


# ─── Algoritmayı çalıştır, SPEA2 için nesil HV geçmişini de üret ──────────────

def run_and_collect(user_id, diversity_mode="none"):
    """
    Her iki algoritmayı çalıştır.
    SPEA2 için convergence: spea2() sarmalanarak nesil bazı objectives kaydedilir.
    Böylece SPEA2 ikinci kez çalıştırılmaz.
    """
    random.seed(SEED)

    # ── NSGA-II ──
    n_result = nsga2(
        user_id=user_id,
        population_size=POP_SIZE,
        generations=GENS,
        objective_names=OBJ_NAMES,
        lambda_penalty=LAMBDA,
        diversity_mode=diversity_mode,
        random_seed=SEED,
    )
    n_front = deduplicate(n_result["best_front"])
    n_hv_hist = []
    for h in n_result["history"]:
        objs = h["first_front_objectives"]
        # Referans henüz bilinmiyor, geçici olarak kaydediyoruz
        n_hv_hist.append(objs)

    # ── SPEA2 — nesil bazı HV için wrapper ──
    s_result, s_gen_objs = _run_spea2_with_history(user_id, diversity_mode)
    s_front = deduplicate(s_result["best_individuals"])

    # Ortak referans noktası
    all_objs = ([ind.objectives for ind in n_front] +
                [ind.objectives for ind in s_front])
    ref = get_reference_point(all_objs, margin=0.10)

    # NSGA-II HV geçmişini hesapla
    n_hv_curve = [hv_from_objs(objs, ref) for objs in n_hv_hist]
    # SPEA2 HV geçmişini hesapla
    s_hv_curve = [hv_from_objs(objs, ref) for objs in s_gen_objs]

    return {
        "nsga2_front": n_front,
        "spea2_front": s_front,
        "ref":         ref,
        "n_hv_curve":  n_hv_curve,
        "s_hv_curve":  s_hv_curve,
    }


def _run_spea2_with_history(user_id, diversity_mode="none"):
    """
    spea2() ile aynı mantık — fakat her nesilde arşiv objectives kaydedilir.
    spea2.py'deki iç fonksiyonları doğrudan kullanır (ikinci çalıştırma yok).
    """
    from spea2 import (Individual, assignment_fitness,
                       environmental_selection, create_offspring,
                       tournament_selection)
    from chromosome import create_individual

    random.seed(SEED)

    population = []
    for _ in range(POP_SIZE):
        chrom, bsize = create_individual(user_id)
        ind = Individual(chrom, bsize, user_id,
                         objective_names=OBJ_NAMES,
                         lambda_penalty=LAMBDA,
                         diversity_mode=diversity_mode)
        ind.evaluate()
        population.append(ind)

    assignment_fitness(population)
    archive = [ind for ind in population if ind.raw_fitness == 0]

    gen_objs = []  # her nesil için arşiv objectives

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

        gen_objs.append([ind.objectives for ind in archive] if archive else [])

    assignment_fitness(population)
    best = [ind for ind in population if ind.raw_fitness == 0]

    return {"best_individuals": best, "population": population}, gen_objs


# ══════════════════════════════════════════════════════════════════════════════
# GRAFİK 1 — PARETO FRONT SCATTER (NSGA-II vs SPEA2, her user için)
# ══════════════════════════════════════════════════════════════════════════════

def plot_pareto_front(user_id, data):
    """
    3 eksen kombinasyonu: Tercih-Maliyet, Tercih-CO2, Maliyet-CO2
    NSGA-II ve SPEA2 aynı eksende farklı renk/marker ile gösterilir.
    """
    nf = data["nsga2_front"]
    sf = data["spea2_front"]

    n_p, n_c, n_co2 = raw(nf,"preference"), raw(nf,"cost"), raw(nf,"co2")
    s_p, s_c, s_co2 = raw(sf,"preference"), raw(sf,"cost"), raw(sf,"co2")

    user_lbl = "User 1 — Et Yiyen" if user_id == 1 else "User 2 — Vejetaryen"
    pairs = [
        (n_p, s_p, n_c,   s_c,   "Tercih Skoru ↑",  "Maliyet (TL) ↓",    "Tercih vs. Maliyet"),
        (n_p, s_p, n_co2, s_co2, "Tercih Skoru ↑",  "CO₂ (kg) ↓",        "Tercih vs. CO₂"),
        (n_c, s_c, n_co2, s_co2, "Maliyet (TL) ↓",  "CO₂ (kg) ↓",        "Maliyet vs. CO₂"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle(f"Pareto Cephesi — {user_lbl}", fontsize=13, fontweight="bold", y=1.02)

    for ax, (nx, sx, ny, sy, xl, yl, title) in zip(axes, pairs):
        ax.scatter(nx, ny, c=COLORS["nsga2"], s=55, alpha=0.85,
                   edgecolors="white", lw=0.6, label="NSGA-II", zorder=4)
        ax.scatter(sx, sy, c=COLORS["spea2"], s=55, alpha=0.85,
                   edgecolors="white", lw=0.6, marker="^", label="SPEA2", zorder=4)
        ax.set_xlabel(xl, fontsize=9.5)
        ax.set_ylabel(yl, fontsize=9.5)
        ax.set_title(title, fontsize=10.5, fontweight="semibold", pad=5)
        ax.legend(fontsize=9, framealpha=0.85)
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.spines[["top","right"]].set_visible(False)
        ax.tick_params(labelsize=8.5)

    plt.tight_layout()
    path = f"{OUT_DIR}/pareto_user{user_id}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# GRAFİK 2 — CONVERGENCE (Hypervolume vs. Nesil)
# ══════════════════════════════════════════════════════════════════════════════

def plot_convergence(user_id, data):
    """Her nesildeki Pareto front / arşiv üzerinden hesaplanan hypervolume."""
    n_hv = data["n_hv_curve"]
    s_hv = data["s_hv_curve"]
    gens = list(range(1, GENS + 1))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    user_lbl = "User 1 — Et Yiyen" if user_id == 1 else "User 2 — Vejetaryen"
    ax.set_title(f"Yakınsama Eğrisi — {user_lbl}", fontsize=12, fontweight="bold", pad=8)

    ax.fill_between(gens[:len(n_hv)], [v*0.97 for v in n_hv], [v*1.03 for v in n_hv],
                    alpha=0.1, color=COLORS["nsga2"])
    ax.fill_between(gens[:len(s_hv)], [v*0.97 for v in s_hv], [v*1.03 for v in s_hv],
                    alpha=0.1, color=COLORS["spea2"])
    ax.plot(gens[:len(n_hv)], n_hv, color=COLORS["nsga2"], lw=2.2,
            marker="o", markersize=3.5, markevery=5, label="NSGA-II")
    ax.plot(gens[:len(s_hv)], s_hv, color=COLORS["spea2"], lw=2.2,
            marker="^", markersize=3.5, markevery=5, linestyle="--", label="SPEA2")

    ax.set_xlabel("Nesil", fontsize=10)
    ax.set_ylabel("Hipervolüm (Monte Carlo)", fontsize=10)
    ax.legend(fontsize=10, framealpha=0.85)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.spines[["top","right"]].set_visible(False)

    ref = data["ref"]
    ax.text(0.02, 0.04, f"Ref. nokta: ({', '.join(f'{x:.1f}' for x in ref)})",
            transform=ax.transAxes, fontsize=7.5, color="#777")
    plt.tight_layout()

    path = f"{OUT_DIR}/convergence_user{user_id}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# GRAFİK 3 — ÖRNEK MENÜ TABLOSU (3 Pareto çözümü)
# ══════════════════════════════════════════════════════════════════════════════

def _pick_3(front):
    """En düşük maliyet, en yüksek tercih, ortadaki (median maliyet)."""
    if len(front) <= 3:
        return front[:3]
    by_cost = sorted(front, key=lambda x: x.fitness_result["cost"])
    lo  = by_cost[0]
    hi  = max(front, key=lambda x: x.fitness_result["preference"])
    mid = by_cost[len(by_cost) // 2]
    chosen, seen = [], set()
    for s in [lo, mid, hi]:
        if id(s) not in seen:
            seen.add(id(s))
            chosen.append(s)
    for s in by_cost:
        if len(chosen) >= 3:
            break
        if id(s) not in seen:
            seen.add(id(s))
            chosen.append(s)
    return chosen[:3]


def plot_sample_menu(user_id, front, algo_name="NSGA-II"):
    """
    Her sütun bir Pareto çözümü:
      - Üst: Özet (tercih/maliyet/co2/uyum)
      - Orta: Besin uyum çubuk grafiği (DRI sınırları işaretli)
      - Alt sol: Kahvaltı listesi
      - Alt sağ: Öğle+Akşam listesi
    """
    if not front:
        print(f"  [SKIP] {algo_name} user{user_id}: boş front")
        return None

    foods_by_id, _ = load_food_data(user_id)
    solutions  = _pick_3(front)
    sol_labels = ["En Düşük Maliyet", "Dengeli", "En Yüksek Tercih"]
    hdr_colors = ["#1B4F72", "#1A5276", "#154360"]

    nut_keys   = ["energy","protein","carbohydrate","fiber","sodium"]
    nut_labels = ["Enerji\n(kcal)","Protein\n(g)","Karbonhidrat\n(g)","Lif\n(g)","Sodyum\n(mg)"]

    user_lbl = "User 1 — Et Yiyen" if user_id == 1 else "User 2 — Vejetaryen"
    fig = plt.figure(figsize=(20, 13))
    fig.patch.set_facecolor("#FAFBFC")
    fig.suptitle(f"Örnek Pareto Çözümleri — {algo_name}  |  {user_lbl}",
                 fontsize=14, fontweight="bold", y=0.99, color="#1a1a2e")

    outer = gridspec.GridSpec(1, 3, figure=fig, wspace=0.05,
                              left=0.02, right=0.98, top=0.94, bottom=0.02)

    for col, (sol, lbl, hcol) in enumerate(zip(solutions, sol_labels, hdr_colors)):
        fr     = sol.fitness_result
        totals = fr["totals"]
        dri    = fr["dri"]
        comply = calculate_constraint_compliance(totals, dri)
        b_ids  = fr["breakfast"]
        m_ids  = fr["main"]

        inner = gridspec.GridSpecFromSubplotSpec(
            4, 1, subplot_spec=outer[col],
            height_ratios=[0.55, 1.7, 2.2, 2.5], hspace=0.08
        )

        # ── Başlık satırı ──────────────────────────────────────────────────
        ax0 = fig.add_subplot(inner[0])
        ax0.set_facecolor(hcol)
        ax0.axis("off")
        ax0.text(0.5, 0.70, lbl, ha="center", va="center",
                 fontsize=12, fontweight="bold", color="white", transform=ax0.transAxes)
        ax0.text(0.5, 0.22,
                 f"Tercih: {fr['preference']:.1f}   Maliyet: {fr['cost']:.2f} TL   "
                 f"CO2: {fr['co2']:.2f} kg   Uyum: {comply}/5",
                 ha="center", va="center", fontsize=8.5, color="#D6EAF8",
                 transform=ax0.transAxes)

        # ── Besin çubuk grafiği ────────────────────────────────────────────
        ax1 = fig.add_subplot(inner[1])
        ax1.set_facecolor("white")
        x    = np.arange(5)
        vals = [totals.get(k, 0) for k in nut_keys]
        rll  = [dri.get(k,{}).get("RLL", 1) for k in nut_keys]
        rul  = [dri.get(k,{}).get("RUL", 1) for k in nut_keys]
        mid  = [(lo+hi)/2 for lo,hi in zip(rll,rul)]
        ratios = [v/m if m > 0 else 0 for v,m in zip(vals,mid)]

        bar_clrs = []
        for v,lo,hi in zip(vals,rll,rul):
            if lo <= v <= hi:   bar_clrs.append(COLORS["green"])
            elif v < lo:        bar_clrs.append(COLORS["orange"])
            else:               bar_clrs.append(COLORS["red"])

        bars = ax1.bar(x, ratios, color=bar_clrs, width=0.52, zorder=3,
                       edgecolor="white", lw=0.8)
        ax1.axhline(1.0, color="#555", lw=1.3, ls="--", alpha=0.6, zorder=4,
                    label="DRI orta noktası")

        # RLL ve RUL çizgileri
        rll_r = [lo/m if m>0 else 0 for lo,m in zip(rll,mid)]
        rul_r = [hi/m if m>0 else 0 for hi,m in zip(rul,mid)]
        for i,(lo_r,hi_r) in enumerate(zip(rll_r,rul_r)):
            ax1.plot([i-0.29,i+0.29],[lo_r,lo_r], color="#2E86C1", lw=1.3, zorder=5)
            ax1.plot([i-0.29,i+0.29],[hi_r,hi_r], color="#C0392B", lw=1.3, zorder=5, ls=":")

        for bar, v in zip(bars, vals):
            ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.03,
                     f"{v:.0f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

        ax1.set_xticks(x)
        ax1.set_xticklabels(nut_labels, fontsize=7.8)
        ax1.set_ylabel("Gerçek / Orta Nokta", fontsize=8)
        ax1.set_ylim(0, max(ratios)*1.25+0.1)
        ax1.set_title("Besin Uyumu (DRI)", fontsize=9, fontweight="semibold", pad=3)
        ax1.grid(axis="y", alpha=0.25, linestyle="--")
        ax1.spines[["top","right"]].set_visible(False)
        patches = [
            mpatches.Patch(color=COLORS["green"],  label="DRI icinde"),
            mpatches.Patch(color=COLORS["orange"], label="Yetersiz"),
            mpatches.Patch(color=COLORS["red"],    label="Fazla"),
        ]
        ax1.legend(handles=patches, fontsize=7, loc="upper right", framealpha=0.8)

        # ── Kahvaltı listesi ───────────────────────────────────────────────
        ax2 = fig.add_subplot(inner[2])
        ax2.set_facecolor("#EBF5FB")
        ax2.axis("off")
        ax2.text(0.5, 0.97, "KAHVALTI", ha="center", va="top",
                 fontsize=10, fontweight="bold", color="#154360",
                 transform=ax2.transAxes)

        b_names = [foods_by_id[fid]["name"][:28] if fid in foods_by_id
                   else f"ID:{fid}" for fid in b_ids]
        col_xs = [0.03, 0.53]
        for i, name in enumerate(b_names[:10]):
            row, c = divmod(i, 2)
            y_pos = 0.86 - row * 0.175
            ax2.text(col_xs[c], y_pos, f"  {name}", va="top",
                     fontsize=8.2, color="#1A3E5A", transform=ax2.transAxes)
        if len(b_names) > 10:
            ax2.text(0.5, 0.86-5*0.175, f"(+{len(b_names)-10} daha)",
                     ha="center", va="top", fontsize=7.5, color="#888",
                     transform=ax2.transAxes)

        # ── Öğle+Akşam listesi ─────────────────────────────────────────────
        ax3 = fig.add_subplot(inner[3])
        ax3.set_facecolor("#FEF9E7")
        ax3.axis("off")
        ax3.text(0.5, 0.97, "OGLE + AKSAM YEMEGI", ha="center", va="top",
                 fontsize=10, fontweight="bold", color="#784212",
                 transform=ax3.transAxes)

        m_names = [foods_by_id[fid]["name"][:28] if fid in foods_by_id
                   else f"ID:{fid}" for fid in m_ids]
        for i, name in enumerate(m_names[:10]):
            row, c = divmod(i, 2)
            y_pos = 0.86 - row * 0.155
            ax3.text(col_xs[c], y_pos, f"  {name}", va="top",
                     fontsize=8.2, color="#5D4037", transform=ax3.transAxes)
        if len(m_names) > 10:
            ax3.text(0.5, 0.86-5*0.155, f"(+{len(m_names)-10} daha)",
                     ha="center", va="top", fontsize=7.5, color="#888",
                     transform=ax3.transAxes)

    plt.savefig(f"{OUT_DIR}/menu_{algo_name.lower().replace('-','')}_user{user_id}.png",
                dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    path = f"{OUT_DIR}/menu_{algo_name.lower().replace('-','')}_user{user_id}.png"
    print(f"  [OK] {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# GRAFİK 4 — USER KARŞILAŞTIRMA (Deney 1)
# ══════════════════════════════════════════════════════════════════════════════

def plot_user_comparison(data_u1, data_u2):
    """
    User 1 vs User 2 — her iki algoritmanın Pareto frontları
    2 satır: üst=NSGA-II, alt=SPEA2
    3 sütun: Tercih-Maliyet, Tercih-CO2, Maliyet-CO2
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Kullanici Karsilastirmasi\n"
                 "Ust: NSGA-II  |  Alt: SPEA2",
                 fontsize=13, fontweight="bold", y=1.01)

    rows_data = [
        (data_u1["nsga2_front"], data_u2["nsga2_front"], "NSGA-II"),
        (data_u1["spea2_front"], data_u2["spea2_front"], "SPEA2"),
    ]
    pairs = [
        ("preference","cost",  "Tercih Skoru","Maliyet (TL)"),
        ("preference","co2",   "Tercih Skoru","CO2 (kg)"),
        ("cost",      "co2",   "Maliyet (TL)","CO2 (kg)"),
    ]

    for row_i, (f1, f2, algo_lbl) in enumerate(rows_data):
        for col_i, (xk, yk, xl, yl) in enumerate(pairs):
            ax = axes[row_i][col_i]
            ax.scatter(raw(f1,xk), raw(f1,yk), c=COLORS["user1"], s=50, alpha=0.8,
                       edgecolors="white", lw=0.5, label="User 1 (Et Yiyen)", zorder=4)
            ax.scatter(raw(f2,xk), raw(f2,yk), c=COLORS["user2"], s=50, alpha=0.8,
                       edgecolors="white", lw=0.5, marker="^",
                       label="User 2 (Vejetaryen)", zorder=4)
            ax.set_xlabel(xl, fontsize=8.5)
            ax.set_ylabel(yl, fontsize=8.5)
            ax.set_title(f"{algo_lbl}", fontsize=9.5, fontweight="semibold")
            ax.grid(True, alpha=0.2, linestyle="--")
            ax.spines[["top","right"]].set_visible(False)
            if col_i == 0:
                ax.legend(fontsize=8, framealpha=0.85)
            ax.tick_params(labelsize=8)

    plt.tight_layout()
    path = f"{OUT_DIR}/user_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# GRAFİK 5 — HYPERVOLUme BAR + ÖZET TABLO (Deney 2)
# ══════════════════════════════════════════════════════════════════════════════

def plot_hv_bar(data_u1, data_u2):
    """Her kullanıcı ve algoritma için hypervolume bar grafiği."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("Hipervoluem Karsilastirmasi (Monte Carlo, 15 000 ornek)",
                 fontsize=12, fontweight="bold")

    for ax, (data, user_id) in zip(axes, [(data_u1,1),(data_u2,2)]):
        nf  = data["nsga2_front"]
        sf  = data["spea2_front"]
        ref = data["ref"]
        n_hv = hv_from_objs([i.objectives for i in nf], ref)
        s_hv = hv_from_objs([i.objectives for i in sf], ref)

        bars = ax.bar(["NSGA-II","SPEA2"], [n_hv, s_hv],
                      color=[COLORS["nsga2"], COLORS["spea2"]],
                      width=0.4, edgecolor="white", lw=1.2, zorder=3)
        for bar, v in zip(bars, [n_hv, s_hv]):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(n_hv,s_hv)*0.01,
                    f"{v:,.0f}", ha="center", va="bottom",
                    fontsize=11, fontweight="bold")

        user_lbl = "User 1 (Et Yiyen)" if user_id==1 else "User 2 (Vejetaryen)"
        winner   = "NSGA-II" if n_hv >= s_hv else "SPEA2"
        ax.set_title(f"{user_lbl}\nKazanan: {winner}", fontsize=10.5, fontweight="semibold")
        ax.set_ylabel("Hipervoluem", fontsize=9.5)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.spines[["top","right"]].set_visible(False)
        ax.set_ylim(0, max(n_hv,s_hv)*1.18)

        ref_txt = "Ref: (" + ", ".join(f"{x:.1f}" for x in ref) + ")"
        ax.text(0.02, 0.97, ref_txt, transform=ax.transAxes,
                fontsize=7.5, color="#777", va="top")

    plt.tight_layout()
    path = f"{OUT_DIR}/hv_bar.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# GRAFİK 6 — DİVERSİTY ETKİSİ (Deney 3)
# ══════════════════════════════════════════════════════════════════════════════

def plot_diversity_impact(user_id):
    """
    diversity_mode: none, penalty, hard
    Her mod için NSGA-II Pareto frontunu çalıştır,
    ortalama food group sayısını ve HV değerini karşılaştır.
    """
    modes      = ["none", "penalty", "hard"]
    mode_lbls  = ["Yok (none)", "Ceza (penalty)", "Sert (hard)"]
    results    = {}

    print(f"\n  [Diversity Impact] User {user_id} icin 3 mod calistiriliyor...")
    for mode in modes:
        res = nsga2(
            user_id=user_id,
            population_size=30,   # hızlı test için
            generations=30,
            objective_names=OBJ_NAMES,
            lambda_penalty=LAMBDA,
            diversity_mode=mode,
            random_seed=SEED,
        )
        front = deduplicate(res["best_front"])
        results[mode] = front

    # Ortak referans noktası
    all_o = [i.objectives for mode in modes for i in results[mode]]
    ref   = get_reference_point(all_o, margin=0.10)

    # Metrikler
    avg_div  = {m: (sum(i.fitness_result["diversity_count"] for i in results[m]) /
                    max(len(results[m]),1)) for m in modes}
    avg_pref = {m: (sum(i.fitness_result["preference"] for i in results[m]) /
                    max(len(results[m]),1)) for m in modes}
    hvs      = {m: hv_from_objs([i.objectives for i in results[m]], ref) for m in modes}
    n_sols   = {m: len(results[m]) for m in modes}

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    user_lbl  = "User 1 (Et Yiyen)" if user_id==1 else "User 2 (Vejetaryen)"
    fig.suptitle(f"Cesitlilik Mekanizmasi Karsilastirmasi — {user_lbl}",
                 fontsize=12, fontweight="bold")

    bar_clrs = ["#5D6D7E", "#1B4F72", "#2E86C1"]

    # Sol: Ortalama food group sayısı
    ax = axes[0]
    vals = [avg_div[m] for m in modes]
    bars = ax.bar(mode_lbls, vals, color=bar_clrs, width=0.45,
                  edgecolor="white", lw=1, zorder=3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                f"{v:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title("Ort. Yemek Grubu Sayisi", fontsize=10.5, fontweight="semibold")
    ax.set_ylabel("Grup sayisi", fontsize=9)
    ax.axhline(4, color="#E74C3C", lw=1.2, ls="--", alpha=0.7, label="Hedef: 4")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines[["top","right"]].set_visible(False)

    # Orta: Hypervolume
    ax = axes[1]
    vals = [hvs[m] for m in modes]
    bars = ax.bar(mode_lbls, vals, color=bar_clrs, width=0.45,
                  edgecolor="white", lw=1, zorder=3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(vals)*0.01,
                f"{v:,.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title("Hipervoluem", fontsize=10.5, fontweight="semibold")
    ax.set_ylabel("Deger", fontsize=9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines[["top","right"]].set_visible(False)

    # Sağ: Ortalama tercih
    ax = axes[2]
    vals = [avg_pref[m] for m in modes]
    bars = ax.bar(mode_lbls, vals, color=bar_clrs, width=0.45,
                  edgecolor="white", lw=1, zorder=3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(vals)*0.01,
                f"{v:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title("Ort. Tercih Skoru", fontsize=10.5, fontweight="semibold")
    ax.set_ylabel("Skor", fontsize=9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    path = f"{OUT_DIR}/diversity_impact_user{user_id}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    all_paths = []
    all_data  = {}

    for user_id in [1, 2]:
        label = "Et Yiyen" if user_id == 1 else "Vejetaryen"
        print(f"\n{'='*55}")
        print(f"  USER {user_id} ({label})")
        print(f"{'='*55}")

        print("  Algoritmalar calistiriliyor (pop=50, gen=50)...")
        data = run_and_collect(user_id)
        all_data[user_id] = data

        print("\n  Grafikler uretiliyor...")

        # 1. Pareto front (NSGA-II vs SPEA2 overlay)
        all_paths.append(plot_pareto_front(user_id, data))

        # 2. Convergence
        all_paths.append(plot_convergence(user_id, data))

        # 3. Menü tabloları (her iki algoritma için)
        all_paths.append(plot_sample_menu(user_id, data["nsga2_front"], "NSGA-II"))
        all_paths.append(plot_sample_menu(user_id, data["spea2_front"],  "SPEA2"))

        # 6. Diversity impact (Deney 3)
        all_paths.append(plot_diversity_impact(user_id))

    # 4. User karşılaştırma (Deney 1)
    print("\n  Kullanici karsilastirmasi...")
    all_paths.append(plot_user_comparison(all_data[1], all_data[2]))

    # 5. Hypervolume bar
    all_paths.append(plot_hv_bar(all_data[1], all_data[2]))

    print(f"\n{'='*55}")
    print(f"  Toplam {len(all_paths)} grafik -> {OUT_DIR}/")
    print(f"{'='*55}")
    return all_paths


if __name__ == "__main__":
    main()