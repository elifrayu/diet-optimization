"""
visualizations.py
Multi-Objective Diet Optimization — Figures for Report
"""

import os
import sys
import random
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nsga2 import nsga2
from spea2 import spea2
from metrics import (
    approximate_hypervolume,
    get_reference_point,
    calculate_constraint_compliance,
)
from load_data import load_food_data


POP_SIZE = 50
GENS = 50
LAMBDA = 10.0
SEED = 42
OBJ_NAMES = ("preference", "cost", "co2")
HV_SAMPLES = 15000

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

COLORS = {
    "nsga2": "#1B4F72",
    "spea2": "#C0392B",
    "user1": "#1B4F72",
    "user2": "#C0392B",
    "green": "#27AE60",
    "orange": "#E67E22",
    "red": "#E74C3C",
    "gray": "#7F8C8D",
    "light": "#F8F9F9",
}


def deduplicate(individuals):
    seen = set()
    unique = []

    for ind in individuals:
        fr = ind.fitness_result
        key = (
            tuple(sorted(fr["menu"])),
            round(fr["preference"], 3),
            round(fr["cost"], 3),
            round(fr["co2"], 3),
        )

        if key not in seen:
            seen.add(key)
            unique.append(ind)

    return unique


def raw(front, key):
    return [ind.fitness_result[key] for ind in front]


def hv_from_objs(objs, ref):
    if not objs:
        return 0.0
    return approximate_hypervolume(objs, ref, samples=HV_SAMPLES, seed=SEED)


def shorten(text, width=32):
    if text is None:
        return ""
    return textwrap.shorten(str(text), width=width, placeholder="...")


def run_and_collect(user_id, diversity_mode="none"):
    random.seed(SEED)

    nsga2_result = nsga2(
        user_id=user_id,
        population_size=POP_SIZE,
        generations=GENS,
        objective_names=OBJ_NAMES,
        lambda_penalty=LAMBDA,
        diversity_mode=diversity_mode,
        random_seed=SEED,
    )

    nsga2_front = deduplicate(nsga2_result["best_front"])

    nsga2_gen_objs = []
    for h in nsga2_result["history"]:
        nsga2_gen_objs.append(h["first_front_objectives"])

    spea2_result, spea2_gen_objs = run_spea2_with_history(user_id, diversity_mode)
    spea2_front = deduplicate(spea2_result["best_individuals"])

    all_objs = [ind.objectives for ind in nsga2_front] + [ind.objectives for ind in spea2_front]
    ref = get_reference_point(all_objs, margin=0.10)

    nsga2_hv_curve = [hv_from_objs(objs, ref) for objs in nsga2_gen_objs]
    spea2_hv_curve = [hv_from_objs(objs, ref) for objs in spea2_gen_objs]

    return {
        "nsga2_front": nsga2_front,
        "spea2_front": spea2_front,
        "ref": ref,
        "n_hv_curve": nsga2_hv_curve,
        "s_hv_curve": spea2_hv_curve,
    }


def run_spea2_with_history(user_id, diversity_mode="none"):
    from spea2 import (
        Individual,
        assignment_fitness,
        environmental_selection,
        create_offspring,
        tournament_selection,
    )
    from chromosome import create_individual

    random.seed(SEED)

    population = []

    for _ in range(POP_SIZE):
        chrom, bsize = create_individual(user_id)
        ind = Individual(
            chrom,
            bsize,
            user_id,
            objective_names=OBJ_NAMES,
            lambda_penalty=LAMBDA,
            diversity_mode=diversity_mode,
        )
        ind.evaluate()
        population.append(ind)

    assignment_fitness(population)
    archive = [ind for ind in population if ind.raw_fitness == 0]

    gen_objs = []

    for _ in range(GENS):
        selection_pool = archive if archive else population

        offspring = []
        while len(offspring) < POP_SIZE:
            parent1 = tournament_selection(selection_pool)
            parent2 = tournament_selection(selection_pool)
            child = create_offspring(parent1, parent2)
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


def plot_pareto_front(user_id, data):
    nsga_front = data["nsga2_front"]
    spea_front = data["spea2_front"]

    user_label = "User 1 - Non-vegetarian" if user_id == 1 else "User 2 - Vegetarian"

    pairs = [
        ("preference", "cost", "Preference Score ↑", "Cost ↓", "Preference vs Cost"),
        ("preference", "co2", "Preference Score ↑", "CO2 ↓", "Preference vs CO2"),
        ("cost", "co2", "Cost ↓", "CO2 ↓", "Cost vs CO2"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    fig.suptitle(f"Pareto Front Comparison - {user_label}", fontsize=14, fontweight="bold")

    for ax, (x_key, y_key, x_label, y_label, title) in zip(axes, pairs):
        ax.scatter(
            raw(nsga_front, x_key),
            raw(nsga_front, y_key),
            c=COLORS["nsga2"],
            s=55,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.6,
            label="NSGA-II",
        )

        ax.scatter(
            raw(spea_front, x_key),
            raw(spea_front, y_key),
            c=COLORS["spea2"],
            s=55,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.6,
            marker="^",
            label="SPEA2",
        )

        ax.set_title(title, fontsize=11, fontweight="semibold")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.legend(fontsize=9)

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    plt.tight_layout()
    path = f"{OUT_DIR}/pareto_user{user_id}.png"
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"[OK] {path}")
    return path


def plot_convergence(user_id, data):
    nsga_hv = data["n_hv_curve"]
    spea_hv = data["s_hv_curve"]

    gens = list(range(1, GENS + 1))
    user_label = "User 1 - Non-vegetarian" if user_id == 1 else "User 2 - Vegetarian"

    fig, ax = plt.subplots(figsize=(9, 4.8))

    ax.plot(
        gens[:len(nsga_hv)],
        nsga_hv,
        color=COLORS["nsga2"],
        linewidth=2.2,
        marker="o",
        markersize=3.5,
        markevery=5,
        label="NSGA-II",
    )

    ax.plot(
        gens[:len(spea_hv)],
        spea_hv,
        color=COLORS["spea2"],
        linewidth=2.2,
        marker="^",
        markersize=3.5,
        markevery=5,
        linestyle="--",
        label="SPEA2",
    )

    ax.set_title(f"Convergence Curve - {user_label}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Hypervolume")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.legend()

    ref_text = "Reference point: (" + ", ".join(f"{x:.1f}" for x in data["ref"]) + ")"
    ax.text(0.02, 0.04, ref_text, transform=ax.transAxes, fontsize=8, color="#666666")

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    path = f"{OUT_DIR}/convergence_user{user_id}.png"
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"[OK] {path}")
    return path


def pick_three_solutions(front):
    if len(front) <= 3:
        return front[:3]

    by_cost = sorted(front, key=lambda x: x.fitness_result["cost"])
    low_cost = by_cost[0]
    balanced = by_cost[len(by_cost) // 2]
    high_preference = max(front, key=lambda x: x.fitness_result["preference"])

    chosen = []
    seen = set()

    for item in [low_cost, balanced, high_preference]:
        if id(item) not in seen:
            seen.add(id(item))
            chosen.append(item)

    for item in by_cost:
        if len(chosen) >= 3:
            break
        if id(item) not in seen:
            seen.add(id(item))
            chosen.append(item)

    return chosen[:3]


def nutrient_status(value, rll, rul):
    if rll <= value <= rul:
        return COLORS["green"]
    if value < rll:
        return COLORS["orange"]
    return COLORS["red"]


def draw_menu_list(ax, title, food_ids, foods_by_id, max_items=16):
    """
    Draws menu items in two columns.
    It does not use '+ more' unless the list is extremely long.
    """
    ax.axis("off")
    ax.set_title(title, fontsize=10, fontweight="bold", pad=4)

    if not food_ids:
        ax.text(0.02, 0.90, "No item selected", fontsize=8, color="#666666")
        return

    items = food_ids[:max_items]

    left_items = items[:8]
    right_items = items[8:16]

    col_x = [0.02, 0.52]
    start_y = 0.92
    step = 0.105

    for col, group in enumerate([left_items, right_items]):
        y = start_y

        for idx, fid in enumerate(group, start=1 + col * 8):
            name = foods_by_id[fid]["name"] if fid in foods_by_id else f"ID: {fid}"

            ax.text(
                col_x[col],
                y,
                f"{idx}. {shorten(name, 30)}",
                fontsize=7.9,
                va="top",
                color="#2C3E50",
            )

            y -= step

    if len(food_ids) > max_items:
        ax.text(
            0.52,
            0.02,
            f"More items not shown: {len(food_ids) - max_items}",
            fontsize=7.2,
            color="#777777",
            va="bottom",
        )


def plot_sample_menu(user_id, front, algo_name="NSGA-II"):
    if not front:
        print(f"[SKIP] Empty front for {algo_name} User {user_id}")
        return None

    foods_by_id, _ = load_food_data(user_id)
    solutions = pick_three_solutions(front)

    titles = ["Low Cost Solution", "Balanced Solution", "High Preference Solution"]
    user_label = "User 1 - Non-vegetarian" if user_id == 1 else "User 2 - Vegetarian"

    nutrients = ["energy", "protein", "carbohydrate", "fiber", "sodium"]
    nutrient_labels = ["Energy", "Protein", "Carb", "Fiber", "Sodium"]

    fig = plt.figure(figsize=(17, 10.5))
    fig.patch.set_facecolor("#F7F9F9")

    fig.suptitle(
        f"Sample Pareto Menu Solutions - {algo_name} | {user_label}",
        fontsize=15,
        fontweight="bold",
        y=0.985,
    )

    outer = gridspec.GridSpec(
        1,
        3,
        figure=fig,
        left=0.025,
        right=0.985,
        top=0.91,
        bottom=0.04,
        wspace=0.11,
    )

    for col, (solution, title) in enumerate(zip(solutions, titles)):
        fr = solution.fitness_result
        totals = fr["totals"]
        dri = fr["dri"]

        compliance = calculate_constraint_compliance(totals, dri)

        inner = gridspec.GridSpecFromSubplotSpec(
            4,
            1,
            subplot_spec=outer[col],
            height_ratios=[0.62, 2.00, 1.80, 2.05],
            hspace=0.36,
        )

        ax_header = fig.add_subplot(inner[0])
        ax_header.axis("off")

        header_color = COLORS["nsga2"] if algo_name == "NSGA-II" else COLORS["spea2"]

        ax_header.add_patch(
            plt.Rectangle(
                (0, 0.08),
                1,
                0.88,
                transform=ax_header.transAxes,
                color=header_color,
                alpha=0.95,
                zorder=0,
            )
        )

        ax_header.text(
            0.5,
            0.68,
            title,
            ha="center",
            va="center",
            fontsize=11,
            color="white",
            fontweight="bold",
        )

        ax_header.text(
            0.5,
            0.30,
            f"Preference: {fr['preference']:.1f} | Cost: {fr['cost']:.2f} | "
            f"CO2: {fr['co2']:.2f} | Compliance: {compliance}/5 | Diversity: {fr['diversity_count']}",
            ha="center",
            va="center",
            fontsize=8.1,
            color="white",
        )

        ax_bar = fig.add_subplot(inner[1])

        values = [totals.get(n, 0) for n in nutrients]
        rll = [dri[n]["RLL"] for n in nutrients]
        rul = [dri[n]["RUL"] for n in nutrients]

        midpoint = [(lo + hi) / 2 for lo, hi in zip(rll, rul)]
        ratios = [v / m if m else 0 for v, m in zip(values, midpoint)]

        colors = [nutrient_status(v, lo, hi) for v, lo, hi in zip(values, rll, rul)]

        y_pos = np.arange(len(nutrients))
        ax_bar.barh(
            y_pos,
            ratios,
            color=colors,
            height=0.52,
            edgecolor="white",
            linewidth=0.8,
        )

        ax_bar.axvline(
            1.0,
            color="#555555",
            linestyle="--",
            linewidth=1.1,
            alpha=0.75,
        )

        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels(nutrient_labels, fontsize=8.5)
        ax_bar.invert_yaxis()
        ax_bar.set_xlabel("Actual / DRI midpoint", fontsize=8.5)
        ax_bar.set_title("Nutrient Compliance Chart", fontsize=10, fontweight="semibold", pad=6)

        max_ratio = max(ratios) if ratios else 1
        ax_bar.set_xlim(0, max(1.8, max_ratio * 1.25))

        for i, (ratio, value) in enumerate(zip(ratios, values)):
            ax_bar.text(
                ratio + 0.03,
                i,
                f"{value:.0f}",
                va="center",
                fontsize=8,
                fontweight="bold",
                color="#333333",
            )

        ax_bar.grid(axis="x", alpha=0.22, linestyle="--")

        for spine in ["top", "right"]:
            ax_bar.spines[spine].set_visible(False)

        ax_breakfast = fig.add_subplot(inner[2])
        draw_menu_list(
            ax_breakfast,
            "Breakfast",
            fr["breakfast"],
            foods_by_id,
            max_items=16,
        )

        ax_main = fig.add_subplot(inner[3])
        draw_menu_list(
            ax_main,
            "Lunch + Dinner",
            fr["main"],
            foods_by_id,
            max_items=16,
        )

    path = f"{OUT_DIR}/menu_{algo_name.lower().replace('-', '')}_user{user_id}.png"
    plt.savefig(path, dpi=170, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[OK] {path}")
    return path


def plot_user_comparison(data_u1, data_u2):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.8))
    fig.suptitle(
        "User Comparison - Pareto Fronts\nTop: NSGA-II | Bottom: SPEA2",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    rows = [
        (data_u1["nsga2_front"], data_u2["nsga2_front"], "NSGA-II"),
        (data_u1["spea2_front"], data_u2["spea2_front"], "SPEA2"),
    ]

    pairs = [
        ("preference", "cost", "Preference Score", "Cost"),
        ("preference", "co2", "Preference Score", "CO2"),
        ("cost", "co2", "Cost", "CO2"),
    ]

    for row_i, (front1, front2, algo_name) in enumerate(rows):
        for col_i, (x_key, y_key, x_label, y_label) in enumerate(pairs):
            ax = axes[row_i][col_i]

            ax.scatter(
                raw(front1, x_key),
                raw(front1, y_key),
                c=COLORS["user1"],
                s=50,
                alpha=0.82,
                edgecolors="white",
                linewidths=0.5,
                label="User 1",
            )

            ax.scatter(
                raw(front2, x_key),
                raw(front2, y_key),
                c=COLORS["user2"],
                s=50,
                alpha=0.82,
                edgecolors="white",
                linewidths=0.5,
                marker="^",
                label="User 2",
            )

            ax.set_title(algo_name, fontsize=10, fontweight="semibold")
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.grid(True, alpha=0.23, linestyle="--")

            if col_i == 0:
                ax.legend(fontsize=8)

            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)

    plt.tight_layout()
    path = f"{OUT_DIR}/user_comparison.png"
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"[OK] {path}")
    return path


def plot_hv_bar(data_u1, data_u2):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    fig.suptitle("Hypervolume Comparison", fontsize=14, fontweight="bold")

    for ax, (data, user_id) in zip(axes, [(data_u1, 1), (data_u2, 2)]):
        nsga_front = data["nsga2_front"]
        spea_front = data["spea2_front"]
        ref = data["ref"]

        nsga_hv = hv_from_objs([ind.objectives for ind in nsga_front], ref)
        spea_hv = hv_from_objs([ind.objectives for ind in spea_front], ref)

        values = [nsga_hv, spea_hv]
        labels = ["NSGA-II", "SPEA2"]

        bars = ax.bar(
            labels,
            values,
            color=[COLORS["nsga2"], COLORS["spea2"]],
            width=0.45,
            edgecolor="white",
            linewidth=1.1,
        )

        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.02,
                f"{value:,.0f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        winner = "NSGA-II" if nsga_hv >= spea_hv else "SPEA2"
        user_label = "User 1" if user_id == 1 else "User 2"

        ax.set_title(f"{user_label} | Winner: {winner}", fontsize=11, fontweight="semibold")
        ax.set_ylabel("Hypervolume")
        ax.set_ylim(0, max(values) * 1.20)
        ax.grid(axis="y", alpha=0.25, linestyle="--")

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    plt.tight_layout()
    path = f"{OUT_DIR}/hv_bar.png"
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"[OK] {path}")
    return path


def plot_diversity_impact(user_id):
    modes = ["none", "penalty"]
    mode_labels = ["No Diversity", "Penalty Diversity"]

    results = {}

    print(f"[Diversity Impact] Running User {user_id} for none and penalty modes...")

    for mode in modes:
        result = nsga2(
            user_id=user_id,
            population_size=30,
            generations=30,
            objective_names=OBJ_NAMES,
            lambda_penalty=LAMBDA,
            diversity_mode=mode,
            random_seed=SEED,
        )

        results[mode] = deduplicate(result["best_front"])

    all_objs = [ind.objectives for mode in modes for ind in results[mode]]
    ref = get_reference_point(all_objs, margin=0.10)

    avg_diversity = {
        mode: sum(ind.fitness_result["diversity_count"] for ind in results[mode]) / max(len(results[mode]), 1)
        for mode in modes
    }

    avg_preference = {
        mode: sum(ind.fitness_result["preference"] for ind in results[mode]) / max(len(results[mode]), 1)
        for mode in modes
    }

    avg_compliance = {
        mode: sum(
            calculate_constraint_compliance(
                ind.fitness_result["totals"],
                ind.fitness_result["dri"],
            )
            for ind in results[mode]
        ) / max(len(results[mode]), 1)
        for mode in modes
    }

    hvs = {
        mode: hv_from_objs([ind.objectives for ind in results[mode]], ref)
        for mode in modes
    }

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.6))
    user_label = "User 1 - Non-vegetarian" if user_id == 1 else "User 2 - Vegetarian"

    fig.suptitle(
        f"Diversity Mechanism Impact - {user_label}",
        fontsize=14,
        fontweight="bold",
    )

    metrics = [
        ("Average Diversity", avg_diversity, "Food groups"),
        ("Average Preference", avg_preference, "Score"),
        ("Average Compliance", avg_compliance, "Nutrients / 5"),
        ("Hypervolume", hvs, "HV"),
    ]

    for ax, (title, metric_dict, ylabel) in zip(axes, metrics):
        values = [metric_dict[mode] for mode in modes]

        bars = ax.bar(
            mode_labels,
            values,
            color=[COLORS["gray"], COLORS["nsga2"]],
            width=0.45,
            edgecolor="white",
            linewidth=1.0,
        )

        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.02,
                f"{value:.2f}" if value < 1000 else f"{value:,.0f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

        ax.set_title(title, fontsize=10.5, fontweight="semibold")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25, linestyle="--")

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    plt.tight_layout()
    path = f"{OUT_DIR}/diversity_impact_user{user_id}.png"
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"[OK] {path}")
    return path


def main():
    all_paths = []
    all_data = {}

    for user_id in [1, 2]:
        label = "Non-vegetarian" if user_id == 1 else "Vegetarian"

        print("\n" + "=" * 60)
        print(f"USER {user_id} - {label}")
        print("=" * 60)

        print("Running algorithms...")
        data = run_and_collect(user_id, diversity_mode="none")
        all_data[user_id] = data

        print("Generating figures...")

        all_paths.append(plot_pareto_front(user_id, data))
        all_paths.append(plot_convergence(user_id, data))
        all_paths.append(plot_sample_menu(user_id, data["nsga2_front"], "NSGA-II"))
        all_paths.append(plot_sample_menu(user_id, data["spea2_front"], "SPEA2"))
        all_paths.append(plot_diversity_impact(user_id))

    all_paths.append(plot_user_comparison(all_data[1], all_data[2]))
    all_paths.append(plot_hv_bar(all_data[1], all_data[2]))

    print("\n" + "=" * 60)
    print(f"Generated {len(all_paths)} figures in: {OUT_DIR}")
    print("=" * 60)

    return all_paths


if __name__ == "__main__":
    main()