import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from ga_schedule import (
    POPULATION_SIZE,
    EVALUATION_BUDGET,
    TOURNAMENT_SIZE,
    CROSSOVER_PROBABILITY,
    MUTATION_PROBABILITY,
    ELITE_COUNT,
    N_SLOTS,
    HARD_PENALTY,
    load_data,
    run_ga,
    run_random_baseline,
    evaluate_schedule,
    slot_label,
)

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"

N_RUNS = 20
SEEDS = list(range(202600, 202600 + N_RUNS))


def save_parameters():
    rows = [
        ["parameter", "value"],
        ["n_runs", N_RUNS],
        ["population_size", POPULATION_SIZE],
        ["evaluation_budget", EVALUATION_BUDGET],
        ["n_slots", N_SLOTS],
        ["tournament_size", TOURNAMENT_SIZE],
        ["crossover_probability", CROSSOVER_PROBABILITY],
        ["mutation_probability", MUTATION_PROBABILITY],
        ["elite_count", ELITE_COUNT],
        ["hard_penalty", HARD_PENALTY],
        ["constraint_comparison", "penalty vs repair"],
        ["crossover_comparison", "uniform vs one_point"],
        ["baseline", "random constructive search"],
    ]

    with open(
        RESULTS_DIR / "parameters.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        csv.writer(f).writerows(rows)


def run_method(
    method_name,
    constraint_mode,
    crossover_type,
):
    rows = []
    histories = []

    print("\nMETHOD:", method_name)

    for seed in SEEDS:
        result = run_ga(
            seed=seed,
            constraint_mode=constraint_mode,
            crossover_type=crossover_type,
        )

        rows.append({
            "method": method_name,
            "seed": seed,
            "best_fitness": result["best_fitness"],
            "best_hard": result["best_hard"],
            "best_soft": result["best_soft"],
        })

        histories.append(result["history"])

        print(
            seed,
            "fitness=", round(result["best_fitness"], 3),
            "hard=", result["best_hard"],
            "soft=", round(result["best_soft"], 3),
        )

    return rows, histories


def run_all():
    all_rows = []
    histories = {}

    rows, hist = run_method(
        "penalty_uniform",
        "penalty",
        "uniform",
    )
    all_rows.extend(rows)
    histories["penalty_uniform"] = hist

    rows, hist = run_method(
        "repair_uniform",
        "repair",
        "uniform",
    )
    all_rows.extend(rows)
    histories["repair_uniform"] = hist

    rows, hist = run_method(
        "repair_one_point",
        "repair",
        "one_point",
    )
    all_rows.extend(rows)
    histories["repair_one_point"] = hist

    print("\nMETHOD: random_baseline")

    for seed in SEEDS:
        result = run_random_baseline(seed)

        all_rows.append({
            "method": "random_baseline",
            "seed": seed,
            "best_fitness": result["best_fitness"],
            "best_hard": result["best_hard"],
            "best_soft": result["best_soft"],
        })

        print(
            seed,
            "fitness=", round(result["best_fitness"], 3),
            "hard=", result["best_hard"],
            "soft=", round(result["best_soft"], 3),
        )

    return all_rows, histories


def save_runs(rows):
    with open(
        RESULTS_DIR / "runs.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "seed",
                "best_fitness",
                "best_hard",
                "best_soft",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)


def summary(values):
    values = np.array(values, dtype=float)

    return {
        "best": float(np.min(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "worst": float(np.max(values)),
    }


def save_summary(rows):
    methods = [
        "penalty_uniform",
        "repair_uniform",
        "repair_one_point",
        "random_baseline",
    ]

    output = []

    for method in methods:
        method_rows = [
            row for row in rows
            if row["method"] == method
        ]

        soft_values = [
            row["best_soft"]
            for row in method_rows
        ]

        hard_values = [
            row["best_hard"]
            for row in method_rows
        ]

        stats = summary(soft_values)

        output.append({
            "method": method,
            "best_soft": stats["best"],
            "mean_soft": stats["mean"],
            "median_soft": stats["median"],
            "std_soft": stats["std"],
            "worst_soft": stats["worst"],
            "feasible_runs": sum(
                hard == 0 for hard in hard_values
            ),
            "total_runs": len(method_rows),
        })

    with open(
        RESULTS_DIR / "summary.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "best_soft",
                "mean_soft",
                "median_soft",
                "std_soft",
                "worst_soft",
                "feasible_runs",
                "total_runs",
            ],
        )

        writer.writeheader()
        writer.writerows(output)


def create_examples():
    exams, _, conflict_matrix = load_data()

    feasible_1 = run_ga(9901, constraint_mode="repair", crossover_type="uniform")["best_schedule"]
    feasible_2 = run_ga(9902, constraint_mode="repair", crossover_type="one_point")["best_schedule"]

    examples = [
        ("feasible_1", feasible_1),
        ("feasible_2", feasible_2),
        ("infeasible_1", np.zeros(len(exams), dtype=int)),
        ("infeasible_2", np.arange(len(exams)) % 2),
    ]

    rows = []

    for name, schedule in examples:
        fitness, hard, soft = evaluate_schedule(schedule, conflict_matrix)
        rows.append({
            "example": name,
            "schedule": " ".join(map(str, schedule.tolist())),
            "hard_conflicts": hard,
            "soft_penalty": soft,
            "fitness": fitness,
        })

    with open(RESULTS_DIR / "examples.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "example",
                "schedule",
                "hard_conflicts",
                "soft_penalty",
                "fitness",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def save_best_schedule(rows):
    exams, _, _ = load_data()

    candidates = [
        row for row in rows
        if row["method"] in ("repair_uniform", "repair_one_point")
    ]

    best_row = min(
        candidates,
        key=lambda row: (row["best_hard"], row["best_soft"]),
    )

    if best_row["method"] == "repair_uniform":
        crossover_type = "uniform"
    else:
        crossover_type = "one_point"

    best = run_ga(
        best_row["seed"],
        constraint_mode="repair",
        crossover_type=crossover_type,
    )

    schedule = best["best_schedule"]

    with open(RESULTS_DIR / "best_schedule.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["exam_id", "exam_name", "slot", "slot_label"])

        for exam_id, exam_name in enumerate(exams):
            slot = int(schedule[exam_id])
            writer.writerow([exam_id, exam_name, slot, slot_label(slot)])

    with open(RESULTS_DIR / "best_schedule.txt", "w", encoding="utf-8") as f:
        f.write(
            f"seed={best['seed']}\n"
            f"crossover={best['crossover_type']}\n"
            f"hard_conflicts={best['best_hard']}\n"
            f"soft_penalty={best['best_soft']}\n\n"
        )

        for slot in range(N_SLOTS):
            f.write(slot_label(slot) + "\n")
            for exam_id, exam_name in enumerate(exams):
                if int(schedule[exam_id]) == slot:
                    f.write(f"  - {exam_name}\n")
            f.write("\n")


def plot_boxplot(rows, methods, labels, filename, title):
    data = []

    for method in methods:
        values = [
            row["best_soft"]
            for row in rows
            if row["method"] == method
            and row["best_hard"] == 0
        ]

        data.append(values)

    plt.figure(figsize=(8, 5))
    plt.boxplot(data, tick_labels=labels)
    plt.ylabel("Мягкий штраф")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=160)
    plt.close()


def plot_feasible_runs(rows):
    methods = [
        "penalty_uniform",
        "repair_uniform",
        "repair_one_point",
        "random_baseline",
    ]

    labels = [
        "Penalty",
        "Repair + uniform",
        "Repair + one-point",
        "Baseline",
    ]

    counts = []

    for method in methods:
        counts.append(sum(
            row["best_hard"] == 0
            for row in rows
            if row["method"] == method
        ))

    plt.figure(figsize=(9, 5))
    plt.bar(labels, counts)
    plt.ylabel("Допустимых запусков из 20")
    plt.title("Надёжность получения допустимого расписания")
    plt.ylim(0, 20)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "feasible_runs.png",
        dpi=160,
    )
    plt.close()


def plot_convergence(histories):
    plt.figure(figsize=(9, 5))

    for method, label in [
        ("repair_uniform", "Repair + uniform"),
        ("repair_one_point", "Repair + one-point"),
    ]:
        all_values = []

        for history in histories[method]:
            values = []

            for row in history:
                if row["best_hard"] == 0:
                    values.append(row["best_soft"])
                else:
                    values.append(row["best_fitness"])

            all_values.append(values)

        matrix = np.array(all_values, dtype=float)
        mean_values = np.mean(matrix, axis=0)

        evaluations = [
            row["evaluations"]
            for row in histories[method][0]
        ]

        plt.plot(
            evaluations,
            mean_values,
            label=label,
        )

    plt.xlabel("Число вычислений целевой функции")
    plt.ylabel("Среднее лучшее значение")
    plt.title("Сходимость по 20 независимым запускам")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "convergence.png",
        dpi=160,
    )
    plt.close()


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    save_parameters()
    create_examples()

    rows, histories = run_all()

    save_runs(rows)
    save_summary(rows)
    save_best_schedule(rows)

    plot_boxplot(
        rows,
        ["penalty_uniform", "repair_uniform"],
        ["Penalty", "Repair"],
        "constraint_comparison.png",
        "Сравнение способов обработки ограничений",
    )

    plot_boxplot(
        rows,
        ["repair_uniform", "repair_one_point"],
        ["Uniform", "One-point"],
        "crossover_comparison.png",
        "Сравнение двух операторов кроссовера",
    )

    plot_boxplot(
        rows,
        ["repair_uniform", "random_baseline"],
        ["ГА + repair", "Baseline"],
        "ga_vs_baseline.png",
        "ГА и случайный конструктивный поиск",
    )

    plot_feasible_runs(rows)
    plot_convergence(histories)

    print("\nAll results saved.")


if __name__ == "__main__":
    main()
