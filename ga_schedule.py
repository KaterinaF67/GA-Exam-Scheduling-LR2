import csv
from pathlib import Path
import numpy as np

# -------------------- ПАРАМЕТРЫ ЛР --------------------

N_SLOTS = 10
POPULATION_SIZE = 30
EVALUATION_BUDGET = 300

TOURNAMENT_SIZE = 3
CROSSOVER_PROBABILITY = 0.90
MUTATION_PROBABILITY = 0.08
ELITE_COUNT = 2

HARD_PENALTY = 1000.0
BACK_TO_BACK_WEIGHT = 1.0
BALANCE_WEIGHT = 0.5

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


# -------------------- ДАННЫЕ --------------------

def load_data():
    exams = []

    with open(DATA_DIR / "exams.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            exams.append(row["exam_name"])

    student_to_exams = {}

    with open(DATA_DIR / "student_exams.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            student_id = int(row["student_id"])
            exam_id = int(row["exam_id"])

            if student_id not in student_to_exams:
                student_to_exams[student_id] = []

            student_to_exams[student_id].append(exam_id)

    n_exams = len(exams)
    conflict_matrix = np.zeros((n_exams, n_exams), dtype=int)

    for exam_list in student_to_exams.values():
        for i in range(len(exam_list)):
            for j in range(i + 1, len(exam_list)):
                a = exam_list[i]
                b = exam_list[j]

                conflict_matrix[a, b] += 1
                conflict_matrix[b, a] += 1

    return exams, student_to_exams, conflict_matrix


# -------------------- ОЦЕНКА РАСПИСАНИЯ --------------------

def hard_conflicts(schedule, conflict_matrix):
    same_slot = schedule[:, None] == schedule[None, :]
    mask = np.triu(same_slot, k=1)
    return int(np.sum(conflict_matrix[mask]))


def back_to_back_penalty(schedule, conflict_matrix):
    same_day = (schedule[:, None] // 2) == (schedule[None, :] // 2)
    different_slot = schedule[:, None] != schedule[None, :]
    mask = np.triu(same_day & different_slot, k=1)
    return int(np.sum(conflict_matrix[mask]))


def balance_penalty(schedule):
    counts = np.bincount(schedule.astype(int), minlength=N_SLOTS)
    return int(counts.max() - counts.min())


def evaluate_schedule(schedule, conflict_matrix):
    hard = hard_conflicts(schedule, conflict_matrix)

    back_to_back = back_to_back_penalty(
        schedule,
        conflict_matrix,
    )

    balance = balance_penalty(schedule)

    soft = (
        BACK_TO_BACK_WEIGHT * back_to_back
        + BALANCE_WEIGHT * balance
    )

    fitness = HARD_PENALTY * hard + soft

    return float(fitness), int(hard), float(soft)


def evaluate_population(population, conflict_matrix):
    fitness_values = []
    hard_values = []
    soft_values = []

    for schedule in population:
        fitness, hard, soft = evaluate_schedule(
            schedule,
            conflict_matrix,
        )

        fitness_values.append(fitness)
        hard_values.append(hard)
        soft_values.append(soft)

    return (
        np.array(fitness_values, dtype=float),
        np.array(hard_values, dtype=int),
        np.array(soft_values, dtype=float),
    )


# -------------------- ГЕНЕТИЧЕСКИЕ ОПЕРАТОРЫ --------------------

def initialize_population(rng, n_exams):
    return rng.integers(
        0,
        N_SLOTS,
        size=(POPULATION_SIZE, n_exams),
        dtype=int,
    )


def tournament_selection(population, fitness_values, rng, n_parents):
    parents = []

    for _ in range(n_parents):
        candidates = rng.choice(
            len(population),
            size=TOURNAMENT_SIZE,
            replace=False,
        )

        winner_local = np.argmin(fitness_values[candidates])
        winner_index = candidates[winner_local]

        parents.append(population[winner_index].copy())

    return np.array(parents, dtype=int)


def uniform_crossover(parent1, parent2, rng):
    if rng.random() >= CROSSOVER_PROBABILITY:
        return parent1.copy(), parent2.copy()

    mask = rng.random(len(parent1)) < 0.5

    child1 = np.where(mask, parent1, parent2)
    child2 = np.where(mask, parent2, parent1)

    return child1.astype(int), child2.astype(int)


def one_point_crossover(parent1, parent2, rng):
    if rng.random() >= CROSSOVER_PROBABILITY:
        return parent1.copy(), parent2.copy()

    point = int(rng.integers(1, len(parent1)))

    child1 = np.concatenate((
        parent1[:point],
        parent2[point:],
    ))

    child2 = np.concatenate((
        parent2[:point],
        parent1[point:],
    ))

    return child1.astype(int), child2.astype(int)


def crossover(parent1, parent2, rng, crossover_type):
    if crossover_type == "uniform":
        return uniform_crossover(parent1, parent2, rng)

    if crossover_type == "one_point":
        return one_point_crossover(parent1, parent2, rng)

    raise ValueError("Unknown crossover type")


def slot_mutation(children, rng):
    children = children.copy()

    for child_id in range(len(children)):
        for exam_id in range(children.shape[1]):
            if rng.random() < MUTATION_PROBABILITY:
                old_slot = int(children[child_id, exam_id])

                new_slot = int(rng.integers(0, N_SLOTS - 1))

                if new_slot >= old_slot:
                    new_slot += 1

                children[child_id, exam_id] = new_slot

    return children


# -------------------- REPAIR --------------------

def local_cost_for_exam(schedule, exam_id, slot, conflict_matrix):
    shared = conflict_matrix[exam_id]

    hard_mask = schedule == slot
    hard = int(np.sum(shared[hard_mask]))

    same_day_mask = (schedule // 2) == (slot // 2)
    different_slot_mask = schedule != slot
    soft = int(np.sum(shared[same_day_mask & different_slot_mask]))

    return hard, soft


def repair_schedule(schedule, conflict_matrix, rng):
    repaired = schedule.copy()

    for _ in range(3):
        changed = False
        exam_order = rng.permutation(len(repaired))

        for exam_id in exam_order:
            current_slot = int(repaired[exam_id])

            current_hard, _ = local_cost_for_exam(
                repaired,
                exam_id,
                current_slot,
                conflict_matrix,
            )

            if current_hard == 0:
                continue

            best_slots = []
            best_key = None

            for slot in range(N_SLOTS):
                hard, soft = local_cost_for_exam(
                    repaired,
                    exam_id,
                    slot,
                    conflict_matrix,
                )

                key = (hard, soft)

                if best_key is None or key < best_key:
                    best_key = key
                    best_slots = [slot]

                elif key == best_key:
                    best_slots.append(slot)

            new_slot = int(rng.choice(best_slots))

            if new_slot != current_slot:
                repaired[exam_id] = new_slot
                changed = True

        if hard_conflicts(repaired, conflict_matrix) == 0:
            break

        if not changed:
            break

    return repaired


def repair_population(population, conflict_matrix, rng):
    repaired = []

    for schedule in population:
        repaired.append(
            repair_schedule(schedule, conflict_matrix, rng)
        )

    return np.array(repaired, dtype=int)


# -------------------- НОВОЕ ПОКОЛЕНИЕ --------------------

def make_next_population(
    population,
    fitness_values,
    conflict_matrix,
    rng,
    constraint_mode,
    crossover_type,
):
    elite_indices = np.argsort(fitness_values)[:ELITE_COUNT]
    elites = population[elite_indices].copy()

    n_children = POPULATION_SIZE - ELITE_COUNT

    if n_children % 2 == 0:
        n_parents = n_children
    else:
        n_parents = n_children + 1

    parents = tournament_selection(
        population,
        fitness_values,
        rng,
        n_parents,
    )

    children = []

    for i in range(0, n_parents, 2):
        child1, child2 = crossover(
            parents[i],
            parents[i + 1],
            rng,
            crossover_type,
        )

        children.append(child1)
        children.append(child2)

    children = np.array(
        children[:n_children],
        dtype=int,
    )

    children = slot_mutation(children, rng)

    if constraint_mode == "repair":
        children = repair_population(
            children,
            conflict_matrix,
            rng,
        )

    return np.vstack((elites, children))


# -------------------- ЗАПУСК ГА --------------------

def run_ga(
    seed,
    constraint_mode="repair",
    crossover_type="uniform",
):
    if constraint_mode not in ("penalty", "repair"):
        raise ValueError("constraint_mode must be penalty or repair")

    exams, _, conflict_matrix = load_data()
    n_exams = len(exams)

    rng = np.random.default_rng(seed)

    population = initialize_population(rng, n_exams)

    if constraint_mode == "repair":
        population = repair_population(
            population,
            conflict_matrix,
            rng,
        )

    fitness_values, hard_values, soft_values = evaluate_population(
        population,
        conflict_matrix,
    )

    evaluations = POPULATION_SIZE

    best_index = int(np.argmin(fitness_values))

    best_schedule = population[best_index].copy()
    best_fitness = float(fitness_values[best_index])
    best_hard = int(hard_values[best_index])
    best_soft = float(soft_values[best_index])

    history = [{
        "evaluations": evaluations,
        "best_fitness": best_fitness,
        "best_hard": best_hard,
        "best_soft": best_soft,
    }]

    while evaluations + POPULATION_SIZE <= EVALUATION_BUDGET:
        population = make_next_population(
            population,
            fitness_values,
            conflict_matrix,
            rng,
            constraint_mode,
            crossover_type,
        )

        fitness_values, hard_values, soft_values = evaluate_population(
            population,
            conflict_matrix,
        )

        evaluations += POPULATION_SIZE

        index = int(np.argmin(fitness_values))

        if fitness_values[index] < best_fitness:
            best_schedule = population[index].copy()
            best_fitness = float(fitness_values[index])
            best_hard = int(hard_values[index])
            best_soft = float(soft_values[index])

        history.append({
            "evaluations": evaluations,
            "best_fitness": best_fitness,
            "best_hard": best_hard,
            "best_soft": best_soft,
        })

    return {
        "seed": seed,
        "constraint_mode": constraint_mode,
        "crossover_type": crossover_type,
        "best_schedule": best_schedule,
        "best_fitness": best_fitness,
        "best_hard": best_hard,
        "best_soft": best_soft,
        "history": history,
    }


# -------------------- BASELINE --------------------

def random_feasible_schedule(conflict_matrix, rng):
    n_exams = len(conflict_matrix)
    schedule = np.full(n_exams, -1, dtype=int)

    exam_order = rng.permutation(n_exams)

    for exam_id in exam_order:
        best_slots = []
        best_key = None

        for slot in range(N_SLOTS):
            hard, soft = local_cost_for_exam(
                schedule,
                exam_id,
                slot,
                conflict_matrix,
            )

            key = (hard, soft)

            if best_key is None or key < best_key:
                best_key = key
                best_slots = [slot]

            elif key == best_key:
                best_slots.append(slot)

        schedule[exam_id] = int(rng.choice(best_slots))

    return schedule


def run_random_baseline(seed):
    _, _, conflict_matrix = load_data()
    rng = np.random.default_rng(seed)

    best_schedule = None
    best_fitness = float("inf")
    best_hard = None
    best_soft = None

    for _ in range(EVALUATION_BUDGET):
        schedule = random_feasible_schedule(
            conflict_matrix,
            rng,
        )

        fitness, hard, soft = evaluate_schedule(
            schedule,
            conflict_matrix,
        )

        if fitness < best_fitness:
            best_schedule = schedule.copy()
            best_fitness = fitness
            best_hard = hard
            best_soft = soft

    return {
        "seed": seed,
        "best_schedule": best_schedule,
        "best_fitness": float(best_fitness),
        "best_hard": int(best_hard),
        "best_soft": float(best_soft),
    }


def slot_label(slot):
    day = slot // 2 + 1

    if slot % 2 == 0:
        part = "утро"
    else:
        part = "день"

    return f"День {day}, {part}"


if __name__ == "__main__":
    result = run_ga(
        seed=202600,
        constraint_mode="repair",
        crossover_type="uniform",
    )

    print("Best fitness:", result["best_fitness"])
    print("Hard conflicts:", result["best_hard"])
    print("Soft penalty:", result["best_soft"])
    print("Schedule:", result["best_schedule"])
