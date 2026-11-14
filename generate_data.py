import csv
from pathlib import Path
import numpy as np

DATA_SEED = 202619
N_EXAMS = 24
N_STUDENTS = 160
N_SLOTS = 10
MIN_EXAMS_PER_STUDENT = 3
MAX_EXAMS_PER_STUDENT = 5

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


def main():
    DATA_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(DATA_SEED)

    exam_names = [f"Exam_{i+1:02d}" for i in range(N_EXAMS)]
    hidden_slots = np.arange(N_EXAMS) % N_SLOTS
    rng.shuffle(hidden_slots)

    with open(DATA_DIR / "exams.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["exam_id", "exam_name"])
        for exam_id, exam_name in enumerate(exam_names):
            writer.writerow([exam_id, exam_name])

    rows = []

    for student_id in range(N_STUDENTS):
        count = int(rng.integers(MIN_EXAMS_PER_STUDENT, MAX_EXAMS_PER_STUDENT + 1))

        available_slots = list(range(N_SLOTS))
        rng.shuffle(available_slots)
        chosen_slots = available_slots[:count]

        chosen_exams = []
        for slot in chosen_slots:
            candidates = np.where(hidden_slots == slot)[0]
            exam_id = int(rng.choice(candidates))
            chosen_exams.append(exam_id)

        for exam_id in sorted(set(chosen_exams)):
            rows.append([student_id, exam_id])

    with open(DATA_DIR / "student_exams.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["student_id", "exam_id"])
        writer.writerows(rows)

    with open(DATA_DIR / "generation_info.txt", "w", encoding="utf-8") as f:
        f.write(f"DATA_SEED={DATA_SEED}\n")
        f.write(f"N_EXAMS={N_EXAMS}\n")
        f.write(f"N_STUDENTS={N_STUDENTS}\n")
        f.write(f"N_SLOTS={N_SLOTS}\n")
        f.write("Data are synthetic and reproducible.\n")
        f.write("A hidden feasible schedule is used only during generation so that a feasible timetable is guaranteed to exist.\n")

    print("Data generated")
    print("exams:", N_EXAMS)
    print("students:", N_STUDENTS)
    print("seed:", DATA_SEED)


if __name__ == "__main__":
    main()
