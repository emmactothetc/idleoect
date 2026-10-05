# Instructor grading workflow

This folder contains behavioral checks, not solution code. Run student submissions
only in an isolated course environment because importing a submission executes its
Python modules.

From the instructor's clean `assignments` folder, grade a copied submission:

```bash
python grading/grade_assignment.py 1 --submission-root /path/to/student/assignments
python grading/grade_assignment.py 2 --submission-root /path/to/student/assignments
python grading/grade_assignment.py 3 --submission-root /path/to/student/assignments
```

Each command writes Markdown and JSON reports under the submission's
`grading_reports/` directory. The report gives points only for automatable code and
artifact behavior. Add the listed manual points after reading the executed notebook.
Use the private `answers/CRITICAL_QUESTIONS_KEY.md` only as an instructor guide; do
not copy it into a submission or the public repository.

Before A2 grading, the submission should contain its A1 protocol. Before A3
grading, it should also contain its A2 manifest, histories, and checkpoints. Missing
handoff artifacts correctly lose the corresponding automated provenance points.

These checks support the notebook rubric; they do not replace instructor judgment.
In particular, readable reasoning, architecture diagrams, critical questions,
failure analysis, and scientific interpretation remain manual.
