# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this repo is

A personal Python **learning/coursework sandbox** — not a package, not an application.
It holds eCornell machine-learning coursework (JCB702 bank-telemarketing, JCB703 COMPAS
fairness, a CIFAR-10 CNN) plus assorted scraping and plotting experiments.

There is **no build, no test suite, no linter, no CI, and no entry point.** Each `.py`
file is a standalone script run directly. Do not invent a package layout, add
`setup.py`/`pyproject.toml`, or introduce a test framework unless explicitly asked.

## Layout

| Path | Contents |
| --- | --- |
| `Hello World.py` | matplotlib sine/cosine demo with an inline assert |
| `Hello World/` | `Module_Test.py` (KMeans on synthetic blobs), `Mouse_Mover` (pyautogui jiggler, **no `.py` extension**), `bootcamp1.py` (empty) |
| `AI Module Final/` | Decision-tree final: script + `decision_tree.png`, `feature_importance.png`, `roc_curve.png` |
| `PRACTICE 1/` | `BankTelemarketing1.py` — logistic regression vs decision tree, ROC/AUC; reads its data from `PRACTICE FINAL/` |
| `PRACTICE 2/` | Output PNGs only, no script |
| `PRACTICE FINAL/` | The two `JCB702_BankTelemarketing*_PRACTICE.csv` datasets — both are class-balanced 50/50, all-numeric, no nulls |
| `Compus/` | `Compus.py` — COMPAS disparate-impact ratio; reads and rewrites its CSVs |
| `DATA/` | Amazon headphone scrapes (`amazon_gemscrape.py`, `amazon_scrape2.py`), `Gemini Test 1.py` (CSV embedded as a string literal), CSVs and histogram PNGs |
| `JumpOff1/eCornell/` | `eCornell.py` — Keras/TensorFlow CNN on CIFAR-10, plus its own `requirements.txt` |
| `cacert.pem` | CA bundle committed at repo root; leave it alone |

## Environment

- Python **3.11** (`python3` / `python` on PATH). No virtualenv is checked in — `.gitignore`
  excludes `env*/`, `venv*/`, `numpy_env/`, `tensorflow_env/`, and `Scripts/`.
- **`requirements.txt` is UTF-16LE with a BOM and CRLF line endings** (both the root one and
  `JumpOff1/eCornell/requirements.txt`). `pip install -r` and anything else expecting UTF-8
  will fail on it. Convert on the fly rather than "fixing" the file in place unless asked:

  ```bash
  iconv -f UTF-16 -t UTF-8 requirements.txt | sed 's/\r$//' > /tmp/req.txt
  pip install -r /tmp/req.txt
  ```

- The root pins include `tensorflow-intel==2.18.0`, a **Windows-only** wheel. It will not
  resolve on Linux/macOS — install plain `tensorflow` instead when running `eCornell.py`.
- `Mouse_Mover` needs `pyautogui` and a real display; it never appears in either
  requirements file and cannot run headless.

## Running scripts

Directory names contain spaces — **always quote paths**:

```bash
python3 "PRACTICE 1/BankTelemarketing1.py"
python3 "AI Module Final/AI MODULE FINAL.py"
```

Scripts end in `plt.show()`, which blocks on a headless machine. Set a non-interactive
backend when running them here:

```bash
MPLBACKEND=Agg python3 "Hello World.py"
```

Plots are written as PNG/JPG next to the script that produced them. Regenerated images are
expected churn; don't treat them as accidental.

## Known landmines

Pre-existing breakage. Fix only what the current task actually covers, and say so —
do not sweep the repo.

1. **`Hello World/bootcamp1.py` is empty** (0 bytes) and `Hello World/Mouse_Mover` calls
   `main()` at import time with an infinite loop — never import it.
2. `DATA/Gemini Test 1.py` embeds its dataset as a triple-quoted CSV string. Edit the
   literal, not a file on disk.
3. `Hello World.py` still saves `complex_plot.jpg` relative to the current working
   directory rather than to the script. Harmless while it is run from the repo root,
   where the script also lives.

Fixed previously, noted so the history reads clearly: the `D:\Codespace\...` Windows
paths in `Compus/Compus.py`, `AI Module Final/AI MODULE FINAL.py`, and
`PRACTICE 1/BankTelemarketing1.py`; the missing `import os` in `AI MODULE FINAL.py`; and
the dataset `BankTelemarketing1.py` reads, which was repointed at
`PRACTICE FINAL/JCB702_BankTelemarketing1_PRACTICE.csv`.

## Conventions

- Match the existing style: flat top-level scripts, `# comment` above each step,
  `print()` for output, `matplotlib` for charts. No classes, no logging framework,
  no type hints — this is coursework, and rewriting it obscures the author's own work.
- Keep datasets beside the script that consumes them.
- Resolve every data and output path from `Path(__file__).resolve().parent`, never from
  the current working directory and never as an absolute path. Scripts must run from
  any cwd.
- Do not reformat, refactor, or "modernize" files a task didn't ask you to touch.
- Do not commit virtualenvs, `.h5` model files, or `.db` files — `.gitignore` already
  covers them; keep it that way.

## Git

- Work on the branch you were assigned; never push to `main` without explicit permission.
- Push with `git push -u origin <branch>`.
- Do not open a pull request unless asked.
