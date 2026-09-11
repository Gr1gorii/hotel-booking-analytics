.PHONY: setup data train report all app test

setup:
	uv sync --frozen

data:
	uv run python run.py prepare

train:
	OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 uv run python run.py train

report:
	uv run python run.py report

all: data train report

app:
	uv run streamlit run app.py

test:
	uv run pytest -q
