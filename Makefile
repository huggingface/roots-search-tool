.PHONY: quality style test

# Check that source code meets quality standards

quality:
	black --check --line-length 119 --target-version py36 tests src metrics comparisons 
	isort --check-only tests src metrics
	flake8 tests src metrics

# Format source code automatically

style:
	black --line-length 119 --target-version py36 scripts web
	black --line-length 115 --target-version py36 notebooks
	isort scripts notebooks web

# Run tests for the library

test:
	python -m pytest -n auto --dist=loadfile -s -v ./tests/