# Makefile

# Define a variable for the poetry command to keep it DRY
POETRY = poetry

# Install dependencies
.PHONY: install
install:
	$(POETRY) install

# Install pre-commit hooks
.PHONE: install-pre-commit
install-pre-commit:
	$(POETRY) run pre-commit uninstall
	$(POETRY) run pre-commit install

# Check code style
.PHONY: lint
lint:
	$(POETRY) run pre-commit run --all-files

# Update project dependencies
.PHONY: update
update: install install-pre-commit ;

# Train and Save best models
.PHONY: test
test:
	$(POETRY) run python test.py

# Try out trained models
.PHONY: run
run:
	$(POETRY) run python exec.py
