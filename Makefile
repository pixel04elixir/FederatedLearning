PYTHON?=python3
ACTIVATE?=. .venv/bin/activate;
BLACK_ARGS=--extend-exclude="migrations|data|lib|etc" .

# Django Configuration
PORT = 8001

virtualenv:
	@echo "-> Making Virtual Environment"
	@${PYTHON} -m venv .venv

install: virtualenv
	@echo "-> Installing Dependencies"
	@${ACTIVATE} pip3 install -r etc/requirements.txt

dev: virtualenv
	@echo "-> Installing Developer Dependencies"
	@${ACTIVATE} pip3 install -r etc/dev.txt

format:
	@echo "-> Run isort imports validation"
	@${ACTIVATE} isort .
	@echo "-> Run black validation"
	@${ACTIVATE} black ${BLACK_ARGS}

check: test
	@echo "-> Run isort imports ordering validation"
	@${ACTIVATE} isort --check-only .
	@echo "-> Run black validation"
	@${ACTIVATE} black --check ${BLACK_ARGS}