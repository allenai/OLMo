IMAGE_NAME_BASE = dolma
BEAKER_WORKSPACE = "ai2/llm-testing"
BEAKER_USER = $(shell beaker account whoami --format=json | jq -r '.[0].name')

.PHONY : run-checks
run-checks :
	isort --check .
	black --check .
	flake8 .
	mypy .
	CUDA_VISIBLE_DEVICES='' pytest -v --color=yes tests/

.PHONY : beaker-info
beaker-info :
	@echo "Beaker user $(BEAKER_USER)"

.PHONY : gantry-image
gantry-image :
	docker build -f Dockerfile.gantry -t $(IMAGE_NAME_BASE)-gantry .
	beaker image create $(IMAGE_NAME_BASE)-gantry --name $(IMAGE_NAME_BASE)-gantry-tmp --workspace $(BEAKER_WORKSPACE)
	beaker image delete $(shell beaker workspace images $(BEAKER_WORKSPACE) --format=json | jq -r -c '.[] | select( .name == "$(IMAGE_NAME_BASE)-gantry" ) | .fullName') || true
	beaker image rename $(BEAKER_USER)/$(IMAGE_NAME_BASE)-gantry-tmp $(IMAGE_NAME_BASE)-gantry

.PHONY : check-cpu-install
check-cpu-install :
	@python -c 'from dolma import check_install; check_install(cuda=False)'

.PHONY : check-cuda-install
check-cuda-install :
	@python -c 'from dolma import check_install; check_install(cuda=True)'
