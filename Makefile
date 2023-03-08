# If you update this, also update BEAKER_IMAGE in .github/workflows/main.yml
IMAGE_NAME_BASE = dolma
# If you update this, also update BEAKER_WORKSPACE in .github/workflows/main.yml
BEAKER_WORKSPACE = "ai2/llm-testing"

BEAKER_USER = $(shell beaker account whoami --format=json | jq -r '.[0].name')
GANTRY_IMAGE = $(shell beaker workspace images $(BEAKER_WORKSPACE) --format=json | jq -r -c '.[] | select( .name == "$(IMAGE_NAME_BASE)-gantry" ) | .fullName')
TEST_IMAGE =  $(shell beaker workspace images $(BEAKER_WORKSPACE) --format=json | jq -r -c '.[] | select( .name == "$(IMAGE_NAME_BASE)-test" ) | .fullName')

.PHONY : run-checks
run-checks :
	isort --check .
	black --check .
	flake8 .
	mypy .
	CUDA_VISIBLE_DEVICES='' pytest -v --color=yes tests/

.PHONY : beaker-info
beaker-info :
	@echo "Beaker user:   $(BEAKER_USER)"
	@echo "Gantry image:  $(GANTRY_IMAGE)"
	@echo "Testing image: $(TEST_IMAGE)"

.PHONY : images
images : gantry-image test-image

.PHONY : gantry-image
gantry-image :
	docker build -f Dockerfile.gantry -t $(IMAGE_NAME_BASE)-gantry .
	beaker image create $(IMAGE_NAME_BASE)-gantry --name $(IMAGE_NAME_BASE)-gantry-tmp --workspace $(BEAKER_WORKSPACE)
	beaker image delete $(GANTRY_IMAGE) || true
	beaker image rename $(BEAKER_USER)/$(IMAGE_NAME_BASE)-gantry-tmp $(IMAGE_NAME_BASE)-gantry

.PHONY : test-image
test-image :
	docker build -f Dockerfile.test -t $(IMAGE_NAME_BASE)-test .
	beaker image create $(IMAGE_NAME_BASE)-test --name $(IMAGE_NAME_BASE)-test-tmp --workspace $(BEAKER_WORKSPACE)
	beaker image delete $(TEST_IMAGE) || true
	beaker image rename $(BEAKER_USER)/$(IMAGE_NAME_BASE)-test-tmp $(IMAGE_NAME_BASE)-test

.PHONY : show-test-image
show-test-image :
	@echo $(TEST_IMAGE)

.PHONY : show-gantry-image
show-gantry-image :
	@echo $(GANTRY_IMAGE)

.PHONY : show-beaker-workspace
show-beaker-workspace :
	@echo $(BEAKER_WORKSPACE)

.PHONY : gantry-test
gantry-test :
	gantry run \
		--workspace "$(BEAKER_WORKSPACE)" \
		--priority "preemptible" \
		--beaker-image "$(GANTRY_IMAGE)" \
		--gpus 1 \
		--description "Test run" \
		--cluster ai2/allennlp-cirrascale \
		--cluster ai2/aristo-cirrascale \
		--cluster ai2/mosaic-cirrascale \
		--cluster ai2/mosaic-cirrascale-a100 \
		--cluster ai2/prior-cirrascale \
		--cluster ai2/s2-cirrascale \
		--cluster ai2/general-cirrascale \
		--cluster ai2/general-cirrascale-a100-80g-ib \
		--allow-dirty \
		--venv base \
		--timeout -1 \
		--yes \
		-- make check-cuda-install

.PHONY : check-cpu-install
check-cpu-install :
	@python -c 'from dolma import check_install; check_install(cuda=False)'

.PHONY : check-cuda-install
check-cuda-install :
	@python -c 'from dolma import check_install; check_install(cuda=True)'
