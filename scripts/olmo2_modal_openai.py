# # Run OpenAI-compatible LLM inference with vLLM

# This is a simple example of running an OLMo 2 model on Modal.com with an OpenAI API and vLLM.
# It is based on Modal.com's own reference example for OpenAI with vLLM here:
# https://github.com/modal-labs/modal-examples/blob/ed89980d7288cd35c57f23861ba1b1c8d198f68d/06_gpu_and_ml/llm-serving/vllm_inference.py

import os

import modal

MODEL_NAME = "allenai/OLMo-2-1124-13B-Instruct"
MODEL_REVISION = "3a5c85baefbb1896a54d56fe2e76c0395627ddf4"
MODEL_DIR = "/root/models/{MODEL_NAME}"

N_GPU = 1
GPU_CONFIG = modal.gpu.A100(size="80GB", count=N_GPU)

APP_NAME = "OLMo-2-1124-13B-Instruct-openai"
APP_LABEL = APP_NAME.lower()

ONE_MINUTE = 60  # seconds
ONE_HOUR = 60 * ONE_MINUTE

# ## Download the model weights

# Our approach here differs from Modal's vllm_inference.py in two key ways:
#
# First, obviously, substituting an OLMo 2 model for the one from their example,
#
# Second, in their example, the weights are manually uploaded to a modal.com
# storage volume, which is then mounted as a directory by the inference container.
# In our example below, we use huggingface_hub's snapshot_download  to download
# the weights from HuggingFace directly into a local directory when building the
# container image.


def download_model_to_image(model_dir, model_name, model_revision):
    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache

    os.makedirs(model_dir, exist_ok=True)

    snapshot_download(
        model_name,
        revision=model_revision,
        local_dir=model_dir,
    )
    move_cache()


# ## Set up the container image

# Our first order of business is to define the environment our server will run in
# the container image. (See https://modal.com/docs/guide/custom-container)

# This differs from vllm_inference.py in two major ways: first, as of the time this
# is being written, the OLMo 2 model architecture requires a version of vLLM that is
# too recent to have a tagged version, requiring a commit-specific wheel from vLLM's
# archives.
#
# Second, we call the download_model_to_image function here, to build the resulting local
# directory with model weights into our image.
#
# The image build can take several minutes the first time you run this script. The good
# news is that, as long as the image definition doesn't change, Modal will cache and reuse
# the image on later runs without having to re-run the full image build for each new container
# instance.

vllm_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "vllm==0.6.6.post1",
        "torch==2.5.1",
        "transformers==4.47.1",
        "ray==2.10.0",
        "huggingface_hub==0.24.0",
        "hf-transfer==0.1.6",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(
        download_model_to_image,
        timeout=60 * ONE_MINUTE,  # typically much faster but set high to be conservative
        kwargs={
            "model_dir": MODEL_DIR,
            "model_name": MODEL_NAME,
            "model_revision": MODEL_REVISION,
        },
    )
)

# ## Build a vLLM engine and serve it

# The code below defines a single app container, with its configuration specified in
# the app.function annotation at the top. This handles setting up the inference
# engine as well as a fastapi webserver, which is configured with completions and
# chat endpoints.
#
# Note that keep_warm=0 means the app will spin down entirely when idle, which is more
# cost efficient if the endpoint is not regularly used because you don't need to pay for
# GPUs when they're not being used, but does require a cold start after idle timeouts
# which will delay the initial responses until the instance finishes starting back up,
# usually on the order of a minute or so.
#
app = modal.App(APP_NAME)


@app.function(
    image=vllm_image,
    gpu=GPU_CONFIG,
    keep_warm=0,  # Spin down entirely when idle
    container_idle_timeout=5 * ONE_MINUTE,
    timeout=24 * ONE_HOUR,
    allow_concurrent_inputs=1000,
    secrets=[modal.Secret.from_name("example-secret-token")],  # contains MODAL_TOKEN used below
)
@modal.asgi_app()
def serve():
    import fastapi
    import vllm.entrypoints.openai.api_server as api_server
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.entrypoints.logger import RequestLogger
    from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
    from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
    from vllm.entrypoints.openai.serving_engine import BaseModelPath
    from vllm.usage.usage_lib import UsageContext

    # create a fastAPI app that uses vLLM's OpenAI-compatible router
    web_app = fastapi.FastAPI(
        title=f"OpenAI-compatible {MODEL_NAME} server",
        description="Run an OpenAI-compatible LLM server with vLLM on modal.com 🚀",
        version="0.0.1",
        docs_url="/docs",
    )

    # security: CORS middleware for external requests
    http_bearer = fastapi.security.HTTPBearer(
        scheme_name="Bearer Token",
        description="See code for authentication details.",
    )
    web_app.add_middleware(
        fastapi.middleware.cors.CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # This example uses a token defined in the Modal secret linked above,
    # as described here: https://modal.com/docs/guide/secrets
    async def is_authenticated(api_key=fastapi.Security(http_bearer)):
        if api_key.credentials != os.getenv("MODAL_TOKEN"):
            raise fastapi.HTTPException(
                status_code=fastapi.status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
            )
        return {"username": "authenticated_user"}

    router = fastapi.APIRouter(dependencies=[fastapi.Depends(is_authenticated)])

    # wrap vllm's router in auth router
    router.include_router(api_server.router)
    # add authed vllm to our fastAPI app
    web_app.include_router(router)

    engine_args = AsyncEngineArgs(
        model=MODEL_DIR,
        tensor_parallel_size=N_GPU,
        gpu_memory_utilization=0.90,
        max_model_len=4096,
        enforce_eager=False,  # capture the graph for faster inference, but slower cold starts (30s > 20s)
    )

    engine = AsyncLLMEngine.from_engine_args(engine_args, usage_context=UsageContext.OPENAI_API_SERVER)

    model_config = get_model_config(engine)

    request_logger = RequestLogger(max_log_len=2048)

    base_model_paths = [BaseModelPath(name=MODEL_NAME.split("/")[1], model_path=MODEL_NAME)]

    api_server.chat = lambda s: OpenAIServingChat(
        engine,
        model_config=model_config,
        base_model_paths=base_model_paths,
        chat_template=None,
        chat_template_content_format=None,
        response_role="assistant",
        lora_modules=[],
        prompt_adapters=[],
        request_logger=request_logger,
    )
    api_server.completion = lambda s: OpenAIServingCompletion(
        engine,
        model_config=model_config,
        base_model_paths=base_model_paths,
        lora_modules=[],
        prompt_adapters=[],
        request_logger=request_logger,
    )

    return web_app


# ## Deploy the server

# To deploy the API on Modal, just run
# ```bash
# modal deploy olmo2_modal_openai.py
# ```

# This will create a new app on Modal, build the container image for it, and deploy.

# ## Interact with the server

# Once it is deployed, you'll see a URL appear in the command line,
# something like `https://your-workspace-name--olmo-2-1124-instruct-openai-serve.modal.run`.

# You can find [interactive Swagger UI docs](https://swagger.io/tools/swagger-ui/)
# at the `/docs` route of that URL, i.e. `https://your-workspace-name--olmo-2-1124-instruct-openai-serve.modal.run/docs`.
# These docs describe each route and indicate the expected input and output
# and translate requests into `curl` commands. They also demonstrate authentication.

# For simple routes like `/health`, which checks whether the server is responding,
# you can even send a request directly from the docs.

# To interact with the API programmatically, you can use the Python `openai` library.

# ## Addenda

# The rest of the code in this example is utility code from Modal's original example.


def get_model_config(engine):
    import asyncio

    try:  # adapted from vLLM source -- https://github.com/vllm-project/vllm/blob/507ef787d85dec24490069ffceacbd6b161f4f72/vllm/entrypoints/openai/api_server.py#L235C1-L247C1
        event_loop = asyncio.get_running_loop()
    except RuntimeError:
        event_loop = None

    if event_loop is not None and event_loop.is_running():
        # If the current is instanced by Ray Serve,
        # there is already a running event loop
        model_config = event_loop.run_until_complete(engine.get_model_config())
    else:
        # When using single vLLM without engine_use_ray
        model_config = asyncio.run(engine.get_model_config())

    return model_config
