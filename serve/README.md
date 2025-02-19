The production FastAPI service and container are maintained in the sibling
[`afri-ai-serving`](../../afri-ai-serving/README.md) repository. Set `ASR_MODEL`
to the exported CTranslate2 directory and `ASR_COMPUTE_TYPE=int8`.

Use `python -m eval.run --backend ct2 ...` here for controlled single-process
CPU timing; HTTP service latency and model inference latency are separate measurements.
