# Azure AI MaaS - Llama + TRT-LLM

Created: December 20, 2024 10:43 AM

## Llama 3.3 70B

Follow these steps here to set up the environment (Docker container) for building and running the engines: [guidance-ai/llgtrt: TensorRT-LLM server with Structured Outputs (JSON) built with Rust](https://github.com/guidance-ai/llgtrt/tree/main?tab=readme-ov-file#building-or-pulling-docker-container)

### Quantization:

`python /opt/TensorRT-LLM-examples/quantization/quantize.py --model_dir /path/to/meta-llama/Llama-3.3-70B-Instruct --dtype bfloat16 --qformat fp8 --kv_cache_dtype fp8 --output_dir /path/to/checkpoints/output_dir --calib_size 512 --tp_size 2 --pp_size 1`

### Build:

`trtllm-build --checkpoint_dir /path/to/checkpoints/output_dir --output_dir /path/to/engines/output_dir --max_num_tokens 4096 --max_input_len 127000 --max_seq_len 128000 --max_batch_size 256 --use_paged_context_fmha enable --use_fp8_context_fmha enable --gemm_plugin auto --workers 2`

### Run:

Follow the steps outlined here: https://github.com/guidance-ai/llgtrt/tree/main?tab=readme-ov-file#running-the-engine

Note: In addition to the `config.json` file and the engines, make sure you have the following files under `/path/to/engines/output_dir` as well:

- `chat_template.j2`
    - Copy this from here: https://github.com/guidance-ai/llgtrt/blob/main/model_configs/llama31/chat_template.j2
- `runtime.json`
    - Create this file like so:
        
        ```json
        {
            "guaranteed_no_evict": false,
            "max_batch_size": 128,
            "max_num_tokens": 4096,
            "enable_chunked_context": true,
            "enable_kv_cache_reuse": true,
            "kv_cache_free_gpu_mem_fraction": 0.9
        }
        ```
        
- `special_tokens_map.json`
    - Copy from `/path/to/meta-llama/Llama-3.3-70B-Instruct`  (downloaded from HuggingFace)
- `tokenizer_config.json`
    - Copy from `/path/to/meta-llama/Llama-3.3-70B-Instruct`  (downloaded from HuggingFace)
- `tokenizer.json`
    - Copy from `/path/to/meta-llama/Llama-3.3-70B-Instruct`  (downloaded from HuggingFace)

---

## Llama 3.1 405B

Follow these steps here to set up the environment (Docker container) for building and running the engines: [guidance-ai/llgtrt: TensorRT-LLM server with Structured Outputs (JSON) built with Rust](https://github.com/guidance-ai/llgtrt/tree/main?tab=readme-ov-file#building-or-pulling-docker-container)

### Convert checkpoints:

`python /opt/TensorRT-LLM-examples/llama/convert_checkpoint.py --model_dir /path/to/meta-llama/Llama-3.1-405B-Instruct-FP8/ --output_dir /path/to/checkpoints/output_dir --dtype bfloat16 --tp_size 8 --pp_size 1`

### Build:

`trtllm-build --checkpoint_dir /path/to/checkpoints/output_dir --output_dir /path/to/engines/output_dir --max_num_tokens 4096 --max_input_len 127000 --max_seq_len 128000 --max_batch_size 256 --use_paged_context_fmha enable --workers 8`

### Run:

Follow the steps outlined here: https://github.com/guidance-ai/llgtrt/tree/main?tab=readme-ov-file#running-the-engine

Note: In addition to the `config.json` file and the engines, make sure you have the following files under `/path/to/engines/output_dir` as well:

- `chat_template.j2`
    - Copy this from here: https://github.com/guidance-ai/llgtrt/blob/main/model_configs/llama31/chat_template.j2
- `runtime.json`
    - Create this file like so:
        
        ```json
        {
            "guaranteed_no_evict": false,
            "max_batch_size": 128,
            "max_num_tokens": 4096,
            "enable_chunked_context": true,
            "enable_kv_cache_reuse": true,
            "kv_cache_free_gpu_mem_fraction": 0.9
        }
        ```
        
- `special_tokens_map.json`
    - Copy from `/path/to/meta-llama/Llama-3.1-405B-Instruct-FP8`  (downloaded from HuggingFace)
- `tokenizer_config.json`
    - Copy from `/path/to/meta-llama/Llama-3.1-405B-Instruct-FP8`  (downloaded from HuggingFace)
- `tokenizer.json`
    - Copy from `/path/to/meta-llama/Llama-3.1-405B-Instruct-FP8`  (downloaded from HuggingFace)

---

## Running benchmarks

Use the wheel provided in the email 🙂

### Requirements:

- Python 3.9+
- pip 20.0+

### Install and explore:

`pip install commonbench_sdk-0.1.4rc0-py3-none-any.whl`

`pip show commonbench-sdk`

`commonbench-sdk --help`

### Start benchmarking:

- Create the following yaml file, name it `benchmark_config.yaml` :
    
    ```json
    schema_version: "1.0"
    description: Benchmark Llama 3.3 70B
    display_name: benchmark_llama_33_70B
    parameters:
      endpoint_url: http://0.0.0.0:3001
      model_name: llama33_70b
      api_payload:
        api_type: ChatCompletion
    api_validation:
      enabled : false
    perf_benchmark:
      enabled: true
      concurrent_clients: [32, 64, 128, 256]
      total_tokens: [250, 500, 1000, 2000, 4000]
      prompt_token_percentages: [20, 40, 60, 80]
      total_requests_limit: 512
      enable_stream: false
    data_upload:
      enabled: false
    ```
    
    - Note: The `endpoint_url` might differ depending on which port you specified to be used
- `commonbench-sdk run-benchmark -i /path/to/benchmark_config.yaml -o /path/to/benchmark_outputs/`