# CSE 151B Competition

## Contents

| File | Description |
|---|---|
| `starter_code_cse151b_comp.ipynb` | Notebook to run and test on public dataset |
| `run_inference.py` | File do inference on private dataset (Final Submission) |
| `judger.py` | Response scoring logic |
| `utils.py` | Utilities used by `judger.py` |
| `data/public.jsonl` | Public dataset with ground-truth answers |
| `data/private.jsonl` | Private dataset with no answers |
| `results/` | Output csv files written at runtime |

GPU type used (e.g., A100, RTX 4090, T4, etc.) and approximate total generation/inference time
- Our team use the A5000 GPU and since we are using transformer, not vllm, each question takes approximately 4 minutes to run and in total for 943 questions, it would take approxiately 3772 minutes (2.6 days) to run. To save time, we partitioned the task of running the public and private dataset into 3 and each of us would run a subset of the data, hence the reason why we included START_INDEX and END_INDEX in our code. Other than that the START_INDEX should typically be 0 and END_INDEX should be length of the dataset. So if the dataset is not size 943 (size of private), then END_INDEX should be adjusted in run_inference.py.

Instructions on how to download/set up model weights (e.g., which directory to place them in)
- Since we use the default "Qwen/Qwen3-4B-Thinking-2507" model, there is no need to download any trained hugging face model. The only thing is to ensure the private.jsonl is in the data folder.

How to call your run_inference() function to reproduce results
- run_inference is a python file, so after setting up the virtual environment following Anthony's piazza post (https://piazza.com/class/mn3rnp8gniz5qv/post/152) and also doing 'pip install pandas', simply run 'python run_inference.py'.