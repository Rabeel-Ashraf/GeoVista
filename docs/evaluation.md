
## Inference & Evaluation Pipeline

### Inference

- Download the GeoBench dataset from [HuggingFace](https://huggingface.co/datasets/LibraTree/GeoBench) and place it in the `./.temp/datasets` directory.

```bash
python3 scripts/download_hf.py --dataset LibraTree/GeoBench --local_dataset_dir ./.temp/datasets
```

- Download the pre-trained model from [HuggingFace](https://huggingface.co/LibraTree/GeoVista-RL-12k-7B) and place it in the `./.temp/checkpoints` directory.

```bash
python3 scripts/download_hf.py --model LibraTree/GeoVista-RL-12k-7B --local_model_dir .temp/checkpoints/
```

- Deploy the GeoVista model with vllm:

```bash
bash inference/vllm_deploy.sh
```

- Configure the settings including the output directory, run the inference script:

```bash
bash inference/run_inference.sh
```

After running the above commands, you should be able to see the inference results in the specified output directory, e.g., `./.temp/outputs/geobench/geovista-rl-12k-7b/`, which contains the `inference_<timestamp>.jsonl` file with the inference results.


### Evaluation

- After obtaining the inference results, you can evaluate the geolocalization performance using the evaluation script:

```bash
MODEL_NAME=geovista-rl-12k-7b
BENCHMARK=geobench
EVALUATION_RESULT=".temp/outputs/${BENCHMARK}/${MODEL_NAME}/evaluation.jsonl"

python3 eval/eval_infer_geolocation.py \
  --pred_jsonl <The inference file path> \
  --out_jsonl ${EVALUATION_RESULT}\
  --dataset_dir .temp/datasets/${BENCHMARK} \
  --num_samples 1500 \
  --model_verifier \
  --no_eval_accurate_dist \
  --timeout 120 --debug | tee .temp/outputs/${BENCHMARK}/${MODEL_NAME}/evaluation.log 2>&1
```

You can acclerate the evaluation process by changing the `workers` argument in the above command (default is 1):

```bash
  --workers 8 \
```

### Nuanced Evaluation

To perform nuanced evaluation on GeoBench to obtain the haversine distance with the groud truth, you need to:

-  Set up Google Cloud API key

1. You need to have [Google Cloud](https://console.cloud.google.com/) account and set up your project.
2. Enable the [Geocoding API](https://console.cloud.google.com/marketplace/product/google/geocoding-backend.googleapis.com) for your project.
3. Create API [credentials](https://console.cloud.google.com/apis/credentials) and obtain your API key.

Finally, update the `GOOGLE_MAPS_API_KEY` variable of the `.env` file with your API key.

- Run the geocoding test script to verify the API key is working:

```bash
python3 eval/utils_geocode.py
```

If the test is successful, you should see an output like:

```text
[extract_pred_address] attempt=0 sending messages
[extract_pred_address] raw response: '{"address":"Einkaufszentrum Rahlstedt-Ost, Schöneberger Straße, 22149 Hamburg, Germany"}'
[extract_pred_address] parsed address: Einkaufszentrum Rahlstedt-Ost, Schöneberger Straße, 22149 Hamburg, Germany
[geocode] trying Google Geocoding: https://maps.googleapis.com/maps/api/geocode/json?address=Einkaufszentrum+Rahlstedt-Ost%2C+Sch%C3%B6neberger+Stra%C3%9Fe%2C+22149+Hamburg%2C+Germany&key=****
[geocode] Google status=OK, results=1
[geocode] Google success lat=53.5868597, lng=10.1533124
[geocode] trying Google Geocoding: https://maps.googleapis.com/maps/api/geocode/json?address=Tencent+beijing+Office%2C+Beijing%2C+China&key=****
[geocode] Google status=OK, results=1
[geocode] Google success lat=39.904211, lng=116.407395
```

- Finally, run the nuanced evaluation script and remove the `--no_eval_accurate_dist` flag:

```bash
MODEL_NAME=geovista-rl-12k-7b
BENCHMARK=geobench
EVALUATION_RESULT=".temp/outputs/${BENCHMARK}/${MODEL_NAME}/evaluation.jsonl"

python3 eval/eval_infer_geolocation.py \
  --pred_jsonl <The inference file path> \
  --out_jsonl ${EVALUATION_RESULT}\
  --dataset_dir .temp/datasets/${BENCHMARK} \
  --num_samples 1500 \
  --model_verifier \
  --timeout 120 --debug | tee .temp/outputs/${BENCHMARK}/${MODEL_NAME}/evaluation.log 2>&1
```
