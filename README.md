
# GeoVista: Web-Augmented Agentic Visual Reasoning for Geolocalization

<!-- <div align="center">
[💻 Code](https://github.com/ekonwang/GeoVista) | 📃 Paper | [🤗 GeoVista-RL-6k-7B](https://huggingface.co/LibraTree/GeoVista-RL-6k-7B)
</div> -->

<div align="center">
<a href="https://github.com/ekonwang/GeoVista">💻 Code</a> | 📃 Paper | <a href="https://huggingface.co/LibraTree/GeoVista-RL-6k-7B">🤗 GeoVista-RL-6k-7B</a>
</div>


<!-- ![](./assets/visuothink.png) -->
![](./assets/agentic_pipeline.png)

## Quick Start

1. Setup the environment:
```bash
conda create -n geo-vista python==3.10 -y
conda activate geo-vista

bash setup.sh
```

2. Set up web search API key

We use [Tavily](https://www.tavily.com/) during inference and training. You can sign up for a free account and get your Tavily API key, and then update the `TAVILY_API_KEY` variable of the `.env` file.

3. Download the pre-trained model 

from [HuggingFace](https://huggingface.co/LibraTree/GeoVista-RL-6k-7B) and place it in the `./.temp/checkpoints/GeoVista-RL-6k-7B` directory.

## Benchmarks

![](assets/figure-2-benchmark-evaluation.png)

We have already released the [GeoBench](https://huggingface.co/datasets/LibraTree/GeoBench) Dataset on HuggingFace 🤗, a benchmark that includes photos and panoramas from around the world, along with a subset of satellite images of different cities to rigorously evaluate the geolocalization ability of agentic models.

![](assets/figure-3-benchmark.png)

GeoBench is a high-resolution, multi-source, globally annotated dataset to evaluate models’ general geolocalization ability.

We assess benchmarks along five axes: **Global Coverage (GC)**, indicating whether images span diverse regions worldwide; **Reasonable Localizability (RC)**, measuring whether non-localizable or trivially localizable images are filtered out to preserve meaningful difficulty; **High Resolution (HR)**, requiring all images to have at least (1,\mathrm{M}) pixels for reliable visual clue extraction; **Data Variety (DV)**, capturing whether multiple image modalities or sources are included to test generalization; and **Nuanced Evaluation (NE)**, which checks whether precise coordinates are available to enable fine-grained distance-based metrics such as haversine error.


| **Benchmark**                                                                 | **Year** | **GC** | **RC** | **HR** | **DV** | **NE** |
| ----------------------------------------------------------------------------- | -------: | :----: | :----: | :----: | :----: | :----: |
| **[Im2GPS](https://doi.org/10.1109/CVPR.2008.4587784)**                       |     2008 |    ✓   |        |        |        |        |
| **[YFCC4k](https://api.semanticscholar.org/CorpusID:7449120)**                |     2017 |    ✓   |        |        |        |        |
| **[Google Landmarks v2](https://api.semanticscholar.org/CorpusID:214802288)** |     2020 |    ✓   |        |        |        |        |
| **[VIGOR](https://api.semanticscholar.org/CorpusID:227151840)**               |     2022 |        |        |        |    ✓   |        |
| **[OSV-5M](https://api.semanticscholar.org/CorpusID:269448726)**              |     2024 |    ✓   |    ✓   |        |        |    ✓   |
| **[GeoComp](https://doi.org/10.48550/arXiv.2502.13759)**                      |     2025 |    ✓   |    ✓   |        |        |    ✓   |
| **GeoBench (ours)**                                                           |     2025 |    ✓   |    ✓   |    ✓   |    ✓   |    ✓   |




## Citation
Please consider citing our paper and starring this repo if you find them helpful. Thank you!
```bibtex
@misc{wang2025geovista,
      title={GeoVista: Web-Augmented Agentic Visual Reasoning for Geolocalization}, 
      author={Yikun Wang et al.},
      year={2025},
      url={https://github.com/ekonwang/GeoVista}, 
}
```
