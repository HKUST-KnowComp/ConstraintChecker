# ConstraintChecker
Official code repository for the EACL2024 paper "ConstraintChecker: A Plugin for Large Language Models to Reason on Commonsense Knowledge Bases"

For the method and result, please refer to our [paper]().

## Code and data

We recommend you to use Google Colab runtime to run our code. In case you set up a local Python environment, please run the install the libraries in `requirements.txt`. After that, please follow below steps. Note that **code files often hardcode the paths to data, and also contain sample bash script to run**.

1. sample and preprocess data:
  - `prompting/convert_triple_to_nl_assertion.py -m conversion`: stratificationally downscale the CKBPv2 dev and test split, and convert triples to the free-text format using all three templates.
  - `prompting/atomic_random_sampling.py`: create the downscaled Synthetic Discriminative (SD) version of the ATOMIC2020 test split, and convert triples to the free-text format using all three templates.
  - `prompting/KATE_search_topk.py`: select instances from CKBP training data as exemplars for few-shot baselines.

2. run all baselines with(out) ConstraintChecker: run `prompting/ChatGPT_prompting.ipynb`, then save all output files to `results`. Subfolders `prompting_tst` and `generalizability` respectively contain output files w.r.t CKBPv2 and SD-ATOMIC2022 data.

3. manually post-process the output to programmatically parse the answer (we did it already :D)

4. result, analysis, and ablation:
  - `prompting/scoring.py` and `prompting/generalizability_scoring.py`: compute the score of baselines with(out) ConstraintChecker, w.r.t CKBPv2 and SD-ATOMIC2022 data.
  - `prompting_data/tst_data_to_eval_analysis.xlsx`: data for analysis "Where does the improvement of Con-
straintChecker come from?"

For the pilot study, please use `prompting/scoring.py -s dev` and corresponding model's output files.


## Contact information

For help or issues using ConstraintChecker, please submit a GitHub issue.

For personal communication related to ConstraintChecker, please contact Quyet V. Do (`vqdo@connect.ust.hk`).


## Citation

If you use or extend our work, please cite the following:
```
@inproceedings{Do2023ConstraintChecker,
    title={{C}onstraint{C}hecker: A Plugin for Large Language Models to Reason on Commonsense Knowledge Bases},
    author={Quyet V. Do and Tianqing Fang and Shizhe Diao and Zhaowei Wang and Yangqiu Song},   
    booktitle={Proceedings of EACL},
    month = mar,
    year={2024},
    address = "St. Julians, Malta",
    publisher = "Association for Computational Linguistics",
}
```