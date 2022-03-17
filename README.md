## Math Dense Retrievers

This is the repository for replication of the experiments in our paper:

Wei Zhong, Jheng-Hong Yang, and Jimmy Lin. *Evaluating Token-Level and Passage-Level Dense Retrieval Models for Math Information Retrieval*.

### Data Download

We have made our prebuilt-indexes (optional), experimenting models and corpus files available for download:

```shell
wget https://vault.cs.uwaterloo.ca/s/AFTWLbRdKSMBpsK/download -O prebuilt-indexes.tar
wget https://vault.cs.uwaterloo.ca/s/oDpJ9oHPcpm7dLp/download -O experiments.tar.gz
wget https://vault.cs.uwaterloo.ca/s/q5tFQRf8RwZr7dW/download -O corpus.tar.gz
```

Extract tarballs:

```shell
tar xzf corpus.tar.gz experiments.tar.gz
tar xf prebuilt-indexes.tar
```

If you want to replicate our prebuilt indexes,  just skip downloading the `prebuilt-indexes` tarball, and create an empty directory to hold the new indexes built by your own:

```shell
mkdir prebuilt-indexes
```

### Get Source Code

Download the pya0 build (source code can be found [here](https://github.com/approach0/pya0/tree/math-dense-retrievers-replication)) and our pyserini fork which are used to replicate our results:

```shell
pip install pya0==0.3.4
git clone -b patch-colbert git@github.com:w32zhong/pyserini.git ./code/pyserini
```

Download the pya0 source code as well, since it contains the evaluation config file and our experiment script:

```shell
git clone -b math-dense-retrievers-replication git@github.com:approach0/pya0.git ./code/pya0
```

Alternatively, you can only download what is needed for running our evaluations:

```shell
wget https://raw.githubusercontent.com/approach0/pya0/math-dense-retrievers-replication/utils/transformer_eval.ini
wget https://raw.githubusercontent.com/approach0/pya0/math-dense-retrievers-replication/experiments/dense_retriever.sh
chmod +x dense_retriever.sh
```

### Modify Config Files

In the config file `code/pya0/utils/transformer_eval.ini`, change the following path to your current working directory (where this README file locates):

```ini
store = /store2/scratch/w32zhong/math-dense-retrievers
```

You may also need to insert your own local GPU information to the default devices (the corresponding array items represent the cuda device and the device capacity in GiB):

```ini
devices = {
        "cpu": ["cpu", "0"],
        "titan_rtx": ["cuda:2", "24"],
        "a6000_0": ["cuda:0", "48"],
        "a6000_1": ["cuda:1", "48"],
        "rtx2080": ["cuda:0", "11"]
    }
```

### Run Experiments

In the following, you will need to run pya0 for evaluation.

For illustration, I assume you have cloned its source code and your working directory is at pya0 source code root:

```shell
cd code/pya0/
```

The reference exepriment script is located at `experiments/dense_retriever.sh`, please refer to this script for running all the experiments. Remember to replace the device name to your own GPU name as inserted in the `transformer_eval.ini`.

Here is an example of indexing the NTCIR12 dataset using our DPR model:

```shell
INDEX='python -m pya0.transformer_eval index ./utils/transformer_eval.ini'
$INDEX index_ntcir12_dpr --device <your_own_device_name>
```

The index will be generated under the `prebuilt-indexes` directory if you have not downloaded our prebuilt indexes.

To prevent from overwriting, we abort the indexer when an existing index is found. So you will need to delete any existing index before start indexing a new one:

```shell
rm -rf /path/to/your/math-dense-retrievers/prebuilt-indexes/index-DPR-ntcir12
```

As another example, use the DPR model (located at `experiments/math-dpr`) we trained to generate results for the NTCIR-12 dataset:

```shell
SEARCH='python -m pya0.transformer_eval search ./utils/transformer_eval.ini'
$SEARCH search_ntcir12_dpr --device cpu
```

(since the DPR searcher only needs to encode queries, feel free to only use CPU device this time)

### Evaluation

For NTCIR-12 run files, evaluate them by:

```shell
./eval-ntcir12.sh ../../experiments/runs/search_ntcir12_dpr.run 
Fully relevant:
P_5                     all     0.3200
P_10                    all     0.2150
P_15                    all     0.1700
P_20                    all     0.1550
bpref                   all     0.5159
Partial relevant:
P_5                     all     0.4100
P_10                    all     0.3050
P_15                    all     0.2567
P_20                    all     0.2400
bpref                   all     0.4269
```

For ARQMath-2 run files, evaluate them by our utility scripts (which internally invokes the [official evaluation script](https://drive.google.com/drive/folders/15uIdGFo7MPK3IdkpMG2emnwpOdzuNwbN?usp=sharing) but adds new statistics like BPref score and Judge Rate):

```shell
./eval-arqmath2-task1/preprocess.sh cleanup
./eval-arqmath2-task1/preprocess.sh ../../experiments/runs/search_arqmath2_dpr.run
./eval-arqmath2-task1/eval.sh
100000 ./eval-arqmath2-task1/input/search_arqmath2_dpr_run
++ sed -i 's/ /\t/g' ./eval-arqmath2-task1/input/search_arqmath2_dpr_run
++ python3 ./eval-arqmath2-task1/arqmath_to_prim_task1.py -qre topics-and-qrels/qrels.arqmath-2021-task1-official.txt -sub ./eval-arqmath2-task1/input/ -tre ./eval-arqmath2-task1/trec-output/ -pri ./eval-arqmath2-task1/prime-output/
++ python3 ./eval-arqmath2-task1/task1_get_results.py -eva trec_eval -qre topics-and-qrels/qrels.arqmath-2021-task1-official.txt -pri ./eval-arqmath2-task1/prime-output/ -res ./eval-arqmath2-task1/result.tsv
trec_eval topics-and-qrels/qrels.arqmath-2021-task1-official.txt ./eval-arqmath2-task1/prime-output/prime_search_arqmath2_dpr_run -m ndcg
trec_eval topics-and-qrels/qrels.arqmath-2021-task1-official.txt ./eval-arqmath2-task1/prime-output/prime_search_arqmath2_dpr_run -l2 -m map
trec_eval topics-and-qrels/qrels.arqmath-2021-task1-official.txt ./eval-arqmath2-task1/prime-output/prime_search_arqmath2_dpr_run -l2 -m P
trec_eval topics-and-qrels/qrels.arqmath-2021-task1-official.txt ./eval-arqmath2-task1/prime-output/prime_search_arqmath2_dpr_run -l2 -m bpref
python -m pya0.judge_rate topics-and-qrels/qrels.arqmath-2021-task1-official.txt ./eval-arqmath2-task1/trec-output/search_arqmath2_dpr_run
++ cat ./eval-arqmath2-task1/result.tsv
++ sed -e 's/[[:blank:]]/ /g'
System nDCG' mAP' p@10 BPref Judge
search_arqmath2_dpr_run 0.2700 0.0869 0.1521 0.0972 66.3
```

The pre-existing run files under that directory are what we have generated for reporting our results. Be aware that, by default, all newly generated run files will overwrite files under the `experiments/runs` directory.
