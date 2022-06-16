# ArMATH dataset

[paper](http://www.lrec-conf.org/proceedings/lrec2022/pdf/2022.lrec-1.37.pdf)

this dataset is the first large-scale dataset for solving Arabic Math Word Problem. There are 6000 samples and 883 templates. A template is an equation once the variables have been replaced with ordered placeholders. 

For the ease of use and uniform evaluation across future work, the dataset was randomly split into 5 folds; 1,200 samples in each. 

There are at most 15 variables and 10 constants. Constants are numbers that do not appear in the question body but the equation in at least five samples. These 10 constants are categorized as follows:

* Geometry: 3.14 and 0.5 
* Time: 12, 7, 60
* 0-4: numbers used in geometry, counting and facts.

The top 10 templates account for half of the samples. The table below shows the top templates and their freuqencies

| **Template** | **Frequency** |
| :---: | :---: |
| N0 / N1 | 631 |
| N0 - N1 | 491 |
| N0 * N1 | 481 |
| N1 * N0 | 361 |
| N0 + N1 | 254 |
| N1 / N0 | 245 |
| (N0 * N1) - N2 | 175 |
| N1 + N0 | 162 |
| (N0 / N1) - N2 | 123 |
| (N0 - N1) + N2 | 80 |

# ArMATH Solver

## Installation

* clone this directory
* install [`aravec`](https://github.com/bakrianoo/aravec#download) and [`fasttext`](https://fasttext.cc/docs/en/crawl-vectors.html#models) models, extract them in word2vec

* install dependencies

```bash
conda env create -f environment.yml
```



## Training

To train the Chinese model (to be used in transfer learning):

```bash
conda activate armath
python code/run.py \
		--output-dir "results/chinese_model" \
    	--n-workers $CPUS_PER_GPU \
        --batch-size $BATCH_SIZE \
        --embedding-size $EMBEDDING \
        --data-path datasets/chinese/Math_23K.json
```

To train the Arabic model: no transfer learning, one-hot encoding:

```bash
conda activate armath
python code/run.py \
		--output-dir "results/one-hot" \
    	--n-workers $CPUS_PER_GPU \
        --batch-size $BATCH_SIZE \
        --embedding-size $EMBEDDING \
        --data-path datasets/armath \
        --arabic
```

To train the Arabic model with no transfer learning, `aravec` embedding [for `fasttext`, replace `aravec` with `fasttext`]:

```bash
conda activate armath
python code/run.py \
		--output-dir "results/aravec" \
    	--n-workers $CPUS_PER_GPU \
        --batch-size $BATCH_SIZE \
        --embedding-size $EMBEDDING \
        --data-path datasets/armath \
        --embedding-type aravec \
        --embedding-model-name $PATH_TO_EMBEDDING \
        --arabic
```

For transfer learning:

```bash
conda activate armath
python code/run.py \
		--output-dir "results/aravec" \
    	--n-workers $CPUS_PER_GPU \
        --batch-size $BATCH_SIZE \
        --embedding-size $EMBEDDING \
        --data-path data/ArMATH \
        --embedding-type "arvec" \
        --embedding-model-name $PATH_TO_EMBEDDING \
        --arabic \
        --transfer-learning \
        --transfer-learning-model $PATH_TO_CONFIG_FILE \
        --transfer-learning-transfer-encoder \
        --transfer-learning-transfer-decoder
        
```



## Evaluation

To evaluate a model:

```bash
conda activate armath
python code/run.py \
        --config-path $PATH_TO_CONFIG_FILE \
        --evaluate
```



# Citations

```
@InProceedings{alghamdi-liang-zhang:2022:LREC,
  author    = {Alghamdi, Reem  and  Liang, Zhenwen  and  Zhang, Xiangliang},
  title     = {ArMATH: a Dataset for Solving Arabic Math Word Problems},
  booktitle      = {Proceedings of the Language Resources and Evaluation Conference},
  month          = {June},
  year           = {2022},
  address        = {Marseille, France},
  publisher      = {European Language Resources Association},
  pages     = {351--362},
  abstract  = {This paper studies solving Arabic Math Word Problems by deep learning. A Math Word Problem (MWP) is a text description of a mathematical problem that can be solved by deriving a math equation to reach the answer. Effective models have been developed for solving MWPs in English and Chinese. However, Arabic MWPs are rarely studied. This paper contributes the first large-scale dataset for Arabic MWPs, which contains 6,000 samples of primary-school math problems, written in Modern Standard Arabic (MSA). Arabic MWP solvers are then built with deep learning models and evaluated on this dataset. In addition, a transfer learning model is built to let the high-resource Chinese MWP solver promote the performance of the low-resource Arabic MWP solver. This work is the first to use deep learning methods to solve Arabic MWP and the first to use transfer learning to solve MWP across different languages. The transfer learning enhanced solver has an accuracy of 74.15\%, which is 3\% higher than the solver without using transfer learning. We make the dataset and solvers available in public for encouraging more research of Arabic MWPs: https://github.com/reem-codes/ArMATH},
  url       = {https://aclanthology.org/2022.lrec-1.37}
}
```

