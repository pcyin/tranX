# TranX

A general-purpose **Tran**sition-based abstract synta**X** parser 
that maps natural language queries into machine executable 
source code (e.g., Python) or logical forms (e.g., lambda calculus). **[Online Demo](http://moto.clab.cs.cmu.edu:8081/)**.

## System Architecture

For technical details please refer to our [ACL '18 paper](https://arxiv.org/abs/1806.07832) and [EMNLP '18 demo paper](https://arxiv.org/abs/1810.02720). 
To cope with different 
domain specific logical formalisms (e.g., SQL, Python, lambda-calculus, 
prolog, etc.), TranX uses abstract syntax trees (ASTs) defined in the 
Abstract Syntax Description Language (ASDL) as intermediate meaning
representation.

![Sysmte Architecture](doc/system.png)

Figure 1 gives a brief overview of the system.

1. TranX first employs a transition system to transform a natural language utterance into a sequence of tree-constructing actions, following the input grammar specification of the target formal language. The grammar specification is provided by users in textual format (e.g., `asdl/lang/py_asdl.txt` for Python grammar).

2. The tree-constructing actions produce an intermediate abstract syntax tree. TranX uses ASTs defined under the ASDL formalism as general-purpose, intermediate meaning representations.

3. The intermediate AST is finally transformed to a domain-specific representation (e.g., Python source code) using customly-defined conversion functions.

**File Structure** tranX is mainly composed of two components: 

1. A general-purpose transition system that defines the generation process of an AST `z`
 using a sequence of tree-constructing actions `a_0, a_1, ..., a_T`.
2. A neural network that computes the probability distribution over action sequences, conditional on the natural language query `x`, `p(a_0, a_1, ..., a_T | x)`.

These two components are implemented in the following two folders, respectively:

* `asdl` defines a general-purpose transition system based on the ASDL formalism, and its instantiations in different programming languages and datasets. The transition system defines how an AST is constructed using a sequence of actions. This package can be used as a standalone library independent of tranX. See Section 2.2 of the technical report for details.

* `model` contains the neural network implementation of the transition system defined in `asdl`, which computes action probabilities using neural networks.See Section 2.3 of the technical report for details.

Here is a detailed map of the file strcuture:
```bash
├── asdl (grammar-based transition system)
├── datasets (dataset specific code like data preprocessing/evaluation/etc.)
├── model (PyTorch implementation of neural nets)
├── server (interactive Web server)
├── components (helper functions and classes like vocabulary)
```

## Supported Language and Datasets

TranX officially supports the following grammatical formalism and datasets.
More languages (C#) are coming! 

Language | Transition System | Grammar Specification | Example Datasets
---------|--------------------| -------- | -------- 
Python 2   | `asdl.PythonTransitionSystem` | `asdl/lang/py/py_asdl.txt` | Django (Oda et al., 2015)
Python 3 | `asdl.Python3TransitionSystem` | `asdl/lang/py3/py3_asdl.simplified.txt` | CoNaLa (Yin et al., 2018) 
Lambda Calculus| `asdl.LambdaCalculusTransitionSystem` | `asdl/lang/lambda_asdl.txt` | ATIS, GeoQuery (Zettlemoyer and Collins, 2005)
Prolog | `asdl.PrologTransitionSystem` | `asdl/lang/prolog_asdl.txt`  | Jobs (Zettlemoyer and Collins, 2005)
SQL | `asdl.SqlTransitionSystem` | `asdl/lang/sql/sql_asdl.txt` | WikiSQL (Zhong et al., 2017)

### Evaluation Results

Here is a list of performance results on six datasets using pretrained models in `data/pretrained_models`

| Dataset | Results      | Metric             |
| ------- | ------------ | ------------------ |
| GEO     | 88.6         | Accuracy           |
| ATIS    | 87.7         | Accuracy           |
| JOBS    | 90.0         | Accuracy           |
| Django  | 77.2         | Accuracy           |
| CoNaLa  | 24.5         | Corpus BLEU        |
| WikiSQL | 79.1         | Execution Accuracy |


## Usage


### TL;DR

```bash
git clone https://github.com/pcyin/tranX
cd tranX

bash ./pull_data.sh  # get datasets and pre-trained models

conda env create -f config/env/tranx.yml  # create conda Python environment.

./scripts/atis/train.sh 0  # train on ATIS semantic parsing dataset with random seed 0
./scripts/geo/train.sh 0  # train on GEO dataset
./scripts/django/train.sh 0  # train on django code generation dataset
./scripts/conala/train.sh 0  # train on CoNaLa code generation dataset
./scripts/wikisql/train.sh 0  # train on WikiSQL SQL code generation dataset
```

### Web Server/HTTP API

`tranX` also ships with a web server for demonstraction and interactive debugging perpuse. It also exposes an HTTP API for online semantic parsing/code generation.


To start the web server, simply run:

```
source activate tranx
PYTHONPATH=. python server/app.py --config_file config/server/config_py3.json
```

This will start a web server at port 8081 with ATIS/GEO/CoNaLa datasets.



**HTTP API** To programmically query `tranX` to get semantic parsing results, send your HTTP GET request to

```
http://<IP Address>:8081/parse/<dataset_name>/<utterance>

# e.g., http://localhost:8081/parse/atis/show me flight from Pittsburgh to Seattle
```



### Conda Environments

TranX supports both Python 2.7 and 3.5. Please note that 
some datasets only support Python 2.7 (e.g., Django) or Python 3+ (e.g., WikiSQL).
The main example conda environment (`config/env/tranx.yml`) supports Python 3, but
we also provide one for Python 2 (`config/env/tranx-py2.yml`).
You can export the enviroments using the following command:

```bash
conda env create -f config/env/(tranx.yml,tranx-py2.yml)
```

## FAQs

#### How to adapt to a new programming language or logical form?

You need to implement the 
`TransitionSystem` class with a bunch of custom functions which (1) convert between 
domain-specific logical forms and intermediate ASTs used by TranX, (2) predictors which 
check if a hypothesis parse if correct during beam search decoding.
You may take a look at the examples in `asdl/lang/*`.

#### How to generate those pickled datasets (.bin files)?

Please refer to `datasets/<lang>/dataset.py` for code snippets that converts 
a dataset into pickled files. 

#### How to run without CUDA?

Simply remove the `--cuda`` flag from the command line arguments. It is included
by default in all scripts in the `scripts` directory.

## Reference

TranX is described/used in the following two papers:

```
@inproceedings{yin18emnlpdemo,
    title = {{TRANX}: A Transition-based Neural Abstract Syntax Parser for Semantic Parsing and Code Generation},
    author = {Pengcheng Yin and Graham Neubig},
    booktitle = {Conference on Empirical Methods in Natural Language Processing (EMNLP) Demo Track},
    year = {2018}
}

@inproceedings{yin18acl,
    title = {Struct{VAE}: Tree-structured Latent Variable Models for Semi-supervised Semantic Parsing},
    author = {Pengcheng Yin and Chunting Zhou and Junxian He and Graham Neubig},
    booktitle = {The 56th Annual Meeting of the Association for Computational Linguistics (ACL)},
    url = {https://arxiv.org/abs/1806.07832v1},
    year = {2018}
}
```

## Thanks

We are also grateful to the following papers that inspire this work :P
```
Abstract Syntax Networks for Code Generation and Semantic Parsing.
Maxim Rabinovich, Mitchell Stern, Dan Klein.
in Proceedings of the Annual Meeting of the Association for Computational Linguistics, 2017

The Zephyr Abstract Syntax Description Language.
Daniel C. Wang, Andrew W. Appel, Jeff L. Korn, and Christopher S. Serra.
in Proceedings of the Conference on Domain-Specific Languages, 1997
```

We also thank [Li Dong](http://homepages.inf.ed.ac.uk/s1478528/) for all the helpful discussions and sharing the data-preprocessing code for ATIS and GEO used in our Web Demo.
