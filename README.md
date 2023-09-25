# Jurisnext: Next Title Prediction Task

This project contains **Python** code to train and evaluate models for the *Next Title Prediction* (*NTP*) task on legal documents.

To use the code, the original dataset (named 'pre-processed_representations.pkl') is required and should be placed in the 
`data/raw` directory located in the project root.

## Requirements

**Python**: The project has been tested on Python 3.10, support for older/newer versions is not guaranteed.

External libraries have been used and are listed in the `requirements.txt` file in the root of the project.
It is possible to easily install them using **pip**

```console
pip install -U -r requirements.txt
```

Before running the experiments to have reproducibility the following environment variables were used during the experiments 
published in the [report](https://wandb.ai/leshi-cs/BD-Next-Title-Prediction/reports/NTP-Results-Report--Vmlldzo1NDU5Mjky):

```console
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:16:8
```

**IMPORTANT**: The *working directory* when running the project, must be the **root** of the project!

## How to use

The project is equipped with a pipeline that can be called via command line, the console output below shows 
the help section of the pipeline describing all the possible customizable parameters

```console
$ python pipeline.py –h

usage: pipeline.py [-h] [-epo 100] [-t_bs 4] [-e_bs 2] [-seed 42] [-monitor [loss]] -m [bert] [-ck None] [-k_c None] [--log_wandb | --no-log_wandb] [-n_ts 10] [-p_s None]
                   [-t5_t None [None ...]] [-t5_kw_min None] [-ngram None] [-seq_sampling random] [-seq_sampling_start beginning] [-test_seq_sampling None]
                   [-clean_kdws | --clean_stopwords_kwds | --no-clean_stopwords_kwds] [-f_e_m | --freeze_emb_model | --no-freeze_emb_model] [-d cuda:0] [-e None]
                   [-phase None [None ...]]

Main script to run Next Title Prediction experiments

optional arguments:
  -h, --help            show this help message and exit
  -epo 100, --epochs 100
                        Number of epochs for which the Next Title Prediction model will be trained
  -t_bs 4, --train_batch_size 4
                        Batch size that will be used during training
  -e_bs 2, --eval_batch_size 2
                        Batch size that will be used during evaluation (validation and testing)
  -seed 42, --random_seed 42
                        random seed
  -monitor [loss], --monitor_strategy [loss]
                        Choose the strategy used to save the best model. If "loss", the validation loss will be used to save the best model, if "metric", the reference metric        
                        (Accuracy weighted or Hit) will be used to save the best model
  -m [bert], --model [bert]
                        t5 to finetune a t5 checkpoint on several tasks for Next Title Prediction, bert to finetune a bert checkpoint for Next Title Prediction, nli_deberta to       
                        finetune a deberta checkpoint for Next Title Prediction, multimodal to train a multimodal concatenation fusion architecture for Next Title Prediction
  -ck None, --checkpoint None
                        Add checkpoint to use for train (e.g. google/flan-t5-small with t5 model)
  -k_c None, --k_clusters None
                        If specified, it sets the number of clustered labels that will be considered as next possible titles instead of the original labels
  --log_wandb, --no-log_wandb
                        Log pipeline information regarding data, train and eval on wandb (default: False)
  -n_ts 10, --n_test_set 10
                        Specify the number of test set to sample for evaluating the model trained
  -p_s None, --prediction_supporter None
                        Specify the name of the folder in the models directory containing the model to use as prediction supporter
  -t5_t None [None ...], --t5_tasks None [None ...]
                        Specify which train task to use to fine tune NTPT55. If not specified, all possible tasks will be used. The first task specified will be used as validation   
                        task. In the eval phase, all possible tasks are evaluated (apart from BoolNTP which is a support task)
  -t5_kw_min None, --t5_keyword_min_occ None
                        Specify what is the min occurrences that a keyword should have in order to be picked as side info. If not specified, keywords will be sampled randomly        
  -ngram None, --ngram_label None
                        Specify the max number of ngram that a label can have. If None, all ngrams are used
  -seq_sampling random, --seq_sampling_strategy random
                        Specify how sampling is performed on the dataset. "random" will consider randomly picked sequences, "augment" will consider all possible incremental
                        sequences
  -seq_sampling_start beginning, --seq_sampling_start_strategy beginning
                        Specify how the sampled sequence should start. This parameter is ignored if --seq_sampling_strategy is set to "augment". "beginning" will sample the
                        sequence starting always from the first element, "random" will sample also the starting point of the sampled sequence
  -test_seq_sampling None, --test_seq_sampling_strategy None
                        Specify how sampling is performed on the test set. "random" will consider randomly picked sequences, "augment" will consider all possible incremental
                        sequences. If not set, the sampling strategy of train/val will be used. If this parameter is set to "augment" and --n_test_set > 1, a warning is printed and  
                        n_test_set is forced to 1
  -clean_kdws, --clean_stopwords_kwds, --no-clean_stopwords_kwds
                        Specify whether to remove stopwords from the keywords column of the dataset or not (default: False)
  -f_e_m, --freeze_emb_model, --no-freeze_emb_model
                        Used by LSTM encoder, define if the model used for embeddings should be frozen during train or not (default: False)
  -d cuda:0, --device cuda:0
                        Specify the device which should be used during the experiment
  -e None, --exp_name None
                        Specify a custom name for the trained model which will be saved in the "models" dir
  -phase None [None ...], --pipeline_phases None [None ...]
                        If specified, only the selected part(s) of the pipeline are carried out. By default, all phases are performed

```

## Results

Results of the experiments performed can be checked on ***wandb***:

<iframe src="https://wandb.ai/leshi-cs/BD-Next-Title-Prediction/reports/NTP-Results-Report--Vmlldzo1NDU5Mjky" style="border:none;height:1024px;width:100%"></iframe>

Project Organization
------------
    ├── 📁 data                          <- Directory containing all data generated/used
    │   ├── 📁 interim                       <- Intermediate data that has been transformed
    │   ├── 📁 processed                     <- The final, canonical data sets used for training
    │   └── 📁 raw                           <- The original, immutable data dump
    │
    ├── 📁 models                        <- Directory where trained and serialized models will be stored
    │
    ├── 📁 reports                       <- Generated metrics results and plots of the distribution of the data
    │   ├── 📁 data_plots                      
    │   └── 📁 metrics                          
    │
    ├── 📁 src                           <- Source code of the project
    │   ├── 📁 data                          <- Scripts to handle the dataset and split it accordingly
    │   │   └── 📄 legal_dataset.py
    │   │
    │   ├── 📁 evaluation                <- Scripts to evaluate models
    │   │   ├── 📁 ntp_model_eval                <- Scripts containing the function to call for each model to start evaluation
    │   │   ├── 📄 metrics.py                    <- Script containing all metrics classes that can be used for evaluation
    │   │   └── 📄 ntp_evaluator.py              <- Script containing the Evaluator class used by any NTP model evaluation main function
    │   │
    │   ├── 📁 model                     <- Scripts to train models
    │   │   ├── 📁 next_title_prediction         <- Scripts containing the Next Title Prediction
    │   │   │   ├── 📁 ntp_models                    <- Scripts containing the implemented models for the NTP task and their train main function
    │   │   │   ├── 📄 ntp_models_abstract.py        <- Script containing the abstract definition of the models
    │   │   │   └── 📄 ntp_trainer.py                <- Script containing the Trainer class used by any NTP model train main
    │   │   │
    │   │   ├── 📄 clustering.py         <- Scripts containing clustering algorithms
    │   │   └── 📄 sentence_encoders.py  <- Scripts containing sentence encoder models
    │   │
    │   ├── 📄 __init__.py                   <- Makes src a Python module
    │   └── 📄 utils.py                      <- Contains utils function for the project
    │
    ├── 📄 LICENSE                       <- MIT License
    ├── 📄 pipeline.py                   <- Script that can be used to reproduce all NTP experiments
    ├── 📄 README.md                     <- The top-level README for developers using this project
    └── 📄 requirements.txt              <- The requirements file for reproducing the analysis environment (src package)

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>