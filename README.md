# BERT-DST

The code has been tested with Python 3 and PyTorch 1.5.0. Note that the code in the folder `pytorch_pretrained_bert` was originally from the [Hugging Face team](https://github.com/huggingface). With minor modifications, you can use the latest version of [huggingface/transformers](https://github.com/huggingface/transformers).

## Commands
An example training command (using BERT-Base) is:
`python main.py --do_train --data_dir=data/woz/ --bert_model=bert-base-uncased --output_dir=outputs`

An example training command (using BERT-Large) is:
`python main.py --do_train --data_dir=data/woz/ --bert_model=bert-large-uncased --output_dir=outputs`

## Results

The table below shows the results on the WoZ restaurant reservation datasets.

Model | Joint Goal (WoZ) | Turn Request (WoZ)|
:---: |:---: | :---: |
Neural Belief Tracker - DNN | 84.4% | 91.2% |
Neural Belief Tracker - CNN | 84.2% | 91.6% |
GLAD | 88.1 ± 0.4% | 97.1 ± 0.2% |
*Simple BERT Model* (BERT-Base) | 90.5% | 97.6% |

## Simple BERT Model

The figure below shows the architecture of the simple BERT Model.

<p align="center">
<img src="https://github.com/laituan245/BERT-Dialog-State-Tracking/blob/master/images/simple_bert_model_for_dst.png" width="80%">
</p>

Please cite our related paper [A Simple but Effective BERT Model for Dialog State Tracking on Resource-Limited Systems](https://ieeexplore.ieee.org/document/9053975) if you find this useful.
