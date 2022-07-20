# Overview
FusedChat is an inter-mode dialogue dataset. It contains dialogue sessions fusing task-oriented dialogues (TOD) and open-domain dialogues (ODD). Based on [MultiWOZ](https://github.com/smartyfh/MultiWOZ2.4), FusedChat appends or prepends an ODD to every existing TOD. See more details in the [paper](https://arxiv.org/pdf/2109.04137.pdf).

# Updates

**09/19/2021** Dataset released.

**02/23/2022** Dataset was further augmented and reorganized.

**04/10/2022** Added author-trained checkpoints, baseline code and evaluation code.



# Code

## Context classification models

**run_prepare_classification_data.py** Prepare context classification data.

set `--context_type` to `last_turn` or `multi_turn` to generate the last-turn or multi-turn data respectively.

**run_train_context_classification_model.py** Train the cross-encoder-based classifier.

**run_test_context_classification_model.py** Test the cross-encoder-based classifier.



## Response generation models

You have to generate the data first using the 3 scripts below before evaluation. We overloaded the training scripts with data generation purposes. Each mode has its own data format.

**run_train_tod_single.py** Train the TOD (single mode) model. This model is trained on FusedChat data where the response is in the TOD mode. Setting only_generating_data to 'yes' will only generate the data (tokenized dataset and tensor cache).

**run_train_chitchat_single.py** Train the chitchat (or ODD, single mode) model. This model is trained on FusedChat data where the response is in the ODD mode. Setting only_generating_data to 'yes' will only generate the data (tokenized dataset and tensor cache).

**run_train_fused.py** Train the fused model. This model is trained on all FusedChat data. Setting only_generating_data to 'yes' will only generate the data (tokenized dataset and tensor cache).



**run_evaluate_classification_based.py** Evaluate the classification-based response generation models.

**run_evaluate_fused.py** Evaluate the two-in-one response generation models.

**run_evaluate_ppl_classification_based.py** Evaluate perplexity in a mode-aware manner. Specifically, the negative log-likelihood of each token is modified by the probablity of determining the correct mode of the response (according to the classifier).

**run_evaluate_ppl_fused.py** Evaluate perplexity in a mode-aware manner. Specifically, the negative log-likelihood of each token is modified by the probablity of determining the correct mode of the response  (according to token generation).

# Author-trained checkpoints
Download the following checkpoint files [here](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/QWEBOS).

(1) fused.zip
  checkpoint file for the fused (two-in-one) model. Put under runs/.

(2) tod_single.zip
  checkpoint file for the TOD (single mode) model. Put under runs/.

(3) chitchat_single.zip
  checkpoint file for the ODD (chitchat, single mode) model. Put under runs/.

(4) last_turn.mdl
  checkpoint file for the context classification model (last-turn) model. Put under cls_models/.

(5) multi_turn.mdl
  checkpoint file for the context classification model (multi-turn) model. Put under cls_models/.


# References
```
@article{young2021fusing,
  title={Fusing task-oriented and open-domain dialogues in conversational agents},
  author={Young, Tom and Xing, Frank and Pandelea, Vlad and Ni, Jinjie and Cambria, Erik},
  journal={arXiv preprint arXiv:2109.04137},
  year={2021}
}
```

# Baseline Performance


## Mode classification accuracy



<table>
<thead>
  <tr>
    <th>Context option</th>
    <th>Accuracy</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Single-turn</td>
    <td>0.993</td>
  </tr>
  <tr>
    <td>Multi-turn</td>
    <td>0.995</td>
  </tr>
</tbody>
</table>

## Inter-mode dialogue evaluation (on full FusedChat testset)

<table class="tg">
<thead>
  <tr>
    <th class="tg-c3ow" rowspan="2"><br>Models</th>
    <th class="tg-c3ow" colspan="7">TOD metrics</th>
    <th class="tg-c3ow" colspan="4">ODD metrics</th>
  </tr>
  <tr>
    <th class="tg-c3ow">Slot Accuracy</th>
    <th class="tg-c3ow">Joint SA</th>
    <th class="tg-c3ow">Inform</th>
    <th class="tg-baqh">Inform_mct</th>
    <th class="tg-c3ow">Success</th>
    <th class="tg-baqh">Success_mct</th>
    <th class="tg-c3ow">BLEU</th>
    <th class="tg-c3ow">PPL</th>
    <th class="tg-c3ow">Sensibleness</th>
    <th class="tg-c3ow">Specificity</th>
    <th class="tg-c3ow">SSA</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-c3ow">Two-in-one model</td>
    <td class="tg-c3ow">0.972</td>
    <td class="tg-c3ow">0.592</td>
    <td class="tg-c3ow">70.4</td>
    <td class="tg-baqh">90.1</td>
    <td class="tg-c3ow">57.0</td>
    <td class="tg-baqh">72.7</td>
    <td class="tg-c3ow">12.05</td>
    <td class="tg-c3ow">10.49</td>
    <td class="tg-c3ow">0.52</td>
    <td class="tg-c3ow">0.47</td>
    <td class="tg-c3ow">0.50</td>
  </tr>
  <tr>
    <td class="tg-c3ow">Classification-based model</td>
    <td class="tg-c3ow">0.973</td>
    <td class="tg-c3ow">0.600</td>
    <td class="tg-c3ow">75.1</td>
    <td class="tg-baqh">90.8</td>
    <td class="tg-c3ow">60.9</td>
    <td class="tg-baqh">74.4</td>
    <td class="tg-c3ow">12.17</td>
    <td class="tg-c3ow">10.50</td>
    <td class="tg-c3ow">0.58</td>
    <td class="tg-c3ow">0.51</td>
    <td class="tg-c3ow">0.55</td>
  </tr>
</tbody>
</table>

Here we additionally report inform_mct and success_mct. MCT stands for "multi-choice tolerant". "multi-choice tolerant" evaluation ignores the requisite of generating entity names.  We think this may better measure the accuracy of the model because sometimes the model may choose to ask for additional restraints, instead of directly providing a recommendation, as shown in the example below.

user: I need some time in the sun, can you help me find a park to visit?

system (groundtruth): Cherry Hinton Water Play is in the east and is free admission.

system (model): Yes I have several parks in the city. What area are you looking for?

Under traditional evaluation, the model's response is considered a failure because the entity's name is never mentioned. However, it recognized the dialogue state correctly and the dialogue flow is normal. Under MCT evaluation, the model's response is considered a success because explicitly mentioning an entity name is no longer considered a requisite.

# Credits
We would like to thank the numerous creators who contributed to FusedChat. We thank Lu Cheng for creating the data collection interface and Low Shi Min and Arya Shashwat for quality control. We thank Peter Young for communication with the creators and dialogue assignment. The code for baselines is based on [NeuralPipeline](https://github.com/KAIST-AILab/NeuralPipeline_DSTC8). The code for evaluation is [MultiWOZ_Evaluation](https://github.com/Tomiinek/MultiWOZ_Evaluation).

