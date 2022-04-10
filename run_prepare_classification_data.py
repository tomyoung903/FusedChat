import os

command = "python "  + \
"prepare_classification_data.py --bert_model_type=\'bert-base-uncased\' "  + \
"--max_length=256 " + \
"--context_type=\'last_turn\' " + \
""

os.system(command)
