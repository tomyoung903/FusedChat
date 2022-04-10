import os

if 'context_cls_results' not in os.listdir('.'):
    os.mkdir('context_cls_results')


command = "python "  + \
"context_classification.py "  + \
"--context_type=\'last_turn\' "  + \
"--mode=\'test\' "  + \
"--model_name_load=\'last_turn\' "  + \
"--results_filename=\'context_cls_results/last_turn.txt\' "  + \
""

os.system(command)


