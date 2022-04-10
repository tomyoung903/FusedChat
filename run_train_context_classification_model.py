import os

if 'context_cls_results' not in os.listdir('.'):
    os.mkdir('context_cls_results')

command = "python "  + \
"context_classification.py "  + \
"--context_type=\'last_turn\' "  + \
"--mode=\'train\' "  + \
"--model_name_save=\'mar_10\' "  + \
"--results_filename=\'context_cls_results/mar_10.txt\' "  + \
""

os.system(command)
