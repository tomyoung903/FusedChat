import os

if 'data_cache' not in os.listdir('.'):
    os.mkdir('data_cache')

if 'runs' not in os.listdir('.'):
    os.mkdir('runs')

command = "python "  + \
"train.py --mode=\'chitchat_single\' "  + \
"--model_checkpoint=\'gpt2\' "  + \
"--prepend_dataset_path=\'./data/prepended_delexicalized.json\' "  + \
"--append_dataset_path=\'./data/appended_delexicalized.json\' "  + \
"--lexicalized_prepend_dataset_path=\'./data/prepended_lexicalized.json\' "  + \
"--lexicalized_append_dataset_path=\'./data/appended_lexicalized.json\' "  + \
"--ckp_dir_name=\'runs/chitchat_single\' " + \
"--fusedchat_cache=\'./data_cache/chitchat_single_cache\' "  + \
"--tensor_dataset_cache=\'./data_cache/chitchat_single_tensor_cache\' "  + \
"--n_epochs=10 " + \
"--device=\'cuda\' " + \
""

os.system(command)
