import os

if not os.path.isdir('./outs/ppl'):
    os.mkdir('./outs/ppl')

command = "python "  + \
"evaluate_ppl.py "  + \
"--tensor_cache_for_test=\'./data_cache/chitchat_single_tensor_cache_test\' " + \
"--model_checkpoint=\'runs/chitchat_single\' " + \
"--weights_name=\'checkpoint_mymodel_5.pth\' " + \
"--eval_out=\'outs/ppl/chitchat_single.txt\' " + \
"--cls_model_checkpoint=\'cls_models/multi_turn.mdl\' " + \
"--test_batch_size=1 " + \
"--device=\'cuda\' " + \
""

os.system(command)




