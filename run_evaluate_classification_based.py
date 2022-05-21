import os

if not os.path.isdir('./outs'):
    os.mkdir('./outs')
command = "python "  + \
"evaluate_slot_accuracy.py --model_type=\'classification-based\' "  + \
"--cls_ckp_dir=\'cls_models/multi_turn.mdl\' " + \
"--tod_ckp_dir=\'runs/tod_single\' " + \
"--tod_weights_name=\'checkpoint_mymodel_7.pth\' " + \
"--tod_generator_log_dir=\'outs/tod_single.log\' " + \
"--chitchat_ckp_dir=\'runs/chitchat_single\' " + \
"--chitchat_weights_name=\'checkpoint_mymodel_5.pth\' " + \
"--chitchat_generator_log_dir=\'outs/chitchat_single.log\' " + \
"--eval_out_path=\'outs/cls-based.out\' " + \
"--predictions_path=\'outs/predictions/cls-based.json\' " + \
"--fused_chat_path=\'data_cache/tod_single_cache_string_version\' " + \
"--results_path=\'outs/cls-based.results\' " + \
"--partition=\'test\' " + \
"--option=\'both\' " + \
"--device=\'cuda\' " + \
""

os.system(command)


