import os

if not os.path.isdir('./outs'):
    os.mkdir('./outs')

command = "python "  + \
"evaluate_slot_accuracy.py --model_type=\'fused\' " + \
"--fused_ckp_dir=\'runs/fused\' " + \
"--fused_weights_name=\'checkpoint_mymodel_7.pth\' " + \
"--generator_log_dir=\'outs/fused_append.log\' " + \
"--eval_out_path=\'outs/fused_append.out\' " + \
"--predictions_path=\'outs/predictions/fused_append.json\' " + \
"--fused_chat_path=\'data_cache/fused_cache_string_version\' " + \
"--results_path=\'outs/fused_append.results\' " + \
"--partition=\'test\' " + \
"--option=\'append\' " + \
"--device=\'cuda\' " + \
""


os.system(command)
