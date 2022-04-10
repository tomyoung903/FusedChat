import os

if not os.path.isdir('./outs/ppl'):
    os.mkdir('./outs/ppl')


command = "python "  + \
"evaluate_ppl.py "  + \
"--tensor_cache_for_test=\'./data_cache/fused_tensor_cache_test\' " + \
"--model_checkpoint=\'runs/fused\' " + \
"--weights_name=\'checkpoint_mymodel_7.pth\' " + \
"--eval_out=\'outs/ppl/fused.txt\' " + \
"--test_batch_size=1 " + \
"--device=\'cuda\' " + \
"--type_of_system=\'fused\' " + \
""


os.system(command)




