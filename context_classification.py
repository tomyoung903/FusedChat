import numpy as np
import torch
from pytorch_pretrained_bert import BertTokenizer
from cross_base import BertEncoder
from argparse import ArgumentParser
import os


parser = ArgumentParser()
parser.add_argument("--bert_model_type", type=str, required=False, 
                    default="bert-base-uncased", help="the type of the bert model")
parser.add_argument("--context_type", type=str, required=False, 
                        default="last_turn", help="the type of the context")
parser.add_argument("--training_batch_size", type=int, required=False, default=4,
                        help="batch size using for training")
parser.add_argument("--testing_batch_size", type=int, required=False, default=1,
                        help="batch size using for testing")
parser.add_argument("--max_epoch", type=int, required=False, default=10,
                        help="maximum epoch")
parser.add_argument("--mode", type=str, required=False, default='train',
                        help="the mode used to run this script")
parser.add_argument("--results_filename", type=str, required=False, default='feb_18.txt',
                        help="filename for the results")
parser.add_argument("--model_name_save", type=str, required=False, default='feb_18',
                        help="name of the model file to save into; used in train mode")
parser.add_argument("--model_name_load", type=str, required=False, default='feb_18',
                        help="name of the model file to load; used in test mode")


args = parser.parse_args()



filenames = {}
npys = {}
for data_type in ['context_ids', 'context_masks', 'labels']:
    filenames[data_type] = {}
    npys[data_type] = {}
    for partition in ['train', 'val', 'test']:
        # assumes the filenames are in the same format as in prepare_classification_data.py
        filenames[data_type][partition] = 'npys/' + data_type + '_' + args.context_type + '_' + partition + '.npy'
        npys[data_type][partition] = np.load(filenames[data_type][partition])



tokenizer = BertTokenizer.from_pretrained(args.bert_model_type)
model = BertEncoder(768, 2, args.bert_model_type)
model = model.cuda()
results = open(args.results_filename, 'w')


if args.mode == 'train':
    if not os.path.exists('cls_models/'):
        os.mkdir('cls_models/')
    model_name = 'cls_models/' + args.model_name_save


    for epoch in range(args.max_epoch):
        model.train_model(npys['context_ids']['train'],
                          npys['labels']['train'],
                          npys['context_masks']['train'],
                          args.training_batch_size)
        torch.save(model.state_dict(), model_name + '_epoch_' + str(epoch) + '.mdl')

        train_acc, report, gts, predictions = model.test(npys['context_ids']['train'],
                                                         npys['labels']['train'],
                                                         npys['context_masks']['train'],
                                                         args.testing_batch_size)
        print('Training Accuracy: ' + str(train_acc) + ' epoch ' + str(epoch) + '\n')
        results.write('Training Accuracy: ' + str(train_acc) + ' epoch ' + str(epoch) + '\n')


        valid_acc, report, gts, predictions = model.test(npys['context_ids']['val'],
                                                         npys['labels']['val'],
                                                         npys['context_masks']['val'],
                                                         args.testing_batch_size)
        print('Validation Accuracy: ' + str(valid_acc) + ' epoch ' + str(epoch) + '\n')
        results.write('Validation Accuracy: ' + str(valid_acc) + ' epoch ' + str(epoch) + '\n')


        test_acc, report, gts, predictions = model.test(npys['context_ids']['test'],
                                                        npys['labels']['test'],
                                                        npys['context_masks']['test'],
                                                        args.testing_batch_size)
        print('Testing Accuracy: ' + str(test_acc) + ' epoch ' + str(epoch) + '\n')
        results.write('Testing Accuracy: ' + str(test_acc) + ' epoch ' + str(epoch) + '\n')
    results.close()

if args.mode == 'test':
    model_checkpoint = 'cls_models/' + args.model_name_load + '.mdl'
    model.load_state_dict(torch.load(model_checkpoint))
    test_acc, report, gts, predictions = model.test(npys['context_ids']['test'],
                                                    npys['labels']['test'],
                                                    npys['context_masks']['test'],
                                                    args.testing_batch_size)
    results.write('Testing Accuracy: ' + str(test_acc) + '\n')
    results.close()
