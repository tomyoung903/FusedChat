'''Evaluate slot accuracy, joint slot accuracy, inform rate, success rate, and BLEU'''
'''
Slot accuracy: how often are slot values correctly predicted
Joint slot accuracy: how often are all slot values in a turn correctly predicted
Inform rate: whether an appropriate entity is provided in the dialogue 
Success rate: and then answers all the requested attributes (Success rate)
BLEU: BLEU score of the generated response against the groundtruth response
'''
import torch
from tqdm import tqdm
import json
import re
from argparse import ArgumentParser
from Generator import Generator
from MultiWOZ_Evaluation.mwzeval import Evaluator
from conversation_mode_classification import ModeClassification
import os

parser = ArgumentParser()
parser.add_argument("--cls_ckp_dir", type=str, required=False,
                    help="checkpoint directory of the model_type classification model")
parser.add_argument("--fused_ckp_dir", type=str, required=False, \
                    help="checkpoint directory of the fused model")
parser.add_argument("--fused_weights_name", type=str, required=False, \
                    default='pytorch_model.bin', \
                    help="weights_name of the fused model")
parser.add_argument("--tod_ckp_dir", type=str, required=False, \
                    help="checkpoint directory of the tod model")
parser.add_argument("--tod_weights_name", type=str, required=False, \
                    default='pytorch_model.bin', \
                    help="weights name of the of the tod model")
parser.add_argument("--chitchat_ckp_dir", type=str, required=False, \
                    default='runs/chitchat_single_nov_2', \
                    help="checkpoint directory")
parser.add_argument("--chitchat_weights_name", type=str, required=False, \
                    default='pytorch_model.bin', \
                    help="weights_name")
parser.add_argument("--model_type", type=str, required=False, \
                    default='classification-based', \
                    help='fused or classification-based')
parser.add_argument("--generator_log_dir", type=str, required=False, \
                    default='outs/generator.log', \
                    help='')
parser.add_argument("--tod_generator_log_dir", type=str, required=False, \
                    default='outs/tod_generator.log', \
                    help='')
parser.add_argument("--chitchat_generator_log_dir", type=str, required=False, \
                    default='outs/chitchat_generator.log', \
                    help='')
parser.add_argument("--eval_out_path", type=str, required=False, \
                    default='outs/eval_out.out', \
                    help='path of the evaluation output file')
parser.add_argument("--predictions_path", type=str, required=False, \
                    default='outs/predictions/nov_11.json', \
                    help='path of the prediction (inform and success) output file')
parser.add_argument("--fused_chat_path", type=str, required=False, \
                    default='data_cache/tod_single_nov_7_id_cache_string_version', \
                    help='path of the fused_chat file (string_version)')
parser.add_argument("--results_path", type=str, required=False, \
                    default='outs/predictions/temp.results', \
                    help='path of the results file')
parser.add_argument("--partition", type=str, required=False, \
                    default='test', \
                    help='partition of the data to be evaluated')
parser.add_argument("--option", type=str, required=False, \
                    default='prepend', \
                    help='prepend, append or both.')
parser.add_argument("--multiple_choice_toleration", type=str, required=False, \
                    default='no')
parser.add_argument("--cls_max_length", type=int, required=False, \
                    default=256)
parser.add_argument("--device", type=str, required=False, \
                    default='cpu')


args = parser.parse_args()

if not os.path.isdir('outs'):
    os.mkdir('outs') 

if not os.path.isdir('outs/predictions'):
    os.mkdir('outs/predictions') 



fused_chat = torch.load(args.fused_chat_path)
if args.model_type == 'fused':
    model = Generator(model_checkpoint=args.fused_ckp_dir, mode=args.model_type, \
        weights_name=args.fused_weights_name, log_dir=args.generator_log_dir, device=args.device)
elif args.model_type == 'classification-based':
    chitchat_model = Generator(model_checkpoint=args.chitchat_ckp_dir, mode='chitchat_single', \
        weights_name=args.chitchat_weights_name, log_dir=args.chitchat_generator_log_dir, device=args.device)
    tod_model = Generator(model_checkpoint=args.tod_ckp_dir, mode='tod_single', \
        weights_name=args.tod_weights_name, log_dir=args.tod_generator_log_dir, device=args.device)
    cls_model = ModeClassification(args.cls_ckp_dir, args.cls_max_length, args.device)


def slot_value_equivalence(groundtruth, prediction):
    '''
        a fast fix for slot value inconsistencies that exist in multiwoz
    '''
    if '|' in groundtruth:
        groundtruth_values = groundtruth.split('|')
        if prediction in groundtruth_values:
            return True
    if groundtruth == 'guest house' and prediction == 'guesthouse':
        return True
    if groundtruth == 'nightclub' and prediction == 'night club':
        return True
    if groundtruth == 'concert hall' and prediction == 'concerthall':
        return True
    if groundtruth == 'museum of archaeology and anthropology' and prediction == 'museum of archaelogy and anthropology':
        return True
    if groundtruth == 'scudamores punting co' and prediction == 'scudamores punters':
        return True
    if groundtruth == 'riverside brasserie' and prediction == 'riveride brasserie':
        return True 
    if groundtruth == 'pizza express fenditton' and prediction == 'pizza hut fenditton':
        return True 
    if groundtruth == 'the slug and lettuce' and prediction == 'slug and lettuce':
        return True
    if groundtruth == 'cafe jello gallery' and prediction == 'jello gallery':
        return True
    if groundtruth == 'alpha milton guest house' and prediction == 'alpha-milton guest house':
        return True  
    if groundtruth == 'city centre north bed and breakfast' and prediction == 'city centre north b and b':
        return True
    if groundtruth == 'portuguese' and prediction == 'portugese':
        return True
    if groundtruth == 'bishops stortford' and prediction == 'bishops strotford':
        return True
    if groundtruth == 'el shaddia guest house' and prediction == 'el shaddia guesthouse':
        return True
    if groundtruth == 'hobsons house' and prediction == 'hobson house':
        return True
    if groundtruth == 'cherry hinton water play' and prediction == 'cherry hinton water park':
        return True    
    if groundtruth == 'centre>west' and prediction == 'centre':
        return True 
    if groundtruth == 'north european' and prediction == 'north european':
        return True
    if groundtruth == 'museum of archaeology and anthropology' and prediction == 'archaelogy and anthropology':
        return True
    if groundtruth == 'riverboat georgina' and prediction == 'the riverboat georgina':
        return True
    if groundtruth == 'grafton hotel restaurant' and prediction == 'graffton hotel restaurant':
        return True
    if groundtruth == 'restaurant one seven' and prediction == 'one seven':
        return True
    if groundtruth == 'arbury lodge guest house' and prediction == 'arbury lodge guesthouse':
        return True
    if groundtruth == 'michaelhouse cafe' and prediction == 'michaelhosue cafe':
        return True
    if groundtruth == 'frankie and bennys' and prediction == "frankie and benny's":
        return True
    if groundtruth == 'london liverpool street' and prediction == 'london liverpoool':
        return True
    if groundtruth == 'the gandhi' and prediction == ' gandhi ':
        return True
    if groundtruth == 'finches bed and breakfast' and prediction == 'flinches bed and breakfast':
        return True
    if groundtruth == 'the cambridge corn exchange' and prediction == 'cambridge corn exchange':
        return True
    if groundtruth == 'broxbourne' and prediction == 'borxbourne':
        return True
    return groundtruth == prediction


def convert_slot_name_to_evaluation_style(response):
    response = response.replace('[attraction_addr]', '[attraction_address]')
    response = response.replace('[hospital_addr]', '[hospital_address]')
    response = response.replace('[hotel_addr]', '[hotel_address]')
    response = response.replace('[police_addr]', '[police_address]')
    response = response.replace('[restaurant_addr]', '[restaurant_address]')
    return response
    

fusedchat_ids_test = [dial['original_id'] for dial in fused_chat['test']]
fusedchat_ids_test = set(fusedchat_ids_test)

eval_out = open(args.eval_out_path, 'w')
total_turns = 0
success_turns = 0
total_slot_predictions = 0
success_slot_predictions = 0
all_slot_types = []

slot_type_value_dict = {
    'attraction': ['area', 'name', 'type'],
    'hotel': ['area', 'day', 'people', 'stay', 'internet', 'name', 'parking', 'pricerange', 'stars', 'type'],
    'restaurant': ['area', 'day', 'people', 'time', 'food', 'name', 'pricerange'],
    'taxi': ['arriveby', 'departure', 'destination', 'leaveat'],
    'train': ['arriveby', 'people', 'day', 'departure', 'destination', 'leaveat'],
    'hospital': ['department']
}


def get_groundtruth_slot_value(metadata_domain, slot_type):
    if slot_type == 'leaveat':
        slot_type = 'leaveAt'
    if slot_type == 'arriveby':
        slot_type = 'arriveBy'
    if slot_type in metadata_domain['book'].keys():
        if metadata_domain['book'][slot_type] == 'not mentioned' or \
                metadata_domain['book'][slot_type] == '':
            return '<nm>'
        if metadata_domain['book'][slot_type] == 'dontcare':
            return '<dc>'
        return metadata_domain['book'][slot_type]
    if slot_type in metadata_domain['semi'].keys():
        if metadata_domain['semi'][slot_type] == 'not mentioned' or \
                metadata_domain['semi'][slot_type] == '':
            return '<nm>'
        if metadata_domain['semi'][slot_type] == 'dontcare':
            return '<dc>'
        return metadata_domain['semi'][slot_type]


NUM_SLOT_TYPES = 31
predictions = {}
partition = args.partition
dataset = fused_chat


for i in tqdm(range(len(dataset[partition]))):
    # if the option is 'prepend', skip the "append" section for fusedchat (the first dialog turn is TOD)
    if args.option == 'prepend':
        if dataset[partition][i]['utterances'][0]['dp'] != ['<chitchat>']:
            continue
    # if the option is 'append', skip the "prepend" section for fusedchat (the first dialog turn is not TOD)
    if args.option == 'append':
        if dataset[partition][i]['utterances'][0]['dp'] == ['<chitchat>']:
            continue
    predictions[dataset[partition][i]['original_id'].lower()] = []
    eval_out.write('original_id:')
    eval_out.write(dataset[partition][i]['original_id'])
    eval_out.write('\n')
    for j in range(len(dataset[partition][i]['utterances'])):
        groundtruth_cs = dataset[partition][i]['utterances'][j]['cs'][0] \
            if dataset[partition][i]['utterances'][j]['cs'] else ''
        # skip the chitchat turns
        if dataset[partition][i]['utterances'][j]['dp'] == ['<chitchat>']:
            continue
        dialog_meta = dataset[partition][i]['utterances'][j]['dialog_meta'][0] \
            if dataset[partition][i]['utterances'][j]['dialog_meta'] else ''
        history = dataset[partition][i]['utterances'][j]['history']
        turn_success = 1
        total_turns += 1

        if args.model_type == 'fused':
            cs_dict, response = model.infer_cs_and_response(history)
        elif args.model_type == 'classification-based':
            mode_label = cls_model.classify(history)
            if mode_label == 1:
                eval_out.write('conversation_mode: TOD\n')
                cs_dict, response = tod_model.infer_cs_and_response(history)
                lexicalized_response = tod_model.infer(history)
            else:
                eval_out.write('conversation_mode: chitchat\n')
                response = chitchat_model.infer(history)
                cs_dict = {}
        eval_out.write('history:\n')
        for turn_in_history in history:
            eval_out.write(turn_in_history)
            eval_out.write('\n')
        eval_out.write('dialog_meta:')
        eval_out.write(str(dialog_meta))
        eval_out.write('\n')
        eval_out.write('cs_dict:')
        eval_out.write(str(cs_dict))
        eval_out.write('\n')
        eval_out.write('response:')
        eval_out.write(str(response))
        eval_out.write('\n')
        total_slot_predictions += NUM_SLOT_TYPES
        if dialog_meta:  # the groundtruth dialogue state is not empty
            # assuming it always predicts single domain and never gets the domain wrong
            domain = re.findall(r"<.*?>", groundtruth_cs)[0][1:-1]
            for key in slot_type_value_dict.keys():
                for slot_type in slot_type_value_dict[key]:
                    if domain == key:
                        if slot_type in cs_dict.keys():
                            predicted_slot_value = cs_dict[slot_type]
                        else:
                            predicted_slot_value = '<nm>'
                        groundtruth_value = get_groundtruth_slot_value(dialog_meta[domain], slot_type)
                        if slot_value_equivalence(groundtruth_value, predicted_slot_value):
                            success_slot_predictions += 1
                        else:
                            eval_out.write('error case:\n')
                            eval_out.write('groundtruth:\n')
                            eval_out.write(str(groundtruth_value))
                            eval_out.write('\n')
                            eval_out.write('predicted:\n')
                            eval_out.write("<" + slot_type + ">" + ' ' + predicted_slot_value + ' ')
                            eval_out.write('\n')
                            turn_success = 0
                    else:
                        success_slot_predictions += 1
        else:  # if the groundtruth dialogue state is empty, all predicted slot values are considered error cases
            if cs_dict:
                turn_success = 0
                success_slot_predictions += NUM_SLOT_TYPES
                for key, value in cs_dict.items():
                    if value != '<nm>':
                        eval_out.write('error case:\n')
                        eval_out.write('groundtruth:\n')
                        eval_out.write('empty')
                        eval_out.write('\n')
                        eval_out.write('predicted:\n')
                        eval_out.write("<" + key + ">" + ' ' + value + ' ')
                        eval_out.write('\n')
                        success_slot_predictions = success_slot_predictions - 1
            else:
                turn_success = 1
                success_slot_predictions += NUM_SLOT_TYPES
        success_turns = success_turns + turn_success
        eval_out.write('\n\n')
        turn_predictions = {}
        turn_predictions['response'] = convert_slot_name_to_evaluation_style(response)
        turn_predictions['state'] = {}
        if 'domain' in locals():
            turn_predictions['state'][domain] = {key:value for key, value in cs_dict.items() if (value != '<nm>' and value != '<dc>')}
            turn_predictions['active_domains'] = [domain]
        predictions[dataset[partition][i]['original_id'].lower()].append(turn_predictions)

with open(args.predictions_path, 'w') as f:
    json.dump(predictions, f)

if args.multiple_choice_toleration == 'yes':
    args.mct = True
else:
    args.mct = False

e = Evaluator(bleu=True, success=True, richness=True, mct=args.mct)
results = e.evaluate(predictions)
with open(args.results_path, 'w') as f:
    json.dump(results, f)

print('total_turns: %d' % total_turns)
print('success_turns: %d' % success_turns)
print('joint accuracy: %f' % (success_turns / total_turns))
print('total_slot_predictions: %d' % total_slot_predictions)
print('success_slot_predictions: %d' % success_slot_predictions)
print('slot accuracy: %f' % (success_slot_predictions / total_slot_predictions))

eval_out.write('bleu|inform|success|richness: %s\n' % results)
eval_out.write('total_turns: %d\n' % total_turns)
eval_out.write('success_turns: %d\n' % success_turns)
eval_out.write('joint accuracy: %f\n' % (success_turns / total_turns))
eval_out.write('total_slot_predictions: %d\n' % total_slot_predictions)
eval_out.write('success_slot_predictions: %d\n' % success_slot_predictions)
eval_out.write('slot accuracy: %f\n' % (success_slot_predictions / total_slot_predictions))
args_dict = vars(args)
args_dict_string = {key:str(value) for key, value in args_dict.items()}
eval_out.write('evaluation setting args: %s\n' % args_dict_string)
eval_out.close()

