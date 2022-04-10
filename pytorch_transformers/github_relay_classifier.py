from github import Github
import sys
sys.path.append('/home/tomyoung/NeuralPipeline_DSTC8/ConvLab/convlab/modules/e2e/multiwoz/Transformer/')
import conversation_mode_classification as cmc
import numpy as np
import torch
from tqdm import tqdm
import json
import time
from datetime import datetime


def post_dialogue(repo, dialogue_id, user_id, time, random_number, dialogue):
    repo.create_file('unprocessed/'+' '.join([str(dialogue_id), str(user_id), str(time), str(random_number)]),
                     "blank message", dialogue)


def post_dialogue_by_path(repo, path, dialogue):
    repo.create_file(path, "blank message", dialogue)


def post_dialogue_labels(repo, dialogue_id, user_id, time, random_number, dialogue_labels):
    repo.create_file('processed/'+' '.join([str(dialogue_id), str(user_id), str(time), str(random_number)]),
                     "blank message", dialogue_labels)


def post_dialogue_labels_by_path(repo, path, dialogue_labels):
    repo.create_file(path, "blank message", dialogue_labels)


def get_dialogue(repo, dialogue_id, user_id, time, random_number):
    contents = repo.get_contents('unprocessed/'+' '.join([str(dialogue_id), str(user_id), str(time), str(random_number)]))
    return contents.decoded_content.decode()


def get_dialogue_by_path(repo, path):
    contents = repo.get_contents(path)
    return contents.decoded_content.decode()


def get_dialogue_labels(repo, dialogue_id, user_id, time, random_number):
    contents = repo.get_contents('processed/'+' '.join([str(dialogue_id), str(user_id), str(time), str(random_number)]))
    return contents.decoded_content.decode()


print('loading classifiers...')
user_mode_classifier = cmc.ModeClassification(
    '/home/tomyoung/NeuralPipeline_DSTC8/ConvLab/convlab/modules/e2e/multiwoz/Transformer/mode_classification'
    '/mdls/cross_mode_single_turn_epoch0.mdl')

sys_mode_classifier = cmc.ModeClassification(
    '/home/tomyoung/NeuralPipeline_DSTC8/ConvLab/convlab/modules/e2e/multiwoz/Transformer/mode_classification'
    '/mdls/permutation_lexicalized_multiwoz_single_is_response_epoch5.mdl')

g = Github("4bb450d7dbc28a95eb74ddae770e9c6221a6aa05")
while True:
    try:
        repo = g.get_user().get_repo('data_collection_relay')
        break
    except:
        print('error 1')
        time.sleep(3)

log_file = open('log file.txt', 'w')

# check the unprocessed directory every 3 seconds
while True:
    time.sleep(3)
    now = datetime.now()
    print("now =", now)
    log_file.write('now = '+str(now))
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print("date and time = ", dt_string)
    log_file.write("date and time = "+str(dt_string))
    while True:
        try:
            contents = repo.get_contents("unprocessed")
            break
        except:
            print('error 2')
            time.sleep(3)
    if len(contents) > 2:
        # read the dialogue files one by one and generate the labels
        for content in contents:  # TODO: take away the placeholder
            if 'placeholder' in content.path or 'done' in content.path:
                continue
            print(content.path)
            while True:
                try:
                    dialogue = get_dialogue_by_path(repo, content.path)
                    break
                except:
                    print('error 3')
                    time.sleep(3)
            print('the dialogue is:')
            print(dialogue)
            new_path = 'unprocessed/done/' + content.path[12:]
            while True:
                try:
                    post_dialogue_by_path(repo, new_path, dialogue)  # post to the 'done' dir
                    break
                except:
                    print('error 4')
                    time.sleep(3)
            while True:
                try:
                    repo.delete_file(content.path, "remove file", content.sha) # TODO delay
                    break
                except:
                    print('error 5')
                    time.sleep(3)
            labels = ''
            dialogue = dialogue.split('\n')
            for turn in dialogue:
                print('turn:')
                print(turn)
                if turn.startswith('user: '):
                    indicator = user_mode_classifier.classify([turn[6:]])
                    labels = labels + str(indicator[0])[7] + '\n'
                elif turn.startswith('system: '):
                    indicator = sys_mode_classifier.classify([turn[8:]])
                    labels = labels + str(indicator[0])[7] + '\n'
                else:
                    print('An error has occurred: the prefix is neither user nor system')
                    print('the dialogue is now stored at: ' + new_path)
                    labels = labels + '0' + '\n'
            print('the labels are:')
            print(labels)
            while True:
                try:
                    post_dialogue_labels_by_path(repo, content.path[2:], labels)
                    break
                except:
                    print('error 6')
                    time.sleep(3)

    # sleep for 10 seconds
    # time.sleep(10)
    # print('I have slept for 10 seconds')







