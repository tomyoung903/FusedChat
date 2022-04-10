import os
import math
import logging
import sys
from pprint import pformat
from argparse import ArgumentParser
from collections import defaultdict
from itertools import chain
from tokenize import String
import numpy
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Accuracy, Loss, MetricsLambda, RunningAverage
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler

from pytorch_transformers import GPT2DoubleHeadsModel, GPT2Tokenizer, AdamW, \
    GPT2DoubleHeadsAdapterModel

from util import SPECIAL_TOKENS_ORIGINAL, \
    SPECIAL_TOKENS_plus_chitchat_sor, SPECIAL_TOKENS_chitchat_single, act_name, get_fusedchat, slot_name, \
    build_input_from_segments

import json
from tqdm import tqdm
        
CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"
MODEL_INPUTS = ["input_ids", "mc_token_ids", "lm_labels", "mc_labels", "token_type_ids", 'task_ids']
PADDED_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]

logger = logging.getLogger(__file__)
logger.setLevel(logging.WARNING)


def average_distributed_scalar(scalar, args):
    """ Average a scalar over the nodes if we are in distributed training. We use this for distributed evaluation. """
    if args.local_rank == -1:
        return scalar
    scalar_t = torch.tensor(scalar, dtype=torch.float, device=args.device) / torch.distributed.get_world_size()
    torch.distributed.all_reduce(scalar_t, op=torch.distributed.ReduceOp.SUM)
    return scalar_t.item()


def pad_dataset(dataset, padding=0):
    """ Pad the dataset. This could be optimized by defining a
    Dataset class and padd only batches but this is simpler. """
    max_l = max(len(x) for x in dataset["input_ids"])
    for name in PADDED_INPUTS:
        dataset[name] = [x + [padding if name != "lm_labels" else -1] * (max_l - len(x)) for x in dataset[name]]
    return dataset


def get_data_loaders_from_cache(args):
    # load the tensor datasets from cache
    tensor_datasets = {"train": torch.load(args.tensor_dataset_cache + '_train'),
                       "valid": torch.load(args.tensor_dataset_cache + '_valid')}

    train_dataset, valid_dataset = TensorDataset(*tensor_datasets["train"]), \
                                   TensorDataset(*tensor_datasets["valid"])
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset) if args.distributed else None
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                              shuffle=(not args.distributed))
    valid_loader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=args.valid_batch_size, shuffle=False)
    logger.info("Train dataset (Batch, Candidates, Seq length): {}".format(train_dataset.tensors[0].shape))
    logger.info("Valid dataset (Batch, Candidates, Seq length): {}".format(valid_dataset.tensors[0].shape))
    return train_loader, valid_loader, train_sampler, valid_sampler


def get_data_loaders(args, tokenizer):
    '''load the tensor datasets'''

    # try to find the cache first
    if os.path.isfile(args.tensor_dataset_cache + '_train'):
        return get_data_loaders_from_cache(args)
    
    datasets = {"train": defaultdict(list), 
                "valid": defaultdict(list), 
                "test": defaultdict(list)}
    
    fusedchat = get_fusedchat(
        tokenizer, 
        args.prepend_dataset_path,
        args.append_dataset_path,
        args.fusedchat_cache, 
        position='both', 
        lexicalized_prepend_dataset_path = args.lexicalized_prepend_dataset_path,
        lexicalized_append_dataset_path = args.lexicalized_append_dataset_path)
    
    for dataset_name, dataset in fusedchat.items():
        num_candidates = len(dataset[0]["utterances"][0]["candidates"])
        if args.num_candidates > 0 and dataset_name == 'train':
            num_candidates = min(args.num_candidates, num_candidates)
        for dialog in tqdm(dataset):
            for utterance in dialog["utterances"]:
                # for single mode, we only train on the respective instances
                logger.info(utterance["dp"][0])
                logger.info(tokenizer.convert_tokens_to_ids(tokenizer.tokenize('<chitchat>')))
                if args.mode == 'tod_single' and utterance["dp"][0] == tokenizer.convert_tokens_to_ids(tokenizer.tokenize('<chitchat>')):
                    logger.info('instance ditched because the mode is different')
                    continue
                if args.mode == 'chitchat_single' and utterance["dp"][0] != tokenizer.convert_tokens_to_ids(tokenizer.tokenize('<chitchat>')):
                    logger.info('instance ditched because the mode is different')
                    continue
                dp = utterance["dp"][0]
                cs = utterance["cs"]
                cs = cs[0] if cs else cs
                history = utterance["history"][-(2 * args.max_history + 1):]
                for j, candidate in enumerate(utterance["candidates"][-num_candidates:]):
                    lm_labels = bool(j == num_candidates - 1)
                    if args.mode == 'fused':
                        if dp != tokenizer.convert_tokens_to_ids(tokenizer.tokenize('<chitchat>')):
                            instance, _ = build_input_from_segments(history, candidate, tokenizer, dp, cs,
                                                                    lm_labels, model=args.model_checkpoint,
                                                                    skill_mode='tod_double')
                        else:
                            instance, _ = build_input_from_segments(history, candidate, tokenizer,
                                                                    [], [], lm_labels,
                                                                    model=args.model_checkpoint,
                                                                    skill_mode='chitchat_double')
                    elif args.mode == 'tod_single':
                        instance, _ = build_input_from_segments(history, candidate, tokenizer, dp, cs,
                                                                    lm_labels, model=args.model_checkpoint,
                                                                    skill_mode='tod_single')
                    elif args.mode == 'chitchat_single':
                        instance, _ = build_input_from_segments(history, candidate, tokenizer,
                                                                    [], [], lm_labels,
                                                                    model=args.model_checkpoint,
                                                                    skill_mode='chitchat_single')
                    elif args.mode == 'adapter':
                        if dp != tokenizer.convert_tokens_to_ids(tokenizer.tokenize('<chitchat>')):
                            instance, _ = build_input_from_segments(history, candidate, tokenizer, dp, cs,
                                                                    lm_labels, model=args.model_checkpoint,
                                                                    skill_mode='tod_single')
                        else:
                            instance, _ = build_input_from_segments(history, candidate, tokenizer,
                                                                    [], [], lm_labels,
                                                                    model=args.model_checkpoint,
                                                                    skill_mode='chitchat_single')
                    for input_name, input_array in instance.items():
                        datasets[dataset_name][input_name].append(input_array)
                datasets[dataset_name]["mc_labels"].append(num_candidates - 1)
                datasets[dataset_name]["n_candidates"] = num_candidates

    logger.info("Pad inputs and convert to Tensor and save it as cache")
    tensor_datasets = {"train": [], "valid": [], "test": []}
    for dataset_name, dataset in datasets.items():
        dataset = pad_dataset(dataset, padding=tokenizer.convert_tokens_to_ids('<pad>'))
        for input_name in MODEL_INPUTS:
            # only two task ids: 1 for chitchat and 0 for tod. This is used for adapter training
            if input_name == 'task_ids':
                if not args.use_adapter:
                    continue
                task_ids = [int(tokenizer.convert_tokens_to_ids('<cs>') not in input_id) for input_id in dataset['input_ids']]
                # exit()
                tensor = torch.tensor(task_ids)
                tensor = tensor.view((-1, datasets[dataset_name]["n_candidates"]) + tensor.shape[1:])
                tensor_datasets[dataset_name].append(tensor)
            else:
                tensor = torch.tensor(dataset[input_name])
                if input_name != "mc_labels":
                    tensor = tensor.view((-1, datasets[dataset_name]["n_candidates"]) + tensor.shape[1:])
                tensor_datasets[dataset_name].append(tensor)

    print('****** saving train and valid tensor datasets ******')
    torch.save(tensor_datasets["train"], args.tensor_dataset_cache + '_train')
    torch.save(tensor_datasets["valid"], args.tensor_dataset_cache + '_valid')
    torch.save(tensor_datasets["test"], args.tensor_dataset_cache + '_test')

    logger.info("Build train and validation dataloaders")
    train_dataset, valid_dataset = TensorDataset(*tensor_datasets["train"]), TensorDataset(*tensor_datasets["valid"])
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset) if args.distributed else None
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, shuffle=(not args.distributed))
    valid_loader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=args.valid_batch_size, shuffle=False)
    logger.info("Train dataset (Batch, Candidates, Seq length): {}".format(train_dataset.tensors[0].shape))
    logger.info("Valid dataset (Batch, Candidates, Seq length): {}".format(valid_dataset.tensors[0].shape))
    return train_loader, valid_loader, train_sampler, valid_sampler


def train():
    parser = ArgumentParser()
    parser.add_argument("--ckp_dir_name", type=str, required=True,
                        help="checkpoint directory name")
    parser.add_argument("--use_adapter", type=bool, default=False,
                        help="use adapter or not (NOTE: using adapter is not yet supported)")
    parser.add_argument("--mode", type=str, default='chitchat_single',
                        help="the mode for which the system is trained? Options: tod_single, chitchat_single, and fused")
    parser.add_argument("--fusedchat_cache", type=str, required=True, 
                        help="fusedchat cache after preprocessing")
    parser.add_argument("--tensor_dataset_cache", type=str, required=True, 
                        help="path for tokenized, tensor-format dataset cache")
    parser.add_argument("--prepend_dataset_path", type=str, required=True, 
                        help="path of the fusedchat dataset (prepend)")
    parser.add_argument("--append_dataset_path", type=str, required=True, 
                        help="path of the fusedchat dataset (append)")
    parser.add_argument("--lexicalized_prepend_dataset_path", type=str, required=True, 
                        help="path of the fusedchat dataset (prepend), lexicalized")
    parser.add_argument("--lexicalized_append_dataset_path", type=str, required=True, 
                        help="path of the fusedchat dataset (append), lexicalized")
    parser.add_argument("--model_checkpoint", type=str, default="gpt2", help="the model checkpoint to be loaded")
    parser.add_argument("--tokenizer_checkpoint", type=str, default="gpt2", help="the tokenizer to be used")
    parser.add_argument("--num_candidates", type=int, default=2, help="Number of candidates for training the doublehead model")
    parser.add_argument("--max_history", type=int, default=20, help="Number of previous exchanges to keep in history")
    parser.add_argument("--train_batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=1, help="Batch size for validation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Accumulate gradients on several steps")
    parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning rate")
    parser.add_argument("--lm_coef", type=float, default=2.0, help="LM loss coefficient")
    parser.add_argument("--mc_coef", type=float, default=1.0, help="Multiple-choice loss coefficient")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--n_epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--eval_before_start", action='store_true', help="If true start with a first evaluation before training")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--fp16", type=str, default="", help="Set to O0, O1, O2 or O3 for fp16 training (see apex documentation)")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training (-1: not distributed)")
    parser.add_argument("--description", type=str, default="fusedchat run feb 23",
                        help="A description of the run")
    parser.add_argument("--only_generating_data", type=str, default='no', help="only running to generate the tokenized and tensor data")

    args = parser.parse_args()
    # logging is set to INFO (resp. WARN) for main (resp. auxiliary) process.
    # logger.info => log main process only, logger.warning => log all processes
    logging.basicConfig(level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Running process %d", args.local_rank)  # This is a logger.warning: it will be printed by all distributed processes
    logger.info("Arguments: %s", pformat(args))

    # Initialize distributed training if needed
    args.distributed = (args.local_rank != -1)
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    logger.info("Prepare tokenizer, pretrained model and optimizer - add special tokens for fine-tuning")
    tokenizer_class = GPT2Tokenizer
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_checkpoint, unk_token='<|unkwn|>')
    if args.use_adapter:
        model_class = GPT2DoubleHeadsAdapterModel
    else:
        model_class = GPT2DoubleHeadsModel
    optimizer_class = AdamW
    model, _ = model_class.from_pretrained(args.model_checkpoint)

    model.to(args.device)
    optimizer = optimizer_class(model.parameters(), lr=args.lr)

    SPECIAL_TOKENS_DICT = {}
    if args.mode == 'fused':
        SPECIAL_TOKENS = SPECIAL_TOKENS_plus_chitchat_sor
    elif args.mode == 'tod_single':
        SPECIAL_TOKENS = SPECIAL_TOKENS_ORIGINAL
    elif args.mode == 'chitchat_single':
        SPECIAL_TOKENS = SPECIAL_TOKENS_chitchat_single
    elif args.mode == 'adapter':
        exit('adapter is not supported yet')
        SPECIAL_TOKENS = SPECIAL_TOKENS_ORIGINAL
    else:
        exit('Unknown mode specified!')
    for st in SPECIAL_TOKENS:
        SPECIAL_TOKENS_DICT[st] = st
    tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)
    model.resize_token_embeddings(len(tokenizer))
    s = ' '.join(act_name) + ' '.join(slot_name)
    print(tokenizer.decode(tokenizer.encode(s)))
    print(len(act_name) + len(slot_name), len(tokenizer.encode(s)))
    # Prepare model for FP16 and distributed training if needed (order is important, distributed should be the last)
    if args.fp16:
        from apex import amp  # Apex is only required if we use fp16 training
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16)
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

    logger.info("Prepare datasets")
    train_loader, val_loader, train_sampler, valid_sampler = get_data_loaders(args, tokenizer)

    if args.only_generating_data == 'yes':
        exit()
    
    # Training function and trainer
    def update(engine, batch):
        model.train()
        batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
        if args.use_adapter:
            input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids, task_id = batch
            lm_loss, mc_loss, *_ = model(input_ids=input_ids, mc_token_ids=mc_token_ids, lm_labels=lm_labels,
                                         mc_labels=mc_labels, token_type_ids=token_type_ids, task_id=task_id)
        else:
            input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids= batch
            lm_loss, mc_loss, *_ = model(input_ids=input_ids, mc_token_ids=mc_token_ids, lm_labels=lm_labels,
                                         mc_labels=mc_labels, token_type_ids=token_type_ids)

        loss = (lm_loss * args.lm_coef + mc_loss * args.mc_coef) / args.gradient_accumulation_steps
        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_norm)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        if engine.state.iteration % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        return loss.item()

    trainer = Engine(update)
    trainer._logger.setLevel(logging.INFO)

    # Evaluation function and evaluator (inference output is the input of the metrics)

    def inference(engine, batch):
        model.eval()
        with torch.no_grad():
            batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
            if args.use_adapter:
                input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids, task_id = batch
            else:
                input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids = batch
            logger.info(tokenizer.decode(input_ids[0, -1, :].tolist()))
            if args.use_adapter:
                model_outputs = model(input_ids, mc_token_ids, token_type_ids=token_type_ids, task_id=task_id)
            else:
                model_outputs = model(input_ids, mc_token_ids, token_type_ids=token_type_ids)
            lm_logits, mc_logits = model_outputs[0], model_outputs[1]  # So we can also use GPT2 outputs
            lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
            lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)
            return (lm_logits_flat_shifted, mc_logits), (lm_labels_flat_shifted, mc_labels)

    evaluator = Engine(inference)
    evaluator._logger.setLevel(logging.INFO)

    # Attach evaluation to trainer: we evaluate when we start the training and at the end of each epoch
    trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: evaluator.run(val_loader))
    if args.n_epochs < 1:
        trainer.add_event_handler(Events.COMPLETED, lambda _: evaluator.run(val_loader))
    if args.eval_before_start:
        trainer.add_event_handler(Events.STARTED, lambda _: evaluator.run(val_loader))

    # Make sure distributed data samplers split the dataset nicely between the distributed processes
    if args.distributed:
        trainer.add_event_handler(Events.EPOCH_STARTED, lambda engine: train_sampler.set_epoch(engine.state.epoch))
        evaluator.add_event_handler(Events.EPOCH_STARTED, lambda engine: valid_sampler.set_epoch(engine.state.epoch))

    # Linearly decrease the learning rate from lr to zero
    scheduler = PiecewiseLinear(optimizer, "lr", [(0, args.lr), (args.n_epochs * len(train_loader), 0.0)])
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    # Prepare metrics - note how we compute distributed metrics
    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")
    metrics = {"nll": Loss(torch.nn.CrossEntropyLoss(ignore_index=-1), output_transform=lambda x: (x[0][0], x[1][0])),
               "accuracy": Accuracy(output_transform=lambda x: (x[0][1], x[1][1]))}
    metrics.update({"average_nll": MetricsLambda(average_distributed_scalar, metrics["nll"], args),
                    "average_accuracy": MetricsLambda(average_distributed_scalar, metrics["accuracy"], args)})
    metrics["average_ppl"] = MetricsLambda(math.exp, metrics["average_nll"])
    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    # On the main process: add progress bar, tensorboard, checkpoints and save model,
    # configuration and tokenizer before we start to train
    if args.local_rank in [-1, 0]:
        pbar = ProgressBar(persist=True)
        pbar.attach(trainer, metric_names=["loss"])
        evaluator.add_event_handler(Events.COMPLETED, lambda _: pbar.log_message("Validation: %s" % pformat(evaluator.state.metrics)))
        tb_logger = TensorboardLogger(log_dir=None)
        # tb_logger.writer.log_dir = tb_logger.writer.file_writer.get_logdir()
        tb_logger.writer.log_dir = args.ckp_dir_name
        # tb_logger.writer.file_writer.get_logdir()
        logger.info(tb_logger.writer.log_dir)
        tb_logger.attach(trainer, log_handler=OutputHandler(tag="training", metric_names=["loss"]), event_name=Events.ITERATION_COMPLETED)
        tb_logger.attach(trainer, log_handler=OptimizerParamsHandler(optimizer), event_name=Events.ITERATION_STARTED)
        tb_logger.attach(evaluator, log_handler=OutputHandler(tag="validation", metric_names=list(metrics.keys()), another_engine=trainer), event_name=Events.EPOCH_COMPLETED)

        checkpoint_handler = ModelCheckpoint(tb_logger.writer.log_dir, 'checkpoint', save_interval=1, n_saved=100)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {'mymodel': getattr(model, 'module', model)})  # "getattr" take care of distributed encapsulation
        torch.save(args, tb_logger.writer.log_dir + '/model_training_args.bin')
        args_dict = vars(args)
        args_dict_string = {key:str(value) for key, value in args_dict.items()}
        with open(tb_logger.writer.log_dir + '/model_training_args.json', 'w') as f:
            json.dump(args_dict_string, f)

        # args_dict = vars(args)
        # args_dict_new = {key: str(value) for key, value in args_dict.item()}
        # with open('model_training_args.json', 'w') as f:
        #     json.dump(args_dict_new, f)

        getattr(model, 'module', model).config.to_json_file(os.path.join(tb_logger.writer.log_dir, CONFIG_NAME))
        tokenizer.save_vocabulary(tb_logger.writer.log_dir)

    # Run the training
    print('training started')
    trainer.run(train_loader, max_epochs=args.n_epochs)

    # On the main process: close tensorboard logger and rename the last checkpoint
    # (for easy re-loading with OpenAIGPTModel.from_pretrained method)

    if args.local_rank in [-1, 0] and args.n_epochs > 0:
        os.rename(checkpoint_handler._saved[-1][1][-1],
                  os.path.join(tb_logger.writer.log_dir, WEIGHTS_NAME))
        # TODO: PR in ignite to have better access to saved file paths (cleaner)
        tb_logger.close()


if __name__ == "__main__":
    train()
