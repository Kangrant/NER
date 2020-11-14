import pdb
import json
import torch
import argparse
import torch.nn as nn
from torch import optim
import config
import time
from model import NERModel
from BERT_model.model_BERT import BERT_NERModel
from dataset_loader import DatasetLoader
from progressbar import ProgressBar
from ner_metrics import SeqEntityScore
from data_processor import CluenerProcessor
from lr_scheduler import ReduceLROnPlateau
from utils_ner import get_entities
from common import (init_logger,
                    logger,
                    json_to_text,
                    load_model,
                    AverageMeter,
                    seed_everything)
from transformers import BertTokenizer


def train(args, model, processor):
    tokenizer = BertTokenizer.from_pretrained('./BERT_model/bert_pretrain/vocab.txt')

    train_dataset = load_and_cache_examples(args, processor, data_type='train')
    train_loader = DatasetLoader(data=train_dataset, batch_size=args.batch_size,
                                 shuffle=False, seed=args.seed, sort=True,
                                 vocab = processor.vocab,label2id = args.label2id,tokenizer=tokenizer)
    # train_loader = DatasetLoader(data=train_dataset, batch_size=args.batch_size,
    #                              shuffle=False, seed=args.seed, sort=True,
    #                              vocab=processor.vocab, label2id=args.label2id)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(parameters, lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3,
                                  verbose=1, epsilon=1e-4, cooldown=0, min_lr=0, eps=1e-8)

    train_metric = SeqEntityScore(args.id2label, markup=args.markup)
    best_f1 = 0
    for epoch in range(1, 1 + args.epochs):
        strat_epoch_time = time.time()
        logger.info(f"Epoch {epoch}/{args.epochs}")
        #pbar = ProgressBar(n_total=len(train_loader), desc='Training') #进度条样式
        train_loss = AverageMeter()
        model.train()
        assert model.training
        for step, batch in enumerate(train_loader):
            strat_batch_time = time.time()
            input_ids, input_mask, input_tags, input_lens = batch
            input_ids = input_ids.to(args.device)
            input_mask = input_mask.to(args.device)
            input_tags = input_tags.to(args.device)

            features, loss = model.forward_loss(input_ids, input_mask, input_lens, input_tags)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
            optimizer.step()
            optimizer.zero_grad()

            # pbar(step=step, info={'loss': loss.item()})
            train_loss.update(loss.item(), n=1)

            tags, _ = model.crf._obtain_labels(features, args.id2label, input_lens)
            input_tags = input_tags.cpu().numpy()
            target = [input_[:len_] for input_, len_ in zip(input_tags, input_lens)]

            pre_train = train_metric.compute_train_pre(label_paths=target, pred_paths=tags)
            logger.info(
                f'time: {time.time() - strat_batch_time:.1f}  train_loss: {loss.item():.4f}  train_pre: {pre_train:.4f}')
        print(" ")
        logger.info(f'train_total_time: {time.time() - strat_epoch_time}')

        if 'cuda' in str(args.device):
            torch.cuda.empty_cache()  # 释放显存

        strat_eval_time = time.time()
        eval_f1 = evaluate(args, model, processor)

        show_info = f'eval_time: {time.time() - strat_eval_time:.1f}   train_avg_loss: {train_loss.avg:.4f}  eval_f1: {eval_f1:.4f} '
        logger.info(show_info)
        scheduler.epoch_step(eval_f1, epoch)

        if eval_f1 > best_f1:
            # Epoch 1: eval_f1 improved from 0 to 0.4023105674481821
            logger.info(f"\nEpoch {epoch}: eval_f1 improved from {best_f1} to {eval_f1}")

            best_f1 = eval_f1

            model_stat_dict = model.state_dict()
            state = {'epoch': epoch, 'arch': args.arch, 'state_dict': model_stat_dict}
            model_path = args.output_dir / 'best-model.bin'
            torch.save(state, str(model_path))

            # 修改：不打印具体标签相关情况
            # print("Eval Entity Score: ")
            # for key, value in class_info.items():
            #    #Subject: name - Acc: 0.3333 - Recall: 0.3398 - F1: 0.3365
            #    #Subject: address - Acc: 0.186 - Recall: 0.193 - F1: 0.1895
            #     info = f"Subject: {key} - Acc: {value['acc']} - Recall: {value['recall']} - F1: {value['f1']}"
            #     logger.info(info)


def evaluate(args, model, processor):
    eval_dataset = load_and_cache_examples(args, processor, data_type='dev')
    eval_dataloader = DatasetLoader(data=eval_dataset, batch_size=args.batch_size,
                                    shuffle=False, seed=args.seed, sort=False,
                                    vocab=processor.vocab, label2id=args.label2id)
    pbar = ProgressBar(n_total=len(eval_dataloader), desc="Evaluating")
    metric = SeqEntityScore(args.id2label, markup=args.markup)
    eval_loss = AverageMeter()
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(eval_dataloader):
            input_ids, input_mask, input_tags, input_lens = batch
            input_ids = input_ids.to(args.device)
            input_mask = input_mask.to(args.device)
            input_tags = input_tags.to(args.device)
            features, loss = model.forward_loss(input_ids, input_mask, input_lens, input_tags)
            eval_loss.update(val=loss.item(), n=input_ids.size(0))
            tags, _ = model.crf._obtain_labels(features, args.id2label, input_lens)
            input_tags = input_tags.cpu().numpy()
            target = [input_[:len_] for input_, len_ in zip(input_tags, input_lens)]
            metric.update(pred_paths=tags, label_paths=target)

            pbar(step=step)
    print(" ")

    # 修改：删除class_info
    # eval_info, class_info = metric.result()
    eval_info = metric.result()
    # eval_info = {f'eval_{key}': value for key, value in eval_info.items()}
    # result = {'eval_loss': eval_loss.avg}
    # result = dict(result, **eval_info)
    return eval_info
    # return result, class_info


def predict(args, model, processor):
    model_path = args.output_dir / 'best-model.bin'
    model = load_model(model, model_path=str(model_path))

    # metric = SeqEntityScore(args.id2label, markup=args.markup)
    # 取数据 test_data = [{id: ,context: ,tag: ,raw_context: },{},{}...]
    start_time = time.time()
    test_data = load_and_cache_examples(args, processor, data_type='test')
    # test_data [{'context':,'tag':}，{},{}]
    origins = []
    founds = []
    rights = []

    results = []
    for step, line in enumerate(test_data):
        token_a = line['context'].split(" ")
        tag_a = line['tag'].split(" ")
        input_ids = [processor.vocab.to_index(w) for w in token_a]
        input_mask = [1] * len(token_a)
        input_lens = [len(token_a)]
        model.eval()
        with torch.no_grad():
            input_ids = torch.tensor([input_ids], dtype=torch.long)
            input_mask = torch.tensor([input_mask], dtype=torch.long)
            input_lens = torch.tensor([input_lens], dtype=torch.long)
            input_ids = input_ids.to(args.device)
            input_mask = input_mask.to(args.device)
            features = model.forward_loss(input_ids, input_mask, input_lens, input_tags=None)
            tags, _ = model.crf._obtain_labels(features, args.id2label, input_lens)
        label_entities = get_entities(tags[0], args.id2label)
        gold_entities = get_entities(tag_a, args.id2label)
        # 记录标签
        origins.extend(gold_entities)
        founds.extend(label_entities)
        rights.extend([pre_entity for pre_entity in label_entities if pre_entity in gold_entities])

        json_d = {}
        # json_d['tag_seq'] = " ".join(tags[0])
        json_d['pre'] = label_entities
        json_d['gold'] = gold_entities
        results.append(json_d)
    # result [{'pre': ,'gold': },{},{}]

    test_submit = []
    for x, y in zip(test_data, results):
        json_d = {}
        context = list(x['context'])
        json_d['context'] = ''.join(context)
        json_d['label'] = y['pre']
        # entities = y['pre']
        # if len(entities) != 0:
        #     for subject in entities:
        #         tag = subject[0]
        #         start = subject[1]
        #         end = subject[2]
        #         word = "".join(context[start:end + 1])
        #         json_d['label'][tag] = word

        json_d['gold'] = y['gold']
        test_submit.append(json_d)

    output_submit_file = str(args.output_dir / "test_submit.json")
    with open(output_submit_file, 'w') as writer:
        for x in test_submit:
            writer.write(json.dumps(x, ensure_ascii=False) + '\n')

    precision = len(rights) / len(founds)
    recall = len(rights) / len(origins)
    test_f1 = (2 * precision * recall) / (precision + recall)
    logger.info(f'test_time: {time.time() - start_time:.1f}  test_f1: {test_f1}')


def load_and_cache_examples(args, processor, data_type='train'):
    # Load data features from cache or dataset file
    cached_examples_file = args.data_dir / 'cached_crf-{}_{}_{}'.format(
        data_type,
        args.arch,
        str(args.task_name))
    if cached_examples_file.exists():
        logger.info("Loading features from cached file %s", cached_examples_file)
        examples = torch.load(cached_examples_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if data_type == 'train':
            examples = processor.get_train_examples()
        elif data_type == 'dev':
            examples = processor.get_dev_examples()
        else:
            examples = processor.get_test_examples()
        logger.info("Saving features into cached file %s", cached_examples_file)
        torch.save(examples, str(cached_examples_file))
    return examples


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--do_train", default=True, action='store_true')
    parser.add_argument('--do_eval', default=False, action='store_true')
    parser.add_argument("--do_predict", default=False, action='store_true')

    parser.add_argument('--markup', default='bieos', type=str, choices=['bios', 'bio', 'bieos'])  # 标签类型
    parser.add_argument("--arch", default='bilstm_crf', type=str)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--gpu', default='', type=str)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--embedding_size', default=128, type=int)
    parser.add_argument('--hidden_size', default=384, type=int)
    parser.add_argument("--grad_norm", default=5.0, type=float, help="Max gradient norm.")
    parser.add_argument("--task_name", type=str, default='ner')
    args = parser.parse_args()
    args.data_dir = config.data_dir
    if not config.output_dir.exists():
        args.output_dir.mkdir()
    args.output_dir = config.output_dir / '{}'.format(args.arch)
    if not args.output_dir.exists():
        args.output_dir.mkdir()
    init_logger(log_file=str(args.output_dir / '{}-{}.log'.format(args.arch, args.task_name)))
    seed_everything(args.seed)
    if args.gpu != '':
        args.device = torch.device(f"cuda:{args.gpu}")
    else:
        args.device = torch.device("cpu")
    args.id2label = {i: label for i, label in enumerate(config.label2id)}
    args.label2id = config.label2id

    processor = CluenerProcessor(data_dir=config.data_dir)
    processor.get_vocab()

    model = BERT_NERModel( device=args.device, label2id=args.label2id,need_birnn=True)
    # model = NERModel(vocab_size=len(processor.vocab), embedding_size=args.embedding_size,
    #                  hidden_size=args.hidden_size, device=args.device, label2id=args.label2id)

    model.to(args.device)

    if args.do_train:
        train(args, model, processor)
    if args.do_eval:
        model_path = args.output_dir / 'best-model.bin'
        model = load_model(model, model_path=str(model_path))
        evaluate(args, model, processor)
    if args.do_predict:
        predict(args, model, processor)


if __name__ == "__main__":
    main()
