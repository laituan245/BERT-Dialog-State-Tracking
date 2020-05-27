import torch
import random
import itertools

from tqdm import tqdm
from utils import RunningAverage, rindex, pad

from torch import nn
from torch import optim
from torch.nn import functional as F

from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification

# Special Tokens
CLS = '[CLS]'
SEP = '[SEP]'

# Generate examples for a turn
def turn_to_examples(t, ontology, tokenizer):
    examples = []
    user_transcript = t.transcript
    if isinstance(user_transcript, list): user_transcript = ' '.join(user_transcript)
    if len(t.asr) > 0: user_transcript = t.asr[0][0]
    context = ' '.join([t.system_transcript] + [SEP] + [user_transcript])
    turn_label = set([(s, v) for s, v in t.turn_label])
    for slot in ontology.slots:
        for value in ontology.values[slot]:
            candidate = slot + ' = ' + value

            # Prepare input_ids
            input_text = ' '.join([CLS, context, SEP, candidate, SEP])
            tokenized_text = tokenizer.tokenize(input_text)
            input_ids = tokenizer.convert_tokens_to_ids(tokenized_text)

            # Prepare token_type_ids
            sent1_len = rindex(tokenized_text[:-1], SEP) + 1
            sent2_len = len(tokenized_text) - sent1_len
            token_type_ids = [0] * sent1_len + [1] * sent2_len

            # Prepare label
            label = int((slot, value) in turn_label)

            # Update examples list
            examples.append((slot, value, input_ids, token_type_ids, label))
    return examples

class Model(nn.Module):
    def __init__(self, tokenizer, bert):
        super(Model, self).__init__()

        self.tokenizer = tokenizer
        self.bert = bert

    @classmethod
    def from_scratch(cls, bert_model, verbose=True):
        tokenizer = BertTokenizer.from_pretrained(bert_model)
        bert = BertForSequenceClassification.from_pretrained(bert_model, num_labels=2)

        model = cls(tokenizer, bert)
        if verbose:
            print('Intialized the model and the tokenizer from scratch')
        return model

    @classmethod
    def from_model_path(cls, output_model_path, verbose=True):
        tokenizer = BertTokenizer.from_pretrained(output_model_path)
        bert = BertForSequenceClassification.from_pretrained(output_model_path, num_labels=2)

        model = cls(tokenizer, bert)
        if verbose:
            print('Restored the model and the tokenizer from {}'.format(output_model_path))
        return model

    def move_to_device(self, args):
        self.bert.to(args.device)
        if args.n_gpus > 1:
            self.bert = torch.nn.DataParallel(self.bert)

    def init_optimizer(self, args, num_train_iters):
        # Optimizer
        param_optimizer = list(self.bert.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        self.optimizer = BertAdam(optimizer_grouped_parameters,
                                  lr=args.learning_rate,
                                  warmup=args.warmup_proportion,
                                  t_total=num_train_iters)
        self.optimizer.zero_grad()

    def run_train(self, dataset, ontology, args):
        model, tokenizer = self.bert, self.tokenizer
        batch_size = args.batch_size
        self.train()

        # Generate training examples
        turns = list(dataset['train'].iter_turns())
        train_examples = [turn_to_examples(t, ontology, tokenizer) for t in turns]
        train_examples = list(itertools.chain.from_iterable(train_examples))
        print('Generated training examples')

        # Random Oversampling
        # Note that: Most of the constructed examples are negative
        if args.random_oversampling:
            negative_examples, positive_examples = [], []
            for example in train_examples:
                if example[-1] == 0: negative_examples.append(example)
                if example[-1] == 1: positive_examples.append(example)
            nb_negatives, nb_positives = len(negative_examples), len(positive_examples)
            sampled_positive_examples = random.choices(positive_examples, k=int(nb_negatives / 8))
            train_examples = sampled_positive_examples + negative_examples
            print('Did Random Oversampling')
            print('Number of positive examples increased from {} to {}'
                  .format(nb_positives, len(sampled_positive_examples)))

        # Initialize Optimizer
        num_train_iters = args.epochs * len(train_examples) / batch_size / args.gradient_accumulation_steps
        self.init_optimizer(args, num_train_iters)

        # Main training loop
        iterations = 0
        best_dev_joint_goal = 0.0
        train_avg_loss = RunningAverage()
        for epoch in range(args.epochs):
            print('Epoch {}'.format(epoch))

            random.shuffle(train_examples)
            pbar = tqdm(range(0, len(train_examples), batch_size))
            for i in pbar:
                iterations += 1

                # Next training batch
                batch = train_examples[i:i+batch_size]
                _, _, input_ids, token_type_ids, labels = list(zip(*batch))

                # Padding and Convert to Torch Tensors
                input_ids, input_masks = pad(input_ids, args.device)
                token_type_ids = pad(token_type_ids, args.device)[0]
                labels = torch.LongTensor(labels).to(args.device)

                # Calculate loss
                loss = model(input_ids, token_type_ids, input_masks, labels=labels)
                if args.n_gpus > 1:
                    loss = loss.mean()
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()
                train_avg_loss.update(loss.item())

                # Update pbar
                pbar.update(1)
                pbar.set_postfix_str(f'Train Loss: {train_avg_loss()}')

                # parameters update
                if iterations % args.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            # Evaluate on the dev set and the test set
            dev_results = self.run_dev(dataset, ontology, args)
            test_results = self.run_test(dataset, ontology, args)

            print('Evaluations after epoch {}'.format(epoch))
            print(dev_results)
            print(test_results)
            if dev_results['joint_goal'] > best_dev_joint_goal:
                best_dev_joint_goal = dev_results['joint_goal']
                self.save(args.output_dir)
                print('Saved the model')

    def predict_turn(self, turn, ontology, args, threshold=0.5):
        model, tokenizer = self.bert, self.tokenizer
        batch_size = args.batch_size
        was_training = model.training
        self.eval()

        preds = []
        examples = turn_to_examples(turn, ontology, tokenizer)
        for i in range(0, len(examples), batch_size):
            batch = examples[i:i+batch_size]
            slots, values, input_ids, token_type_ids, _ = list(zip(*batch))

            # Padding and Convert to Torch Tensors
            input_ids, input_masks = pad(input_ids, args.device)
            token_type_ids = pad(token_type_ids, args.device)[0]

            # Forward Pass
            logits = model(input_ids, token_type_ids, input_masks)
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().data.numpy()

            # Update preds
            for j in range(len(batch)):
                if probs[j] >= threshold:
                    preds.append((slots[j], values[j]))

        if was_training:
            self.train()

        return preds

    def run_dev(self, dataset, ontology, args):
        turns = list(dataset['dev'].iter_turns())
        preds = [self.predict_turn(t, ontology, args) for t in turns]
        return dataset['dev'].evaluate_preds(preds)

    def run_test(self, dataset, ontology, args):
        turns = list(dataset['test'].iter_turns())
        preds = [self.predict_turn(t, ontology, args) for t in turns]
        return dataset['test'].evaluate_preds(preds)

    def save(self, output_model_path, verbose=True):
        model, tokenizer = self.bert, self.tokenizer
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

        output_model_file = output_model_path / WEIGHTS_NAME
        output_config_file = output_model_path / CONFIG_NAME

        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        tokenizer.save_vocabulary(output_model_path)

        if verbose:
            print('Saved the model, the model config and the tokenizer')
