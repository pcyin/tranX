# coding=utf-8
import argparse


class cached_property(object):
    """ A property that is only computed once per instance and then replaces
        itself with an ordinary attribute. Deleting the attribute resets the
        property.

        Source: https://github.com/bottlepy/bottle/commit/fa7733e075da0d790d809aa3d2f53071897e6f76
        """

    def __init__(self, func):
        self.__doc__ = getattr(func, '__doc__')
        self.func = func

    def __get__(self, obj, cls):
        if obj is None:
            return self
        value = obj.__dict__[self.func.__name__] = self.func(obj)
        return value


def init_arg_parser():
    arg_parser = argparse.ArgumentParser()

    #### General configuration ####
    arg_parser.add_argument('--seed', default=0, type=int, help='Random seed')
    arg_parser.add_argument('--cuda', action='store_true', default=False, help='Use gpu')
    arg_parser.add_argument('--lang', choices=['python', 'lambda_dcs', 'wikisql', 'prolog', 'python3'], default='python')
    arg_parser.add_argument('--asdl_file', type=str, help='Path to ASDL grammar specification')
    arg_parser.add_argument('--mode', choices=['train', 'self_train', 'train_reconstructor', 'train_paraphrase_identifier',
                                               'test', 'rerank', 'interactive'], default='train', help='Run mode')
    arg_parser.add_argument('--evaluator', type=str, default='default_evaluator', required=False)

    #### Model configuration ####
    arg_parser.add_argument('--lstm', choices=['lstm'], default='lstm', help='Type of LSTM used, currently only standard LSTM cell is supported')

    # Embedding sizes
    arg_parser.add_argument('--embed_size', default=128, type=int, help='Size of word embeddings')
    arg_parser.add_argument('--action_embed_size', default=128, type=int, help='Size of ApplyRule/GenToken action embeddings')
    arg_parser.add_argument('--field_embed_size', default=64, type=int, help='Embedding size of ASDL fields')
    arg_parser.add_argument('--type_embed_size', default=64, type=int, help='Embeddings ASDL types')

    # Hidden sizes
    arg_parser.add_argument('--hidden_size', default=256, type=int, help='Size of LSTM hidden states')
    arg_parser.add_argument('--ptrnet_hidden_dim', default=32, type=int, help='Hidden dimension used in pointer network')
    arg_parser.add_argument('--att_vec_size', default=256, type=int, help='size of attentional vector')

    # readout layer
    arg_parser.add_argument('--no_query_vec_to_action_map', default=False, action='store_true',
                            help='Do not use additional linear layer to transform the attentional vector for computing action probabilities')
    arg_parser.add_argument('--readout', default='identity', choices=['identity', 'non_linear'],
                            help='Type of activation if using additional linear layer')
    arg_parser.add_argument('--query_vec_to_action_diff_map', default=False, action='store_true',
                            help='Use different linear mapping ')

    # supervised attention
    arg_parser.add_argument('--sup_attention', default=False, action='store_true', help='Use supervised attention')

    # parent information switch for decoder LSTM
    arg_parser.add_argument('--no_parent_production_embed', default=False, action='store_true',
                            help='Do not use embedding of parent ASDL production to update decoder LSTM state')
    arg_parser.add_argument('--no_parent_field_embed', default=False, action='store_true',
                            help='Do not use embedding of parent field to update decoder LSTM state')
    arg_parser.add_argument('--no_parent_field_type_embed', default=False, action='store_true',
                            help='Do not use embedding of the ASDL type of parent field to update decoder LSTM state')
    arg_parser.add_argument('--no_parent_state', default=False, action='store_true',
                            help='Do not use the parent hidden state to update decoder LSTM state')

    arg_parser.add_argument('--no_input_feed', default=False, action='store_true', help='Do not use input feeding in decoder LSTM')
    arg_parser.add_argument('--no_copy', default=False, action='store_true', help='Do not use copy mechanism')

    # Model configuration parameters specific for wikisql
    arg_parser.add_argument('--column_att', choices=['dot_prod', 'affine'], default='affine', help='How to perform attention over table columns')
    arg_parser.add_argument('--answer_prune', dest='answer_prune', action='store_true', help='Whether to use answer pruning [default: True]')
    arg_parser.set_defaults(answer_prune=True)
    arg_parser.add_argument('--no_answer_prune', dest='answer_prune', action='store_false', help='Do not use answer prunning')

    #### Training ####
    arg_parser.add_argument('--vocab', type=str, help='Path of the serialized vocabulary')
    arg_parser.add_argument('--glove_embed_path', default=None, type=str, help='Path to pretrained Glove mebedding')

    arg_parser.add_argument('--train_file', type=str, help='path to the training target file')
    arg_parser.add_argument('--dev_file', type=str, help='path to the dev source file')

    arg_parser.add_argument('--batch_size', default=10, type=int, help='Batch size')
    arg_parser.add_argument('--dropout', default=0., type=float, help='Dropout rate')
    arg_parser.add_argument('--word_dropout', default=0., type=float, help='Word dropout rate')
    arg_parser.add_argument('--decoder_word_dropout', default=0.3, type=float, help='Word dropout rate on decoder')
    arg_parser.add_argument('--primitive_token_label_smoothing', default=0.0, type=float,
                            help='Apply label smoothing when predicting primitive tokens')
    arg_parser.add_argument('--src_token_label_smoothing', default=0.0, type=float,
                            help='Apply label smoothing in reconstruction model when predicting source tokens')

    arg_parser.add_argument('--negative_sample_type', default='best', type=str, choices=['best', 'sample', 'all'])

    # training schedule details
    arg_parser.add_argument('--valid_metric', default='acc', choices=['acc'],
                            help='Metric used for validation')
    arg_parser.add_argument('--valid_every_epoch', default=1, type=int, help='Perform validation every x epoch')
    arg_parser.add_argument('--log_every', default=10, type=int, help='Log training statistics every n iterations')

    arg_parser.add_argument('--save_to', default='model', type=str, help='Save trained model to')
    arg_parser.add_argument('--save_all_models', default=False, action='store_true', help='Save all intermediate checkpoints')
    arg_parser.add_argument('--patience', default=5, type=int, help='Training patience')
    arg_parser.add_argument('--max_num_trial', default=10, type=int, help='Stop training after x number of trials')
    arg_parser.add_argument('--uniform_init', default=None, type=float,
                            help='If specified, use uniform initialization for all parameters')
    arg_parser.add_argument('--glorot_init', default=False, action='store_true', help='Use glorot initialization')
    arg_parser.add_argument('--clip_grad', default=5., type=float, help='Clip gradients')
    arg_parser.add_argument('--max_epoch', default=-1, type=int, help='Maximum number of training epoches')
    arg_parser.add_argument('--optimizer', default='Adam', type=str, help='optimizer')
    arg_parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    arg_parser.add_argument('--lr_decay', default=0.5, type=float,
                            help='decay learning rate if the validation performance drops')
    arg_parser.add_argument('--lr_decay_after_epoch', default=0, type=int, help='Decay learning rate after x epoch')
    arg_parser.add_argument('--reset_optimizer', action='store_true', default=False, help='Whether to reset optimizer when loading the best checkpoint')
    arg_parser.add_argument('--verbose', action='store_true', default=False, help='Verbose mode')
    arg_parser.add_argument('--eval_top_pred_only', action='store_true', default=False,
                            help='Only evaluate the top prediction in validation')

    #### decoding/validation/testing ####
    arg_parser.add_argument('--load_model', default=None, type=str, help='Load a pre-trained model')
    arg_parser.add_argument('--beam_size', default=5, type=int, help='Beam size for beam search')
    arg_parser.add_argument('--decode_max_time_step', default=100, type=int, help='Maximum number of time steps used '
                                                                                  'in decoding and sampling')
    arg_parser.add_argument('--sample_size', default=5, type=int, help='Sample size')
    arg_parser.add_argument('--test_file', type=str, help='Path to the test file')
    arg_parser.add_argument('--save_decode_to', default=None, type=str, help='Save decoding results to file')

    #### reranking ####
    arg_parser.add_argument('--features', nargs='+')
    arg_parser.add_argument('--load_reconstruction_model', type=str, help='Load reconstruction model')
    arg_parser.add_argument('--load_paraphrase_model', type=str, help='Load paraphrase model')
    arg_parser.add_argument('--tie_embed', action='store_true', help='tie source and target embedding in training paraphrasing model')
    arg_parser.add_argument('--train_decode_file', default=None, type=str, help='Decoding results on training set')
    arg_parser.add_argument('--test_decode_file', default=None, type=str, help='Decoding results on test set')
    arg_parser.add_argument('--dev_decode_file', default=None, type=str, help='Decoding results on dev set')
    arg_parser.add_argument('--metric', default='accuracy', choices=['bleu', 'accuracy'])
    arg_parser.add_argument('--num_workers', default=1, type=int, help='number of multiprocess workers')

    #### self-training ####
    arg_parser.add_argument('--load_decode_results', default=None, type=str)
    arg_parser.add_argument('--unsup_loss_weight', default=1., type=float, help='loss of unsupervised learning weight')
    arg_parser.add_argument('--unlabeled_file', type=str, help='Path to the training source file used in semi-supervised self-training')

    #### interactive mode ####
    arg_parser.add_argument('--dataset_name', default=None, type=str)

    return arg_parser


def update_args(args, arg_parser):
    for action in arg_parser._actions:
        if isinstance(action, argparse._StoreAction) or isinstance(action, argparse._StoreTrueAction) \
                or isinstance(action, argparse._StoreFalseAction):
            if not hasattr(args, action.dest):
                setattr(args, action.dest, action.default)
