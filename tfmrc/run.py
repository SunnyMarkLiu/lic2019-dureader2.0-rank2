#!/usr/bin/python
# _*_ coding: utf-8 _*_

"""
This module prepares and runs the whole system.
"""
from model import MultiAnsModel
from util.dataset import Vocab
from util.dataset import Dataset
import logging
import argparse
import pickle
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"  # INFO/warning/ERROR/FATAL

def parse_args():
    """
    Parses command line arguments.
    """
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Unsupported value encountered.')

    parser = argparse.ArgumentParser(
        description='Reading Comprehension on BaiduRC dataset')
    parser.add_argument('--prepare', action='store_true',
                        help='create the directories, prepare the vocabulary and embeddings')
    parser.add_argument('--train', action='store_true',
                        help='train the model')
    parser.add_argument('--evaluate', action='store_true',
                        help='evaluate the model on dev set')
    parser.add_argument('--predict', action='store_true',
                        help='predict the answers for test set with trained model')
    parser.add_argument('--gpu', type=str, default='0',
                        help='specify gpu device')

    extra_settings = parser.add_argument_group('extra settings')
    # loss选择
    extra_settings.add_argument('--use_multi_ans_loss', type=str2bool, default=True,
                                help='whether answer prediction with multi-answer')
    extra_settings.add_argument('--multi_ans_norm', choices=['None', 'max_min', 'sum'], default='None',
                                help='how to normalize the multi ans')
    extra_settings.add_argument('--ps_loss_weight', type=float, default=0,
                                help='the passage selection loss weight, if 0, not use')
    extra_settings.add_argument('--mrt_loss_weight', type=float, default=0,
                                help=' the Minimum Risk Training loss weight, if 0, not use')
    # 特征选择
    extra_settings.add_argument('--pos_size', type=int, default=30,
                                help='size of pos tagging, if 0, will not use pos')
    extra_settings.add_argument('--use_pos_freq', type=str2bool, default=True,
                                help='whether use the pos frequence')
    extra_settings.add_argument('--use_wiq_feature', type=str2bool, default=False,
                                help='whether use the word-in-question feature')
    extra_settings.add_argument('--use_keyword_feature', type=str2bool, default=True,
                                help='whether use the keyword feature')
    # 问题的粗分类和细分类特征
    extra_settings.add_argument('--use_rough_classify_feature', type=str2bool, default=True,
                                help='whether use the rough classify feature')
    extra_settings.add_argument('--use_fine_classify_feature', type=str2bool, default=True,
                                help='whether use the fine classify feature')
    extra_settings.add_argument('--fine_cls_num', type=int, default=13,
                                help='fine classify nums')

    # 词表选择相关
    extra_settings.add_argument('--initial_tokens_random', type=str2bool, default=False,
                                help='whether init the initial tokens random, if False, init them 0')
    extra_settings.add_argument('--use_oov2unk', type=str2bool, default=True,
                                help='if True, all oov words project to unk')
    extra_settings.add_argument('--vocab_dir', default='cache/vocab',
                                help='the dir to save/load vocabulary')
    extra_settings.add_argument('--vocab_file', default='v4_baidu_cnt2_vocab.data',
                                help='the file to save/load vocabulary')
    extra_settings.add_argument('--create_vocab', type=str2bool, default=False,
                                help='whether create vocab file when run prepare function')
    extra_settings.add_argument('--vocab_min_cnt', type=int, default=2,
                                help='filter the vocab where their cnt < vocab_min_cnt')
    # 文档rank分数选择
    extra_settings.add_argument('--use_para_prior_scores', choices=["None", "baidu", "zhidao", "search", "all", "best"], default='None',
                                help='choose one of ["None", "baidu", "zhidao", "search", "all", "best"]')
    # bad case记录路径
    extra_settings.add_argument('--badcase_sample_log_file', type=str, default='badcase_sample_log_file.json',
                                help='badcase_sample_log_file')
    # 随机种子
    extra_settings.add_argument('--random_seed', type=int, default=777,
                                help='a seed value for randomness.')

    train_settings = parser.add_argument_group('train settings')
    train_settings.add_argument('--optim', default='adam',
                                help='optimizer type')
    train_settings.add_argument('--learning_rate', type=float, default=0.001,
                                help='learning rate')
    train_settings.add_argument('--weight_decay', type=float, default=0,
                                help='weight decay')
    train_settings.add_argument('--dropout_keep_prob', type=float, default=1,
                                help='dropout keep rate')
    train_settings.add_argument('--batch_size', type=int, default=32,
                                help='train batch size')
    train_settings.add_argument('--epochs', type=int, default=15,
                                help='train epochs')

    model_settings = parser.add_argument_group('model settings')
    model_settings.add_argument('--algo', choices=['BIDAF', 'MLSTM'], default='BIDAF',
                                help='choose the algorithm to use')
    model_settings.add_argument('--embed_size', type=int, default=300,
                                help='size of the embeddings')
    model_settings.add_argument('--hidden_size', type=int, default=150,
                                help='size of LSTM hidden units')
    model_settings.add_argument('--max_p_num', type=int, default=5,
                                help='max passage num in one sample')
    model_settings.add_argument('--max_p_len', type=int, default=500,
                                help='max length of passage')
    model_settings.add_argument('--max_q_len', type=int, default=60,
                                help='max length of question')
    model_settings.add_argument('--max_a_len', type=int, default=200,
                                help='max length of answer')

    path_settings = parser.add_argument_group('path settings')
    path_settings.add_argument('--train_files', nargs='+',
                               default=['../input/demo/search.train.json'],
                               help='list of files that contain the preprocessed train data')
    path_settings.add_argument('--dev_files', nargs='+',
                               default=['../input/demo/search.dev.json'],
                               help='list of files that contain the preprocessed dev data')
    path_settings.add_argument('--test_files', nargs='+',
                               default=['../input/demo/search.test1.json'],
                               help='list of files that contain the preprocessed test data')

    # path_settings.add_argument('--train_files', nargs='+',
    #                            default=[
    #                                '/home/lq/projects/deep_learning/multi_passages_mrc/input/dureader_2.0_v4/final_mrc_dataset/trainset/search.train.json'],
    #                            help='list of files that contain the preprocessed train data')
    # path_settings.add_argument('--dev_files', nargs='+',
    #                            default=[
    #                                '/home/lq/projects/deep_learning/multi_passages_mrc/input/dureader_2.0_v4/final_mrc_dataset/devset/search.dev.json',
    #                                '/home/lq/projects/deep_learning/multi_passages_mrc/input/dureader_2.0_v4/final_mrc_dataset/devset/cleaned_18.search.dev.json'
    #                                ],
    #                            help='list of files that contain the preprocessed dev data')
    # path_settings.add_argument('--test_files', nargs='+',
    #                            default=[
    #                                '/home/lq/projects/deep_learning/multi_passages_mrc/input/dureader_2.0_v4/final_mrc_dataset/testset/search.test1.json'],
    #                            help='list of files that contain the preprocessed test data')

    path_settings.add_argument('--model_dir', default='cache/models/',
                               help='the dir to store models')
    path_settings.add_argument('--result_dir', default='cache/results/',
                               help='the dir to output the results')
    path_settings.add_argument('--summary_dir', default='cache/summary/',
                               help='the dir to write tensorboard summary')
    path_settings.add_argument('--log_path',
                               help='path of the log file. If not set, logs are printed to console')
    path_settings.add_argument('--pretrained_word_path', default='../../../pretrained_embeddings/chinese/sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5',
                               help='pretrained word path. If not set, word embeddings will be randomly init')
    return parser.parse_args()


def prepare(args):
    """
    checks data, creates the directories, prepare the vocabulary and embeddings
    """
    logger = logging.getLogger()
    logger.info('Checking the data files...')
    for data_path in args.train_files + args.dev_files + args.test_files:
        assert os.path.exists(data_path), '{} file does not exist.'.format(data_path)
    logger.info('Preparing the directories...')
    for dir_path in [args.vocab_dir, args.model_dir, args.result_dir, args.summary_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    if args.create_vocab:
        logger.info('Building vocabulary...')
        brc_data = Dataset(args.max_p_num, args.max_p_len,
                           args.max_q_len, args.train_files, badcase_sample_log_file=args.badcase_sample_log_file)
        vocab = Vocab(init_random=args.initial_tokens_random)
        for word in brc_data.word_iter('train'):
            vocab.add(word)

        unfiltered_vocab_size = vocab.size()
        vocab.filter_tokens_by_cnt(min_cnt=args.vocab_min_cnt)
        filtered_num = unfiltered_vocab_size - vocab.size()
        logger.info('After filter {} tokens, the final vocab size is {}'.format(
            filtered_num, vocab.size()))

        logger.info('Assigning embeddings...')
        if args.pretrained_word_path is not None:
            logger.info('load the pretrained word embeddings...')
            vocab.build_embedding_matrix(args.pretrained_word_path)
        else:
            logger.info('random init word embeddings...')
            vocab.randomly_init_embeddings(args.embed_size)

        logger.info('Saving vocab...')
        with open(os.path.join(args.vocab_dir, args.vocab_file), 'wb') as fout:
            pickle.dump(vocab, fout)

    logger.info('Done with preparing!')


def train(args):
    """
    trains the reading comprehension model
    """
    logger = logging.getLogger("brc")

    logger.info('check the directories...')
    for dir_path in [args.model_dir, args.result_dir, args.summary_dir]:
        if not os.path.exists(dir_path):
            logger.warning(
                "don't exist {} directory, so we create it!".format(dir_path))
            os.makedirs(dir_path)

    logger.info('Load data_set and vocab...')
    with open(os.path.join(args.vocab_dir, args.vocab_file), 'rb') as fin:
        vocab = pickle.load(fin)
    brc_data = Dataset(args.max_p_num, args.max_p_len, args.max_q_len,
                       args.train_files, args.dev_files,
                       badcase_sample_log_file=args.badcase_sample_log_file)
    logger.info('Converting text into ids...')
    brc_data.convert_to_ids(vocab, args.use_oov2unk)
    logger.info('Initialize the model...')
    rc_model = MultiAnsModel(vocab, args)
    logger.info('Training the model...')
    rc_model.train(brc_data, args.epochs, args.batch_size,
                   save_dir=args.model_dir,
                   save_prefix=args.algo,
                   dropout_keep_prob=args.dropout_keep_prob)
    logger.info('Done with model training!')


def evaluate(args):
    """
    evaluate the trained model on dev files
    """
    logger = logging.getLogger("brc")
    logger.info('Load data_set and vocab...')
    with open(os.path.join(args.vocab_dir, args.vocab_file), 'rb') as fin:
        vocab = pickle.load(fin)
    assert len(args.dev_files) > 0, 'No dev files are provided.'
    brc_data = Dataset(args.max_p_num, args.max_p_len,
                       args.max_q_len, dev_files=args.dev_files,
                       badcase_sample_log_file=args.badcase_sample_log_file)
    logger.info('Converting text into ids...')
    brc_data.convert_to_ids(vocab, args.use_oov2unk)
    logger.info('Restoring the model...')
    rc_model = MultiAnsModel(vocab, args)
    rc_model.restore(model_dir=args.model_dir, model_prefix=args.algo)
    logger.info('Evaluating the model on dev set...')
    dev_batches = brc_data.gen_mini_batches('dev', args.batch_size,
                                            pad_id=vocab.get_id(vocab.pad_token), shuffle=False)
    dev_loss, dev_bleu_rouge = rc_model.evaluate(
        dev_batches, result_dir=args.result_dir, result_prefix='dev.predicted')
    logger.info('Loss on dev set: {}'.format(dev_loss))
    logger.info('Result on dev set: {}'.format(dev_bleu_rouge))
    logger.info('Predicted answers are saved to {}'.format(
        os.path.join(args.result_dir)))


def predict(args):
    """
    predicts answers for test files
    """
    logger = logging.getLogger("brc")
    logger.info('Load data_set and vocab...')
    with open(os.path.join(args.vocab_dir, args.vocab_file), 'rb') as fin:
        vocab = pickle.load(fin)
    assert len(args.test_files) > 0, 'No test files are provided.'
    brc_data = Dataset(args.max_p_num, args.max_p_len, args.max_q_len,
                       test_files=args.test_files,
                       badcase_sample_log_file=args.badcase_sample_log_file)
    logger.info('Converting text into ids...')
    brc_data.convert_to_ids(vocab, args.use_oov2unk)
    logger.info('Restoring the model...')
    rc_model = MultiAnsModel(vocab, args)
    rc_model.restore(model_dir=args.model_dir, model_prefix=args.algo)
    logger.info('Predicting answers for test set...')
    test_batches = brc_data.gen_mini_batches('test', args.batch_size,
                                             pad_id=vocab.get_id(vocab.pad_token), shuffle=False)
    rc_model.evaluate(test_batches,
                      result_dir=args.result_dir, result_prefix='test.predicted')


def run():
    """
    Prepares and runs the whole system.
    """
    args = parse_args()

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if args.log_path:
        file_handler = logging.FileHandler(args.log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.info('Running with args: ')
    logger.info('*' * 100)
    for k, v in vars(args).items():
        logger.info('   {}:\t{}'.format(k, v))
    logger.info('*' * 100)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.prepare:
        prepare(args)
    if args.train:
        train(args)
    if args.evaluate:
        evaluate(args)
    if args.predict:
        predict(args)


if __name__ == '__main__':
    run()
