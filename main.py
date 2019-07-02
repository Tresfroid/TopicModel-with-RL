import numpy as np
import os
from optparse import OptionParser
import torch
import torch.nn as nn
from model.model import Vocab_Selector,TopicModel
from train_model import train
from config.file_handing import log
import config.file_handing as fh
from data_process.topic_data import preprocess_data
from data_process.model_data import DataIter,load_data

def main():

    ########################           option            #############################################################
    usage = "%prog input_dir"
    parser = OptionParser(usage=usage)
    parser.add_option('--train_data_file', dest='train_data_file',
                      default="/data/lengjia/topic_model/20ng/train.json")
    parser.add_option('--test_data_file', dest='test_data_file',
                      default="/data/lengjia/topic_model/20ng/test.json")
    parser.add_option('--cuda', dest='cuda', default=0)
    parser.add_option('--batchsize', dest='batchsize', default=32)
    parser.add_option('--temp_batch_num', dest='temp_batch_num', default=300)
    parser.add_option('--dt', dest='dt', default=30, help='Number of topics: default=%default')
    parser.add_option('--vocab', dest='topic_vocabsize', default=5000)
    parser.add_option('--reward_w', dest='reward_w', default=0.5)
    parser.add_option('--p_w', dest='p_w', default=0.0001)
    parser.add_option('--max_epochs', dest='max_epochs', default=50)
    parser.add_option('--min_epochs', dest='min_epochs', default=10)

    parser.add_option('--lr', dest='lr', default=1e-4)
    parser.add_option('--de', dest='de', default=500)
    parser.add_option('--encoder_layers', dest='encoder_layers', default=1)
    parser.add_option('--generator_layers', dest='generator_layers', default=4)
    parser.add_option('--topic_num', dest='topic_num', default=30)
    parser.add_option('--input', dest='input_prefix', default='train')
    parser.add_option('--output', dest='output_prefix', default='output')
    parser.add_option('--test', dest='test_prefix', default='test')
    #######################################         parameter            #########################################################
    options, args = parser.parse_args()
    input_dir = args[0]
    train_data_file = options.train_data_file
    test_data_file = options.test_data_file
    cuda = int(options.cuda)
    batchsize = int(options.batchsize)
    temp_batch_num = int(options.temp_batch_num)
    dt = int(options.dt)
    topic_vocabsize = int(options.topic_vocabsize)
    reward_w = float(options.reward_w)
    p_w = float(options.p_w)
    max_epochs = int(options.max_epochs)
    min_epochs = int(options.min_epochs)

    lr = float(options.lr)
    de = int(options.de)
    encoder_layers = int(options.encoder_layers)
    generator_layers = int(options.generator_layers)
    topic_num = int(options.topic_num)

    input_prefix = options.input_prefix
    output_prefix = options.output_prefix
    test_prefix = options.test_prefix
    l1_strength = np.array(0.0, dtype=np.float32)
    model_file = os.path.join(input_dir, 'RL_TopicModel'+ '_batchsize' + str(batchsize) + '_vocab' + str(topic_vocabsize) +  '_reward_w' + str(reward_w) + '_p_w' + str(p_w) + '_dt' + str(dt)+ '.pkl')
    log_file = os.path.join(input_dir, 'RL_TopicModel'+ '_batchsize' + str(batchsize) + '_vocab' + str(topic_vocabsize) +  '_reward_w' + str(reward_w) + '_p_w' + str(p_w) + '_dt' + str(dt)+ '.log')
    topic_file = os.path.join(input_dir, 'RL_TopicModel'+ '_batchsize' + str(batchsize) + '_vocab' + str(topic_vocabsize) +  '_reward_w' + str(reward_w) + '_p_w' + str(p_w) + '_dt' + str(dt)+ '.txt')

#*********************************************************************************************************************************************************

    #### topic: data->one_hot
    preprocess_data(train_data_file, test_data_file, input_dir, topic_vocabsize, log_transform=False )
    train_X, vocab, train_labels, train_indices, train_index_arrays = load_data(input_dir, input_prefix, log_file)
    test_X, _, test_labels, test_indices, test_index_arrays = load_data(input_dir, test_prefix, log_file, vocab)
    # train_X = normalize(np.array(train_X, dtype='float32'), axis=1)
    # test_X = normalize(np.array(test_X, dtype='float32'), axis=1)
    print(train_X.shape, train_labels.shape)
    train_X_dataset = DataIter(train_X, train_indices, train_index_arrays, batchsize)
    test_X_dataset = DataIter(test_X, test_indices, test_index_arrays, batchsize)

    selector = Vocab_Selector()
    model = TopicModel(selector, topic_vocabsize, de, dt, encoder_layers, generator_layers, encoder_shortcut=False,
                       generator_shortcut=False, generator_transform=False)
    # model.load_state_dict(torch.load(model_file))
    optimizer_tm = torch.optim.Adam(model.continuous_parameters(), lr)
    # optimizer_pg = torch.optim.Adam(model.discrete_parameters(),lr)
    cv_list, vocab_num_list, vocab_new = train(log_file, model_file, model, optimizer_tm, train_X_dataset,test_X_dataset,
                                               max_epochs, topic_num, temp_batch_num, vocab, topic_file, reward_w, p_w, cuda=cuda)
    fh.write_to_json(vocab_new, os.path.join(input_dir, input_prefix + '.vocab_new.json'))


if __name__ == '__main__':
    main()
