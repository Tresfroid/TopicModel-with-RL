from config.file_handing import log
import numpy as np
import re
import time
ISOTIMEFORMAT='%Y-%m-%d %X'
from subprocess import Popen, PIPE
import torch
import torch.nn as nn
from model.loss import loss_function as loss_function
from sklearn.preprocessing import normalize

def print_topics(model, vocab, log_file, topic_num, write_to_log=True):
    vocab = {value:key for key,value in vocab.items()}
    n_topics = model.d_t
    highest_topic_list = []
    if n_topics > 1:
        log(log_file, "Topics:", write_to_log)
        weights = model.de.weight.detach().cpu().numpy()
#         mean_sparsity = 0.0
        for j in range(n_topics):
            highest_list = []
            order = list(np.argsort(weights[:, j]).tolist()) #返回的是数组值从小到大的索引值
            order.reverse()
            k = 0
            for i in order:
                if k>=topic_num:
                    break
                if i in vocab:
                    highest_list.append(vocab[i])
                    k+=1
            highest = ' '.join(highest_list)
            print("%d %s" % (j, highest))

            log(log_file, "%d %s" % (j, highest), write_to_log)

def get_reward_cv(model, vocab, log_file,topic_file,cuda):
    start =time.clock()
    vocab = {value:key for key,value in vocab.items()}
    n_topics = model.d_t
    cv_list = []
    topic_list = []
    if n_topics > 1:
        weights = model.de.weight.detach().cpu().numpy()
        for j in range(n_topics):
            highest_list = []
            order = list(np.argsort(weights[:, j]).tolist()) #返回的是数组值从小到大的索引值
            order.reverse()
            k = 0
            for i in order:
                if k>=5:
                    break
                if i in vocab:
                    highest_list.append(vocab[i])
                    k+=1
#             highest_list = [vocab[i] for i in order[:5]]
            topic_list.append(highest_list)

        f = open(topic_file,'w')
        for topic in topic_list:
            for word in topic:
                f.write(word+' ')
            f.write('\n')
        f.close()
        p = Popen(['/data/lengjia/jdk1.8.0_201/bin/java', '-jar', '/data/lengjia/topic_model/palmetto-0.1.0-jar-with-dependencies.jar', '/data/lengjia/topic_model/wiki/wikipedia_bd/','C_V',topic_file], stdin=PIPE, stdout=PIPE, stderr=PIPE)
        output, err = p.communicate(b"input data that is passed to subprocess' stdin")
        temp_list = output.decode("utf-8").split("\n")
        for line in temp_list:
            if re.search(r"\t(.*)\t",line):
                co_herence_cv = re.search(r"\t(.*)\t",line)
                cv = float(co_herence_cv.group(1))
                cv_list.append(cv)
        cv_vector =torch.Tensor(cv_list).unsqueeze(0).cuda(cuda)  #1*d_t
    end =time.clock()
    print('CV Running time: %s Seconds'%(end-start))
    return cv_vector

def evaluate(log_file, model, test_X_dataset, policy_prob, cuda):
    model.train(False)
    #     test_X = normalize(np.array(test_X, dtype='float32'), axis=1)
    #     n_items, dv = test_X.shape
    bound = 0
    batch_index_arrays_num = 0
    batch_num = 0

    for i, batch in enumerate(test_X_dataset):
        batch_data, batch_indices, batch_index_arrays = batch[0], batch[1], batch[2]
        if cuda is None:
            x = torch.from_numpy(np.array(batch_data, dtype='float32'))
            x_indices = torch.from_numpy(np.array(batch_indices, dtype='float32'))
        else:
            x = torch.from_numpy(np.array(batch_data, dtype='float32')).cuda(cuda)
            x_indices = torch.from_numpy(np.array(batch_indices, dtype='float32')).cuda(cuda)

        mean, logvar, p_x_given_h, x_new, x_indices_new, policy_action_mask, log_prob = model(x, x_indices, policy_prob,cuda)
        loss, nll_term, KLD, penalty = loss_function(x_new, mean, logvar, p_x_given_h, x_indices_new)

        counts_list = []
        for i in x_new:
            counts_list.append(torch.sum(i).detach().cpu())
        if np.mean(counts_list) != 0:
            bound += (loss.detach().cpu().numpy() / np.array(counts_list)).mean()
            batch_num += 1

    bound = np.exp(bound / float(batch_num))
    print("Estimated perplexity upper bound on test set = %0.3f" % bound)
    log(log_file, "Estimated perplexity upper bound on test set = %0.3f" % bound)

    return bound


def train(log_file, model_file, model, optimizer_tm, train_X_dataset, test_X_dataset,
          max_epochs, topic_num, temp_batch_num, vocab, topic_file, reward_w, p_w, cuda=True):
    print("Start Tranining")
    log(log_file, "Start Tranining")
    if cuda != None:
        model.cuda(cuda)

    #     train_X = normalize(np.array(train_X, dtype='float32'), axis=1)
    epochs_since_improvement = 0
    min_bound = np.inf
    # self_dictionary = Dictionary(parsed)
    cv_list = []
    vocab_num_list = []
    reward_history = torch.zeros(1, len(vocab)).cuda(cuda)
    policy_prob = torch.rand(1, len(vocab)).cuda(cuda)
    vocab_size = torch.range(0, len(vocab) - 1).unsqueeze(0)
    cv_time = 0
    start = time.clock()
    for epoch_i in range(max_epochs):
        print("\nEpoch %d" % epoch_i)
        print(time.strftime(ISOTIMEFORMAT, time.localtime()))
        log(log_file, "\nEpoch %d" % epoch_i)
        log(log_file, time.strftime(ISOTIMEFORMAT, time.localtime()))

        temp_batch_index = 0
        bound = 0
        model.train(True)
        for i, batch in enumerate(train_X_dataset):
            batch_data, batch_indices, batch_index_arrays = batch[0], batch[1], batch[2]
            if cuda is None:
                x = torch.from_numpy(np.array(batch_data, dtype='float32'))
                x_indices = torch.from_numpy(np.array(batch_indices, dtype='float32'))
            else:
                x = torch.from_numpy(np.array(batch_data, dtype='float32')).cuda(cuda)
                x_indices = torch.from_numpy(np.array(batch_indices, dtype='float32')).cuda(cuda)

            mean, logvar, p_x_given_h, x_new, x_indices_new, policy_action_mask, log_prob = model(x, x_indices,
                                                                                                  policy_prob, cuda)
            vocab_num_list.append(policy_action_mask.sum().detach().cpu().numpy())
            optimizer_tm.zero_grad()
            loss, nll_term, KLD, penalty = loss_function(x_new, mean, logvar, p_x_given_h, x_indices_new)  # size50)
            loss.mean().backward()  # size1 scaler
            nn.utils.clip_grad_norm(model.continuous_parameters(), max_norm=5)
            optimizer_tm.step()

            if i != 0 and i % temp_batch_num == 0:

                print("\nepoch batch nll_term KLD l1p loss")
                print("%d %d %0.4f %0.4f %0.4f %0.4f" % (epoch_i, i, nll_term.mean(), KLD.mean(), penalty, loss.mean()))
                log(log_file, "\nepoch batch nll_term KLD l1p loss")
                log(log_file,
                    "%d %d %0.4f %0.4f %0.4f %0.4f" % (epoch_i, i, nll_term.mean(), KLD.mean(), penalty, loss.mean()))
                bound = evaluate(log_file, model, test_X_dataset, policy_prob, cuda)

                if bound < min_bound:
                    print("New best dev bound = %0.3f" % bound)
                    log(log_file, "New best dev bound = %0.3f" % bound)
                    min_bound = bound
                    # print("Saving model")
                    # torch.save(model.state_dict(),model_file)
                    epochs_since_improvement = 0
                else:
                    epochs_since_improvement += 1
                    print("No improvement in %d batches(s)" % epochs_since_improvement)
                    log(log_file, "No improvement in %d batches(s)" % epochs_since_improvement)

        # Reinforcement Learning
        policy_action_mask = policy_action_mask.cpu()
        vocab_reverse = {value: key for key, value in vocab.items()}
        vocab_index = torch.mul(policy_action_mask.t(), vocab_size.t()).t().squeeze(0).numpy()
        vocab_new = {}
        for index in vocab_index:
            vocab_new[vocab_reverse[int(index)]] = int(index)

        start_cv = time.clock()
        batch_cv = get_reward_cv(model, vocab_new, log_file, topic_file, cuda)
        end_cv = time.clock()
        cv_time += (end_cv - start_cv)
        de_weight = model.de.weight.detach()
        dt = de_weight.size()[1]
        de_weight_norm = torch.from_numpy(normalize(de_weight.cpu(), axis=0)).float().cuda(cuda)
        det = torch.eye(dt).cuda() - torch.mm(de_weight_norm.t(), de_weight_norm)
        reward1 = torch.mm(batch_cv, de_weight.t()) * log_prob
        reward2 = 0.1 * abs(np.linalg.det(det.cpu().numpy()))
        reward = reward1 - reward2 - reward_history  # 1*5000
        policy_prob = policy_prob + p_w * reward
        reward_history = (1 - reward_w) * reward + reward_w * reward_history
        print('co_herence_cv: ' + str(torch.mean(batch_cv).detach().cpu().numpy()))
        log(log_file, "co_herence_cv: %0.3f" % float(float(torch.mean(batch_cv).detach().cpu().numpy())))
        # print_topics(model, vocab, log_file, topic_num, write_to_log=True)
        cv_list.append(torch.mean(batch_cv).detach().cpu().numpy())
        # if epochs_since_improvement >= 20:
        #    break

    print("The best dev bound = %0.3f" % min_bound)
    log(log_file, "The best dev bound = %0.3f" % min_bound)
    log(log_file, "Final topics:")
    print_topics(model, vocab_new, log_file, topic_num, write_to_log=True)
    end = time.clock()
    print('Running time: %s Seconds' % (end - start - cv_time))

    return cv_list, vocab_num_list, vocab_new


