import numpy as np
import ipdb

def pro_data_times(dataset):
    path = '../data/' + dataset + '/train'

    click_index = np.load(path + '/click_news_indexes.npy')
    candidate_index = np.load(path + '/candidate_news_indexes.npy')
    news_cat_index = np.load(path + '/news_cat_index.npy')
    news_num = news_cat_index.shape[0]
    stages = np.load(path + '/click_stages.npy')
    popularity = np.zeros(news_num, max(stages))
    assert len(stages) == candidate_index.shape[0]
    # appearrence = np.zeros(news_num)
    # for x in click_index:
    #     for y in x:
    #         if y == 0:
    #             continue
    #         else:
    #             popularity[y] += 1
    for x in candidate_index:
        popularity[x[0]] += 1
        # for y in x[1:]:
        #     if y == 0:
        #         continue
        #     else:
        #         appearrence[y] += 1

    np.save(f"{path}/news_popularity.npy", np.array(popularity, dtype=np.int32))
    ipdb.set_trace()

def read_tsv(path, test):
    behavior_path = path + '/' + test + '/behaviors.tsv'
    news_path = path + '/' + test + '/news.tsv'
    print('============== {} dataset ==============='.format(test))
    
    user_list = {}
    diff_user = []
    news_list = []
    history_news = []
    impr_news = []
    with open(news_path, "r") as rd:
            for line in rd:
                nid, vert, subvert, title, ab, url, _, _ = line.strip("\n").split("\t")
                news_list.append(nid)
    with open(behavior_path, 'r') as rd:
        for line in rd:
            uid, time, history, impr = line.strip('\n').split('\t')[-4:]
            history = history.split()
            impr = impr.split()
            for hn in history:
                history_news.append(hn)
            for _in in impr:
                impr_news.append(_in[:-2])
            # break
            if uid in diff_user:
                continue
            if uid not in user_list.keys():
                user_list[uid] = [history, impr]
            else:
                his, imp = user_list[uid]
                if his != history:
                    print('user {} has the different history {} {}.'.format(uid, his, history))
                    diff_user.append(uid)

    
    news_list = list(set(news_list))
    user_list = list(set(user_list))
    impr_news = list(set(impr_news))
    history_news = list(set(history_news))
    print('{} dataset: news {} user {} impression_news {} history_news {}'.format(test, len(news_list), len(user_list), len(impr_news), len(history_news)))
    new_double = []
    # for x in impr_news:
    #     if x in history_news:
    #         new_double.append(x)
    # print('There are {} news both shown in history and impression'.format(len(new_double)))
    return list(set(news_list)), list(set(user_list)), impr_news, history_news

def count_news_users(dataset):
    
    path = './raw/' + dataset
    train_news, train_user, train_impr, train_his = read_tsv(path, 'train')
    dev_news, dev_user, dev_impr, dev_his = read_tsv(path, 'dev')
    cnt = 0
    for x in dev_impr:
        if x in train_impr:
            cnt += 1
            continue
        elif x in train_his:
            cnt += 1
        # else:
        #     if cnt < 5:
        #         print(x)
    print(f"{cnt/len(dev_impr):.2f}% news in dev dataset have also shown in train dataset")
    all_news = train_news + dev_news
    all_user = train_user + dev_user 
    if dataset != 'small':
        test_news, test_user, test_impr, test_his = read_tsv(path, 'test')
        all_news = all_news + test_news
        all_user = all_user + test_user

    all_news = list(set(all_news))
    all_user = list(set(all_user))
    print('all dataset: news {} user {}'.format(len(all_news), len(all_user)))

def analyse_userid(dataset):
    path_train = './raw/' + dataset + '/train/uindexes_list.npy'
    path_dev = './raw/' + dataset + '/train/uindexes_list.npy'


if __name__ == '__main__':
    # pro_data_times('small')
    count_news_users('large')