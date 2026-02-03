import time
import os
import torch
import yaml
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from Plan.baseline.DSAM_allocation import AttentionNet
from Monitoring.user_gen import gen_dataset

# 读取 config.yaml 文件
with open('../config.yaml', 'r') as file:
    config = yaml.safe_load(file)

train_data_size = config.get('train_data_size')
miu = config.get('eua_miu')
sigma = config.get('eua_sigma')
radius_low = config.get('eua_radius_low')
radius_high = config.get('eua_radius_high')
batch_size = config.get('batch_normal_size')
user_num = config.get('eua_train_user_num')
server_percent = config.get('eua_train_server_percent')
dataset_save_path = config.get('eua_train_dataset_save_path')
server_path = config.get('server_eua_path')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
lr = 1.0e-4  # 学习率
lr_decay = 0.96  # 衰减率
wait_best_reward_epoch = config.get('patience')
data_set = 'eua'


def DSAM_train_normal():
    # =======================
    # 数据加载
    # =======================
    data_type = {'train': [], 'valid': []}
    dataset = gen_dataset(
        user_num, train_data_size, server_path, dataset_save_path, server_percent,
        radius_low, radius_high, miu, sigma, device, data_type, data_set
    )
    train_data = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True)
    valid_data = DataLoader(dataset['valid'], batch_size=batch_size, shuffle=False)


    model = AttentionNet(6, 7, hidden_dim=256, device=device,
                              exploration_c=10,
                              capacity_reward_rate=0.5,
                              user_embedding_type='transformer',
                              server_embedding_type='linear',
                              transformer_n_heads=8,
                              transformer_n_layers=3,
                              transformer_feed_forward_hidden=512,
                              user_scale_alpha=0.0625)

    optimizer = Adam(model.parameters(), lr=lr)

    lr_scheduler = ExponentialLR(optimizer, lr_decay)
    print("当前学习率：", lr_scheduler.get_last_lr())
    epoch = 0
    all_valid_reward_list = []
    all_valid_user_list = []
    all_valid_server_list = []
    all_valid_capacity_list = []
    best_r = 0
    start_time = time.time()
    # 模型保存路径

    model_filename = "./model/DSMA/" + time.strftime('%m%d%H%M', time.localtime(time.time())) \
                     + "_server_" + str(server_percent) \
                     + "_user_" + str(user_num) \
                     + "_mean_" + str(miu) + "_std_" + str(sigma)

    if not os.path.exists(model_filename):
        os.makedirs(model_filename)
    while True:

        model.train()
        model.policy = 'sample'

        for batch_idx, (server_seq, user_seq, masks, latency) in enumerate(train_data):
            server_seq, user_seq, masks, latency= server_seq.to(device), user_seq.to(device), masks.to(device), latency.to(device)
            reward, actions_probs, _, allocated_users_num, user_allocated_props, server_used_props, capacity_used_props, _, \
            propagation_delay_aver= model(user_seq, server_seq, masks, latency)

            model.policy = 'greedy'
            with torch.no_grad():
                reward2, *_ = model(user_seq, server_seq, masks, latency)
                advantage = reward - reward2
            model.policy = 'sample'

            log_probs = torch.zeros(user_seq.size(0), device=device)
            for prob in actions_probs:
                log_prob = torch.log(prob)
                log_probs += log_prob
            log_probs[log_probs < -1000] = -1000.

            reinforce = torch.dot(advantage.detach(), log_probs)
            actor_loss = reinforce.mean()

            optimizer.zero_grad()
            actor_loss.backward()
            optimizer.step()

            if batch_idx % int(1024 / batch_size) == 0:
                log_message = (
                    'Epoch {}: Train [{}/{} ({:.1f}%)]\tR:{:.6f}\tuser_nums: {:.6f}\tuser_props: {:.6f}'
                    '\tserver_props: {:.6f}\tcapacity_props:{:.6f}'.format(
                        epoch,
                        (batch_idx + 1) * len(user_seq),
                        train_data_size['train'],
                        100. * (batch_idx + 1) / len(train_data),
                        -torch.mean(reward),
                        torch.mean(allocated_users_num),
                        torch.mean(user_allocated_props),
                        torch.mean(server_used_props),
                        torch.mean(capacity_used_props)))
                print(time.strftime('%m-%d %H:%M:%S', time.localtime(time.time())) + log_message)

        model.eval()
        model.policy = 'greedy'
        with torch.no_grad():
            # Validation
            valid_R_list = []
            valid_user_allocated_props_list = []
            valid_server_used_props_list = []
            valid_capacity_used_props_list = []
            for batch_idx, (server_seq, user_seq, masks, latency) in enumerate(valid_data):
                server_seq, user_seq, masks, latency = server_seq.to(device), user_seq.to(device), masks.to(device), latency.to(device)

                reward, _, _, allocated_users_num, user_allocated_props, server_used_props, capacity_used_props, _, \
                propagation_delay_aver= model(user_seq, server_seq, masks, latency)

                if batch_idx % int(1024 / batch_size) == 0:
                    log_message = (
                        'Epoch {}: Valid [{}/{} ({:.1f}%)]\tR:{:.6f}\tuser_nums: {:.6f}\tuser_props: {:.6f}'
                        '\tserver_props: {:.6f}\tcapacity_props:{:.6f}'.format(
                            epoch,
                            (batch_idx + 1) * len(user_seq),
                            train_data_size['train'],
                            100. * (batch_idx + 1) / len(train_data),
                            -torch.mean(reward),
                            torch.mean(allocated_users_num),
                            torch.mean(user_allocated_props),
                            torch.mean(server_used_props),
                            torch.mean(capacity_used_props)))
                    print(time.strftime('%m-%d %H:%M:%S', time.localtime(time.time())) + log_message)

                valid_R_list.append(reward)
                valid_user_allocated_props_list.append(user_allocated_props)
                valid_server_used_props_list.append(server_used_props)
                valid_capacity_used_props_list.append(capacity_used_props)

            valid_R_list = torch.cat(valid_R_list)
            valid_user_allocated_props_list = torch.cat(valid_user_allocated_props_list)
            valid_server_used_props_list = torch.cat(valid_server_used_props_list)
            valid_capacity_used_props_list = torch.cat(valid_capacity_used_props_list)
            valid_r = torch.mean(valid_R_list)
            valid_user_allo = torch.mean(valid_user_allocated_props_list)
            valid_server_use = torch.mean(valid_server_used_props_list)
            valid_capacity_use = torch.mean(valid_capacity_used_props_list)

            all_valid_reward_list.append(valid_r)
            all_valid_user_list.append(valid_user_allo)
            all_valid_server_list.append(valid_server_use)
            all_valid_capacity_list.append(valid_capacity_use)

            # 每次遇到更好的reward就保存一次模型，并且更新model_bl
            if valid_r < best_r:
                best_r = valid_r
                best_epoch_id = epoch
                best_time = 0
                print("目前本次reward最好: {}\n".format(valid_r))
                model_save_filename = model_filename + '/' + time.strftime(
                    '%m%d%H%M', time.localtime(time.time())
                ) + "_{:.2f}_{:.2f}_{:.2f}".format(all_valid_user_list[best_epoch_id] * 100,
                                                   all_valid_server_list[best_epoch_id] * 100,
                                                   all_valid_capacity_list[best_epoch_id] * 100) + '.mdl'
                torch.save(model.state_dict(), model_save_filename)
                print("模型已存储到: {}".format(model_save_filename))
            else:
                best_time += 1
                print("已经有{}轮效果没变好了\n".format(best_time))

        # 学习率衰减
        lr_scheduler.step()
        print("当前学习率：", lr_scheduler.get_last_lr())

        # 如果超过设定的epoch次数valid奖励都没有再提升，就停止训练
        if best_time >= wait_best_reward_epoch:
            break
        epoch = epoch + 1

    print("训练结束，第{}个epoch效果最好，最好的reward: {} ，用户分配率: {:.2f} ，服务器租用率: {:.2f} ，资源利用率: {:.2f}"
                .format(best_epoch_id, -best_r,
                        all_valid_user_list[best_epoch_id] * 100,
                        all_valid_server_list[best_epoch_id] * 100,
                        all_valid_capacity_list[best_epoch_id] * 100))
    end_time = time.time()
    print("训练时间: {:.2f}h".format(((end_time - start_time) / 3600)))


if __name__ == '__main__':
    if torch.cuda.is_available():
        DSAM_train_normal()
    else:
        print("cuda.unavailable!")
