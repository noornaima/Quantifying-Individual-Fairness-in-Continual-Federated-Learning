from FLAlgorithms.users.userpFedCL import UserpFedCL
from FLAlgorithms.servers.serverbase import Server
from FLAlgorithms.generator.model import WGAN
from utils.model_utils import read_data, read_user_data, aggregate_user_data, create_generative_model, read_user_data_cl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision.utils import save_image
import os
import copy
import time

MIN_SAMPLES_PER_LABEL = 1
import random

# writer
from torch.utils.tensorboard import SummaryWriter


class FedCL(Server):
    def __init__(self, args, model, seed):
        super().__init__(args, model, seed)

        # Initialize data for all users
        self.data = read_data(args.dataset)
        data = self.data

        # data contains: clients, groups, train_data, test_data, proxy_data
        clients = data[0]
        total_users = len(clients)
        self.total_users = total_users
        self.total_test_samples = 0
        self.local = 'local' in self.algorithm.lower()
        self.use_adam = 'adam' in self.algorithm.lower()
        self.init_loss_fn()
        self.current_task = 0
        self.init_ensemble_configs()
        self.generator_private = None
        self.local_epochs = args.local_epochs
        self.old_classes = []
        self.new_classes = []
        self.user_metrics = {}

        # define generative_model as a AC-GAN

        if 'EMnist-B' in args.dataset:
            self.generator = WGAN(
                z_size=126,  # 100 + 26 (one-hot label)
                image_size=32,
                image_channel_size=1,
                c_channel_size=args.generator_c_channel_size,
                g_channel_size=args.generator_g_channel_size,
                dataset='EMNIST-B'
            )
            # initialize AC-GAN (server side)
            self.initialize_AC_GAN(args)
            self.unique_labels = 47

        elif 'EMnist-L' in args.dataset:
            self.generator = WGAN(
                z_size=126,  # 100 + 26 (one-hot label)
                image_size=32,
                image_channel_size=1,
                c_channel_size=args.generator_c_channel_size,
                g_channel_size=args.generator_g_channel_size,
                dataset='EMNIST-L'
            )
            # initialize AC-GAN (server side)
            self.initialize_AC_GAN(args)
            self.unique_labels = 26

        elif 'Mnist-cl-5' == args.dataset:
            self.generator = WGAN(
                z_size=args.generator_z_size,
                image_size=32,
                image_channel_size=1,
                c_channel_size=args.generator_c_channel_size,
                g_channel_size=args.generator_g_channel_size,
                dataset='MNIST'
            )
            # initialize AC-GAN (server side)
            self.initialize_AC_GAN(args)
            self.unique_labels = 10

        elif 'CIFAR10' in args.dataset:
            self.generator = WGAN(
                z_size=args.generator_z_size,
                image_size=32,
                image_channel_size=1,
                c_channel_size=args.generator_c_channel_size,
                g_channel_size=args.generator_g_channel_size,
                dataset='CIFAR10'
            )
            # initialize AC-GAN (server side)
            self.initialize_AC_GAN_CIFAR(args)
            self.unique_labels = 10

            # =========================
        # init users with task = 0
        # =========================
        self.users = []
        for i in range(total_users):
            id, train_data, test_data, label_info = read_user_data_cl(i, data, dataset=args.dataset, count_labels=True,
                                                                      task=0)

            # count total samples (accumulative)
            self.total_train_samples += len(train_data)
            self.total_test_samples += len(test_data)
            id, train, test = read_user_data_cl(i, data, dataset=args.dataset, task=0)
            id = int(id[-1])
            """
            # Ensure label_info['labels'] is a list before sampling
            labels_list = list(label_info['labels'])

            # Randomly choose a new set of classes for each user in each iteration
            num_classes_to_select = 2  # Specify the number of classes to select
            if len(labels_list) >= num_classes_to_select:
                selected_classes = random.sample(labels_list, num_classes_to_select)
                print(f"User {i} selected classes: {selected_classes}")
            else:
                selected_classes = labels_list  
                print(f"User selected classes: {selected_classes}")
            """
            # ============ initialize Users with data =============

            # generator for Clinets: using the same initialization
            g = copy.deepcopy(self.generator)
            # init.
            user = UserpFedCL(
                args,
                id,
                model,
                self.generator,
                train_data,
                test_data,
                label_info,
                g,
                use_adam=self.use_adam,
                my_model_name='fedcl',
                unique_labels=self.unique_labels
            )

            self.users.append(user)

            # update classes so far & current labels
            user.classes_so_far.extend(label_info['labels'])
            user.current_labels.extend(label_info['labels'])
            # user.classes_so_far = selected_classes  # Set to the selected classes instead of extending
            # user.current_labels = selected_classes  # Set to the selected classes instead of extending
            # print(user.classes_so_far)
        print("Number of Train/Test samples:", self.total_train_samples, self.total_test_samples)
        print("Data from {} users in total.".format(total_users))
        print("Finished creating FedAvg server.")

        # ==================
        # training devices:
        # ==================
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('Using device: ', device)

        # user models:
        for u in self.users:
            u.generator.to(device)

        # server model:
        self.generator.to(device)

    def train(self, args):

        N_TASKS = 5

        # Initialize the user metrics dictionary
        user_metrics = {user_id: [] for user_id in range(self.total_users)}
        round_wise_cosine_matrices = []

        for task in range(N_TASKS):
            # round_wise_cosine_matrices = []
            # Update old_classes and new_classes
            # self.old_classes = self.get_old_classes()
            # self.new_classes = self.get_new_classes()

            if task > 0:

                # Update old_classes before moving to the new task
                """
                for u in self.users:
                    self.old_classes.extend(self.new_classes)
                    self.old_classes = list(set(self.old_classes))  # Remove duplicates
                print("Old Classes after update:", self.old_classes)
                """
                # threshold_value = 20
                # self.test_(selected=False)
                # self.calculate_fairness_metric()
                self.calculate_class_accuracies()
                self.store_initial_accuracies_for_all_users()
                """
                for metric_name in ['Mean', 'Median', 'Standard Deviation', 'Variance', 'Composite Score']:
                    for metric_type in ['similarity', 'dissimilarity', 'only cosine', 'equation']:
                        self.plot_metric_over_time(metric_name, task - 1, metric_type)
                """
                """
                for metric_type in ['similarity', 'dissimilarity', 'equation']:
                    self.plot_chi_square_over_time(metric_name)
                """
                # self.plot_and_save_mean_matrices(glob_iter, task - 1)
                # self.plot_and_save_median_matrices(glob_iter, task - 1)
                # self.plot_and_save_std_matrices(glob_iter, task - 1)
                #self.plot_cumulative_boxplots(task - 1)
                #self.plot_individual_and_aggregated_scatter(task - 1)
                #self.elementwise_multiply_matrices_for_task(task - 1)
                #self.calculate_roundwise_statistics(task - 1)
                #self.plot_high_similarity_scatter(task - 1)
                #self.plot_individual_and_aggregated_boxplots(task - 1)
                #self.calculate_task_statistics(task - 1)
                self.plot_forgetting_measure_history(task -1)
                self.plot_bwt_history(task - 1)
                self.plot_round_statistics(task - 1)
                self.plot_round_fairness(task - 1)

                # self.plot_and_save_mode_matrices(glob_iter, task)

                # self.aggregate_metrics()

                self.evaluate_per_client_per_task_(selected=False, personal=False)

                self.end_of_task_processing(glob_iter)

                for u in self.users:
                    self.old_classes.extend(u.classes_past_task)
                self.old_classes = list(set(self.old_classes))

                # After all tasks and rounds are complete, you can access the stored metrics
                #for user_id, metrics in user_metrics.items():
                    #print(f"Metrics for User {user_id} across rounds: {metrics}")
                """    
                # At the end of the task, print each matrix
                print(f"--- Cosine Similarity Matrices for Task {task} ---")
                print(glob_iter)
                for glob_iter, matrix in enumerate(round_wise_cosine_matrices):
                    print(f"Round {glob_iter} Matrix:\n{matrix}\n")
                """
                for user_id in range(self.total_users):
                    user_metrics[user_id] = []

                # round_wise_cosine_matrices.clear()
            """
            if task > 0:
                print("Old Classes before update:")
                for u in self.users:
                    print("Client ID:", u.id, "Old Classes:", u.old_classes)

                for u in self.users:
                    u.old_classes.extend(u.new_classes)
                    u.old_classes = list(set(u.old_classes))  # Remove duplicates

                print("Old Classes after update:")
                for u in self.users:
                    print("Client ID:", u.id, "Old Classes:", u.old_classes) 
            """

            # ===================
            # The First Task
            # ===================
            if task == 0:

                # update labels info. for the first task
                available_labels = set()
                available_labels_current = set()
                for u in self.users:
                    available_labels = available_labels.union(set(u.classes_so_far))
                    available_labels_current = available_labels_current.union(set(u.current_labels))

                for u in self.users:
                    u.available_labels = list(available_labels)
                    u.available_labels_current = list(available_labels_current)


            # ===================
            # Initialize new Task
            # ===================
            else:
                self.current_task = task

                for i in range(self.total_users):

                    id, train_data, test_data, label_info = read_user_data_cl(i, self.data, dataset=args.dataset,
                                                                              count_labels=True, task=task)
                    self.users[i].next_task(train_data, test_data, label_info)
                    # update dataset 
                    #                     assert (self.users[i].id == id)
                    # self.users[i].next_task(train_data, test_data, label_info)

                    # update labels info.
                    available_labels = set()
                    available_labels_current = set()
                    for u in self.users:
                        available_labels = available_labels.union(set(u.classes_so_far))
                        available_labels_current = available_labels_current.union(set(u.current_labels))

                    for u in self.users:
                        u.available_labels = list(available_labels)
                        u.available_labels_current = list(available_labels_current)

                    self.new_classes = []  # Reset for the current task
                    for u in self.users:
                        self.new_classes.extend(u.current_labels)
                        # print("Client ID:", u.id, "New Classes for current task:", u.current_labels)
                    self.new_classes = list(set(self.new_classes))

                    # Ensure no overlap between old and new classes
                    # assert not set(self.old_classes).intersection(self.new_classes), "Overlap between old and new classes"

                # for u in self.users:
                #    print("Client ID:", u.id, "New Classes for current task:", u.current_labels)

                # print("Updated New Classes for the current task:", self.new_classes)

            # ===================
            #    print info.
            # ===================
            if True:
                for u in self.users:
                    print("Client ID:", u.id)
                    print("classes so far: ", u.classes_so_far)
                    # print("old classes:", u.classes_past_task)
                    print("available labels for the Client: ", u.available_labels)
                    print("available labels (current) for the Client: ", u.available_labels_current)
                    print("new classes", u.current_labels)

            # ===================
            #    visualization
            # ===================
            # 1. server side:
            # 2. user side:

            # ============ train ==============
            epoch_per_task = int(self.num_glob_iters / N_TASKS)

            for glob_iter_ in range(epoch_per_task):

                # ===================
                #    visualization
                # ===================
                if self.args.visual == 1:
                    my_classes_so_far = None
                    # 1. user side:
                    for num, u in enumerate(self.users):
                        my_classes_so_far = u.classes_so_far
                        for label in my_classes_so_far:
                            title = 'N' + str(args.naive) + '-' + str(self.current_task) + '-C' + str(num) + '-' + str(
                                label) + '-' + str(args.dataset)
                            visualize(self.generator, 25, title, [label])

                        my_classes_so_far = u.available_labels

                    # 2. user side:
                    for label in my_classes_so_far:
                        title = 'N' + str(args.naive) + '-' + str(self.current_task) + '-S-' + str(label) + '-' + str(
                            args.dataset)
                        visualize(self.generator, 25, title, [label])

                glob_iter = glob_iter_ + epoch_per_task * task

                print("\n\n------------- Round number:", glob_iter, " | Current task:", task, "-------------\n\n")

                # select users
                self.selected_users, self.user_idxs = self.select_users(glob_iter, self.num_users, return_idx=True)
                for u in self.users:
                    self.update_user_classes(u.id, u.classes_so_far)

                # if arg.algorithm contains "local". In most cases, it does not.
                # broadcast averaged prediction model to clients
                if not self.local:
                    assert self.mode == 'all'

                    # send parameteres: server -> client
                    if args.offline == 0:
                        self.send_parameters_(mode=self.mode, only_critic=args.only_critic)
                    else:
                        print('offline mode: No S-> sending')

                # At the end of each round, calculate and store the metrics for each user
                for user_id in range(self.total_users):
                    metrics = self.calculate_participation_metrics(user_id, glob_iter)
                    user_metrics[user_id].append(metrics)
                # cosine_matrix = self.calculate_fairness_metric()
                # round_wise_cosine_matrices.append(np.copy(cosine_matrix))
                # print(round_wise_cosine_matrices)
                # self.compute_delta_output_matrix()

                if glob_iter > 3:
                   self.avg_of_feature_set(glob_iter)

                self.elementwise_multiply_matrices(glob_iter, task)
                self.calculate_class_accuracies()

                chosen_verbose_user = np.random.randint(0, len(self.users))
                self.timestamp = time.time()  # log user-training start time

                # ---------------
                #   train user
                # ---------------
                '''
                1. regularization from global model: kd with its/global labels 
                2. regularization from itself
                '''

                for user_id, user in zip(self.user_idxs, self.selected_users):  # allow selected users to train
                    verbose = user_id == chosen_verbose_user

                    # perform regularization using generated samples after the first communication round
                    user.train(
                        glob_iter_,
                        self.generator,
                        glob_iter,
                        personalized=self.personalized,
                        verbose=verbose and glob_iter > 0,
                        regularization=glob_iter > 0)

                #                     user.generator.visualize()

                # log training time
                curr_timestamp = time.time()

                # Check if any users have been selected to avoid division by zero
                if len(self.selected_users) > 0:
                    train_time = (curr_timestamp - self.timestamp) / len(self.selected_users)
                    self.metrics['user_train_time'].append(train_time)
                else:
                    print("No users selected for training in this round, so train time cannot be calculated.")
                    # Optionally set train_time to a default value like 0 or None and handle it later
                    train_time = 0  # or None, depending on how you want to handle this case
                    # Since we can't calculate a meaningful train_time, you might choose not to append it
                    # self.metrics['user_train_time'].append(train_time)

                self.timestamp = time.time()

                # =================
                # 2. Server update
                # =================

                # ======== train a AC-GAN on the server side =======

                # =================== FedAVG ==================
                if args.fedavg == 1 or args.fedlwf == 1:
                    # send parameteres: client -> server
                    self.aggregate_parameters_(partial=False)

                    # =================== FedCL ===================
                elif args.fedcl == 1:
                    if args.offline == 0:
                        # FedAvg: initialize Gen with FedAvg
                        self.aggregate_parameters_(partial=False)

                        if args.naive == 0:
                            assert args.fedavg == 0
                            # fine tune the generator with client models
                            self.fine_tune_generator(args)
                        else:
                            print('naive mode: No fine-tune')

                    else:
                        print('offline mode: No C -> S aggregating')

                curr_timestamp = time.time()  # log  server-agg end time
                agg_time = curr_timestamp - self.timestamp
                print("Time Taken", agg_time)
                self.metrics['server_agg_time'].append(agg_time)

        # self.test_(selected=False)  # For selected users
        # self.test_per_task_(selected = False)
        # self.test_all_(selected=False, personal=True, matrix=False)  # For selected users on all classes
        self.calculate_class_accuracies()
        # self.calculate_fairness_metric()
        # self.aggregate_metrics()
        # self.print_most_participated_user()
        self.end_of_task_processing(glob_iter)

        # self.plot_and_save_mean_matrices(glob_iter, task-1)
        # self.plot_and_save_median_matrices(glob_iter, task-1)
        # self.plot_and_save_std_matrices(glob_iter, task - 1)
        # self.evaluate_all_(selected = False, matrix=True, personal=False)
        self.evaluate_per_client_per_task_(selected=False, personal=True)
        self.calculate_task_accuracies(selected=False, personal=True)
        #self.calculate_cumulative_fairness_ratio()

        #self.plot_cumulative_boxplots(N_TASKS - 1)
        #self.plot_individual_and_aggregated_scatter(N_TASKS - 1)
        #self.elementwise_multiply_matrices_for_task(N_TASKS - 1)
        #self.calculate_roundwise_statistics(N_TASKS - 1)
        #self.plot_high_similarity_scatter(N_TASKS - 1)
        #self.plot_individual_and_aggregated_boxplots(N_TASKS - 1)
        #self.calculate_task_statistics(N_TASKS - 1)
        self.plot_forgetting_measure_history(N_TASKS - 1)
        self.plot_bwt_history(N_TASKS - 1)
        self.plot_round_statistics(N_TASKS - 1)
        self.plot_round_fairness(N_TASKS - 1)
        self.evaluate_(personal=False)
        # self.save_results(args)
        self.save_model()

    def get_old_classes(self):
        if self.current_task == 0:
            return []
        old_classes = set()
        for task_classes in self.classes_in_tasks[:-1]:  # All tasks except the current one
            old_classes.update(task_classes)
        return list(old_classes)

    def get_new_classes(self):
        if self.current_task == 0:
            return list(self.classes_in_tasks[0]) if self.classes_in_tasks else []
        return list(self.classes_in_tasks[-1])

    def fine_tune_generator(self, args):
        # set to train mode
        self.generator.train()

        # train begins
        for iteration in range(args.md_iter):
            # holders:
            x_all = y_all = None
            x_all_ = y_all_ = None

            # get label info:
            labels_current = self.users[0].available_labels_current
            labels_all = self.users[0].available_labels

            # generate samples for current task:
            for user in self.selected_users:
                x, y = user.generator.sample(args.gen_batch_size, user.current_labels)
                x_all = torch.cat((x_all, x), axis=0) if x_all is not None else x
                y_all = torch.cat((y_all, y), axis=0) if y_all is not None else y

            # Before calling get_balanced_samples, check if x_all and y_all have been populated
            if x_all is not None and y_all is not None:
                x_balanced, y_balanced = self.get_balanced_samples(x_all, y_all, args)
            else:
                print("No samples generated for the current task. Skipping balance and training for this iteration.")
                continue  # Skip this iteration if no samples were generated

            # generate samples for previous task:
            if self.current_task != 0:
                for user in self.selected_users:
                    x_, y_ = user.last_generator.sample(user.gen_batch_size, user.classes_past_task)
                    x_all_ = torch.cat((x_all_, x_), axis=0) if x_all_ is not None else x_
                    y_all_ = torch.cat((y_all_, y_), axis=0) if y_all_ is not None else y_

                # Similarly, check before calling get_balanced_samples for previous task samples
                if x_all_ is not None and y_all_ is not None:
                    x_balanced_, y_balanced_ = self.get_balanced_samples(x_all_, y_all_, args)
                else:
                    print(
                        "No samples generated for the previous task. Proceeding without balancing previous task samples.")
                    x_balanced_ = y_balanced_ = None  # Ensure these are set to None if no samples

                result = self.generator.train_a_batch(
                    x_balanced, y_balanced, x_=x_balanced_, y_=y_balanced_,
                    importance_of_new_task=.5, classes_so_far=labels_all)

            else:
                result = self.generator.train_a_batch(
                    x_balanced, y_balanced, x_=None, y_=None,
                    importance_of_new_task=.5, classes_so_far=labels_current)
            """
            if self.current_task != 0:
                x_all_ = None
                y_all_ = None
                for user in self.selected_users:
                    # Check if the last_generator exists before sampling
                    if user.last_generator is not None:
                        x_, y_ = user.last_generator.sample(user.gen_batch_size, user.classes_past_task)
                        x_all_ = torch.cat((x_all_, x_), axis=0) if x_all_ is not None else x_
                        y_all_ = torch.cat((y_all_, y_), axis=0) if y_all_ is not None else y_
                    else:
                        print(f"User {user.id} does not have a last_generator.")

                if x_all_ is not None and y_all_ is not None:
                    x_balanced_, y_balanced_ = self.get_balanced_samples(x_all_, y_all_, args)
                    result = self.generator.train_a_batch(
                        x_balanced, y_balanced, x_=x_balanced_, y_=y_balanced_,
                        importance_of_new_task=0.5, classes_so_far=labels_all)
                else:
                    print("No previous task data available for replay.")
                    # Handle the case where there is no data for replay
                    result = self.generator.train_a_batch(
                        x_balanced, y_balanced, x_=None, y_=None,
                        importance_of_new_task=0.5, classes_so_far=labels_current)
            else:
                result = self.generator.train_a_batch(
                    x_balanced, y_balanced, x_=None, y_=None,
                    importance_of_new_task=0.5, classes_so_far=labels_current)
            """

    def get_balanced_samples(self, x_all, y_all, args):
        '''
        output: x and y
        '''
        x_balanced = y_balanced = []
        labels = set()

        # obtain label set:
        for i in y_all:
            labels.add(int(i))
        labels = list(labels)

        # dict of indexes of labels 
        d = {}
        for label in labels:
            d[label] = []

        for index, label in enumerate(y_all):
            label = int(label)
            d[label].append(index)

        y_balanced = np.random.choice(labels, args.gen_batch_size)

        # pick up x data:
        for label in y_balanced:
            x_balanced.append(random.choice(d[label]))  # d[label] is empty

        x_balanced = x_all[x_balanced]

        return x_balanced, torch.tensor(y_balanced)

    def get_label_weights(self):

        label_weights = []
        qualified_labels = []
        for label in range(self.unique_labels):

            weights = []
            for user in self.selected_users:
                weights.append(user.label_counts[label])

            # weights: 
            if np.max(weights) > MIN_SAMPLES_PER_LABEL:  # 1
                qualified_labels.append(label)

            # uniform
            if np.sum(weights) == 0:  # this label is unavailable for all clients this round
                label_weights.append(np.zeros(len(weights)))
            else:
                label_weights.append(np.array(weights) / np.sum(weights))

        label_weights = np.array(label_weights).reshape((self.unique_labels, -1))

        return label_weights, qualified_labels

    def get_label_so_far_global(self):
        l = set()
        for u in self.selected_users:
            c = u.classes_so_far()
            l = l.union(set(c))

        return list(l)

    def get_label_weights_all(self):
        # build labels so far:
        # w: [num of labels, num of Clients]

        qualified_labels = []
        weights = []
        one_hots = []

        for u in self.users:
            qualified_labels.extend(u.classes_so_far)
            weights.append(u.classes_so_far)

        # make it unique 
        qualified_labels = list(set(qualified_labels))

        # weights:
        for w in weights:
            one_hot = np.eye(self.unique_labels)[w].sum(axis=0)
            one_hots.append(one_hot)

        one_hots = np.array(one_hots)
        one_hots = np.transpose(one_hots)

        # normalize weights according to each label
        one_hots_sum = one_hots.sum(axis=1)

        # replace 0 with -1
        one_hots_sum[one_hots_sum == 0] = -1

        one_hots_sum = one_hots_sum.reshape(len(one_hots_sum), -1)

        one_hots = one_hots / one_hots_sum

        return one_hots, qualified_labels
