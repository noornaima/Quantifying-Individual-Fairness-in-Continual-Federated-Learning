import torch
import random
import os
import numpy as np
import h5py
from utils.model_utils import get_dataset_name, RUNCONFIGS
import copy
import torch.nn.functional as F
import time
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity
from utils.model_utils import get_log_path, METRICS
from torch import optim
from scipy.stats import chisquare
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pickle
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from matplotlib.lines import Line2D
import plotly.express as px
import kaleido
from scipy.stats import linregress
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import zscore



class Server:
    def __init__(self, args, model, seed):

        # Set up the main attributes
        self.dataset = args.dataset
        self.num_glob_iters = args.num_glob_iters
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.total_train_samples = 0
        self.K = args.K
        self.args = args

        self.model = copy.deepcopy(model[0])
        self.model_name = model[1]

        self.users = []
        self.selected_users = []
        self.num_users = args.num_users
        self.beta = args.beta
        self.lamda = args.lamda
        self.algorithm = args.algorithm
        self.personalized = 'pFed' in self.algorithm
        self.mode = 'partial' if 'partial' in self.algorithm.lower() else 'all'
        self.seed = seed
        self.deviations = {}
        self.metrics = {key: [] for key in METRICS}
        self.timestamp = None
        self.save_path = args.result_path
        self.active_users = set(self.users)  # Start with all users active
        self.user_leave_probability = 0.2
        self.user_join_probability = 0.3
        self.recently_left_users = set()  # Keep track of users who recently left
        # Initialize a dictionary to keep track of how many times each user is selected
        self.user_participation_count = {}
        self.task_participation_count = {}
        self.selected_users_info = []
        self.total_rounds = 0
        self.user_participation_count = {}
        self.user_participation_rounds = {}
        self.fairness_metrics_similarity_history = []
        self.fairness_metrics_dissimilarity_history = []
        self.fairness_metrics_equation_history = []
        self.delta_output_matrix = None
        self.cosine_sim_matrix = None
        self.abs_multiplied_sim_matrix = None
        self.mean_cosine_sim_matrix_history = []
        self.mean_delta_output_matrix_history = []
        self.mean_multiplied_sim_matrix_history = []
        self.median_cosine_sim_matrix_history = []
        self.median_delta_output_matrix_history = []
        self.median_multiplied_sim_matrix_history = []
        self.mode_cosine_sim_matrix_history = []
        self.mode_delta_output_matrix_history = []
        self.mode_multiplied_sim_matrix_history = []
        self.fairness_metrics_cosine_history = []
        self.multiplied_dif_std_matrix_history = []
        self.user_classes = {}
        self.total_accuracies = {}
        self.all_rounds_cosine_data = []
        self.all_rounds_delta_data = []
        self.all_rounds_mul_inv_data = []
        self.all_rounds_del_data = []
        self.all_rounds_mul_sim_data = []
        self.task_summary_stats = []
        self.task_rounds_data = []
        self.task_scores = {}
        self.task_fairness_ratios = {}
        self.high_similarity_multiplied = []
        self.high_similarity_data = []
        self.last_successful_cosine_sim_matrix = None
        self.last_successful_delta_output_matrix = None
        self.consecutive_rounds_without_selection = 0
        self.regularity = {}
        self.user_averages = {}
        self.round_statistics = []
        self.task_acc_statistics = {}
        self.forgetting_measurements = []
        self.bwt_measurements = []
        self.fairness_ratio = 0
        self.fairness_threshold = 0.8
        self.current_technique = 'low participation'
        self.techniques = ['low participation', 'low accuracy', 'low average']
        self.technique_performance = {tech: [] for tech in self.techniques}
        self.moving_average_window = 5  # Number of rounds to consider for moving average

        os.system("mkdir -p {}".format(self.save_path))

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)

        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1, 0.02)
            m.bias.data.fill_(0)

    def initialize_AC_GAN_CIFAR(self, args):

        self.generator.critic.apply(self.weights_init)
        self.generator.generator.apply(self.weights_init)

        optimizerD = optim.Adam(self.generator.critic.parameters(), lr=args.lr_CIFAR, betas=(0.5, 0.999))
        optimizerG = optim.Adam(self.generator.generator.parameters(), lr=args.lr_CIFAR, betas=(0.5, 0.999))

        self.generator.set_generator_optimizer(optimizerG)
        self.generator.set_critic_optimizer(optimizerD)

    def initialize_AC_GAN(self, args):
        # define solver criterion and generators for the scholar model.
        beta1 = args.beta1
        beta2 = args.beta2
        lr = args.lr
        weight_decay = args.weight_decay

        generator_g_optimizer = optim.Adam(
            self.generator.generator.parameters(),
            lr=lr, weight_decay=weight_decay, betas=(beta1, beta2),
        )
        generator_c_optimizer = optim.Adam(
            self.generator.critic.parameters(),
            lr=lr, weight_decay=weight_decay, betas=(beta1, beta2),
        )

        self.generator.set_lambda(args.generator_lambda)
        self.generator.set_generator_optimizer(generator_g_optimizer)
        self.generator.set_critic_optimizer(generator_c_optimizer)

        # initialize model parameters
        self.gaussian_intiailize(self.generator, std=.02)

    def initialize_Classifier(self, args):

        beta1 = args.beta1
        beta2 = args.beta2
        lr = args.lr
        weight_decay = args.weight_decay

        self.classifier.optimizer = optim.Adam(
            self.classifier.critic.parameters(),
            lr=lr, weight_decay=weight_decay, betas=(beta1, beta2),
        )

        # initialize model parameters
        self.gaussian_intiailize(self.classifier.critic, std=.02)

        return

    def gaussian_intiailize(self, model, std=.01):

        # batch norm is not initialized
        modules = [m for n, m in model.named_modules() if 'conv' in n or 'fc' in n]
        parameters = [p for m in modules for p in m.parameters()]

        for p in parameters:
            if p.dim() >= 2:
                nn.init.normal_(p, mean=0, std=0.02)
            else:
                nn.init.constant_(p, 0)

        # normalization for batch norm
        modules = [m for n, m in model.named_modules() if 'bn' in n]

        for m in modules:
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def init_ensemble_configs(self):
        #### used for ensemble learning ####
        dataset_name = get_dataset_name(self.dataset)
        self.ensemble_lr = RUNCONFIGS[dataset_name].get('ensemble_lr', 1e-4)
        self.ensemble_batch_size = RUNCONFIGS[dataset_name].get('ensemble_batch_size', 128)
        self.ensemble_epochs = RUNCONFIGS[dataset_name]['ensemble_epochs']
        self.num_pretrain_iters = RUNCONFIGS[dataset_name]['num_pretrain_iters']
        self.temperature = RUNCONFIGS[dataset_name].get('temperature', 1)
        self.unique_labels = RUNCONFIGS[dataset_name]['unique_labels']
        self.ensemble_alpha = RUNCONFIGS[dataset_name].get('ensemble_alpha', 1)
        self.ensemble_beta = RUNCONFIGS[dataset_name].get('ensemble_beta', 0)
        self.ensemble_eta = RUNCONFIGS[dataset_name].get('ensemble_eta', 1)
        self.weight_decay = RUNCONFIGS[dataset_name].get('weight_decay', 0)
        self.generative_alpha = RUNCONFIGS[dataset_name]['generative_alpha']
        self.generative_beta = RUNCONFIGS[dataset_name]['generative_beta']
        self.ensemble_train_loss = []
        self.n_teacher_iters = 5
        self.n_student_iters = 1
        print("ensemble_lr: {}".format(self.ensemble_lr))
        print("ensemble_batch_size: {}".format(self.ensemble_batch_size))
        print("unique_labels: {}".format(self.unique_labels))

    def if_personalized(self):
        return 'pFed' in self.algorithm or 'PerAvg' in self.algorithm

    def if_ensemble(self):
        return 'FedE' in self.algorithm

    def send_parameters(self, mode='all', beta=1, selected=False):
        users = self.users
        if selected:
            assert (self.selected_users is not None and len(self.selected_users) > 0)
            users = self.selected_users

        for user in users:
            if mode == 'all':  # share all parameters
                user.set_parameters(self.model, beta=beta)
            else:  # share a part parameters
                user.set_shared_parameters(self.model, mode=mode)

    def send_parameters_(self, mode='all', beta=1, selected=False, only_critic=False, gr=False):
        users = self.users
        if selected:
            assert (self.selected_users is not None and len(self.selected_users) > 0)
            users = self.selected_users

        for user in users:
            if gr == True:
                user.set_parameters_(self.generator, beta=beta, only_critic=only_critic, mode=mode, gr=gr,
                                     classifier=self.classifier)  # classifier: from server
            else:
                user.set_parameters_(self.generator, beta=beta, only_critic=only_critic, mode=mode, gr=gr)

    def add_parameters(self, user, ratio, partial=False):
        if partial:
            for server_param, user_param in zip(self.model.get_shared_parameters(), user.model.get_shared_parameters()):
                server_param.data = server_param.data + user_param.data.clone() * ratio
        else:
            # replace all!
            for server_param, user_param in zip(self.model.parameters(), user.model.parameters()):
                server_param.data = server_param.data + user_param.data.clone() * ratio

    def add_parameters_(self, user, ratio, partial=False, gr=False):

        if gr == False:
            for server_param, user_param in zip(self.generator.parameters(), user.generator.parameters()):
                server_param.data = server_param.data + user_param.data.clone() * ratio
        else:
            for server_param, user_param in zip(self.classifier.critic.parameters(),
                                                user.classifier.critic.parameters()):
                server_param.data = server_param.data + user_param.data.clone() * ratio

    def aggregate_parameters(self, partial=False):
        assert (self.selected_users is not None and len(self.selected_users) > 0)

        if partial:
            for param in self.model.get_shared_parameters():
                param.data = torch.zeros_like(param.data)
        else:
            for param in self.model.parameters():
                param.data = torch.zeros_like(param.data)  # initilize w with zeros

        total_train = 0
        for user in self.selected_users:
            total_train += user.train_samples  # length of the train data for weighted importance

        for user in self.selected_users:
            self.add_parameters(user, user.train_samples / total_train, partial=partial)

    def aggregate_parameters_(self, partial=False, gr=False):
        '''
        Clients -> Server model
        '''
        # Check if there are any selected users. If not, skip the aggregation process.
        if self.selected_users is None or len(self.selected_users) == 0:
            print("No users selected for aggregation. Skipping this step.")
            return  # Early return to skip the function execution

        if gr == False:
            for param in self.generator.parameters():
                param.data = torch.zeros_like(param.data)  # initialize w with zeros
        else:
            for param in self.classifier.critic.parameters():
                param.data = torch.zeros_like(param.data)  # initialize w with zeros

        total_train = 0
        for user in self.selected_users:
            total_train += user.train_samples  # length of the train data for weighted importance

        for user in self.selected_users:
            self.add_parameters_(user, user.train_samples / total_train, partial=partial, gr=gr)

    def save_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model, os.path.join(model_path, "server" + ".pt"))

    def load_model(self):
        model_path = os.path.join("models", self.dataset, "server" + ".pt")
        assert (os.path.exists(model_path))
        self.model = torch.load(model_path)

    def model_exists(self):
        return os.path.exists(os.path.join("models", self.dataset, "server" + ".pt"))

    def select_users(self, round, num_users, return_idx=False):
        '''Selects num_users clients, prioritizing by low average of feature set from the available pool of users.
        Args:
            num_users: number of users to select; defaults to len(self.users) if not specified.
            round: the current round of selection.
            return_idx: whether to return the indices of selected users.
        Returns:
            A list of selected user objects and optionally their indices.
        '''
        if num_users == len(self.users):
            print("All users are selected")
            return self.users, [i for i in range(len(self.users))]

        num_users = min(num_users, len(self.users))

        if round > 6:
            print("select on the basis of average value")
            # Presuming self.user_averages is updated in calculate_participation_metrics

            sorted_users = sorted(self.users, key=lambda user: self.user_averages.get(user.id, float('inf')))
            selected_users = sorted_users[:num_users]
        else:
            # Random selection for the first round
            selected_users = random.sample(self.users, num_users)

        # Update and print participation info for each selected user
        for user in selected_users:
            user_id = user.id
            self.user_participation_count[user_id] = self.user_participation_count.get(user_id, 0) + 1
            self.task_participation_count[user_id] = self.task_participation_count.get(user_id, 0) + 1
            if user_id not in self.user_participation_rounds:
                self.user_participation_rounds[user_id] = []
            self.user_participation_rounds[user_id].append(round)

            # Print user ID and regularity score for each selected user
            print(f"User {user_id} selected with average score: {self.user_averages.get(user_id, 'N/A')} in round {round}")

        # Print the total number of users selected
        print(f"Round {round}: Selected User IDs: {[user.id for user in selected_users]}")
        print(f"Number of users selected in round {round}: {len(selected_users)}")

        if return_idx:
            selected_indices = [self.users.index(user) for user in selected_users] if selected_users else []
            return selected_users, selected_indices
        else:
            return selected_users, None

    def end_of_task_processing(self, round):
        # Print the total participation counts for each user in this task
        for user_id, total_count in self.task_participation_count.items():
            print(f"Client ID {user_id}: {total_count} participation(s) in this task")

        if self.task_participation_count:
            # Find the maximum number of participations
            max_part = max(self.task_participation_count.values())
            # Find all users who have this maximum participation count
            most_active_user = [user_id for user_id, total_count in self.task_participation_count.items() if
                                total_count == max_part]

            print(f"The most active user(s) in this task: {most_active_user} with {max_part} participations.")
        else:
            print("No participations in this task.")
            # Since there are no participations, you might set max_part to a default value or handle it differently
            max_part = 0  # or None, or however you wish to handle this case

        # Optionally, print the most active users here
        # Note: Ensure this function handles the case where max_part is 0 or None as set above
        self.print_most_participated_user()

        # Reset participation data for the next task
        self.selected_users_info = []
        self.task_participation_count = {}

    def print_most_participated_user(self):
        if not self.user_participation_count:
            print("No user participation data available.")
            return
        print("User participation counts:")
        for user_id, count in sorted(self.user_participation_count.items(), key=lambda item: item[1], reverse=True):
            print(f"User ID {user_id}: {count} participation(s)")
        # Find the maximum number of participations
        max_participation = max(self.user_participation_count.values())
        # Find all users who have this maximum participation count
        most_active_users = [user_id for user_id, count in self.user_participation_count.items() if
                             count == max_participation]

        print(f"The most active user(s): {most_active_users} with {max_participation} participations.")

    def get_user_participation(self):
        # Returns the participation count dictionary
        return self.user_participation_count

    def calculate_intervals_between_participation(self, user_id):

        if user_id not in self.user_participation_rounds or len(self.user_participation_rounds[user_id]) < 2:
            return []

        participation_rounds = self.user_participation_rounds[user_id]
        intervals = [participation_rounds[i] - participation_rounds[i - 1] for i in range(1, len(participation_rounds))]
        return intervals

    def calculate_participation_metrics(self, user_id, round):
        frequency = self.user_participation_count.get(user_id, 0)
        intervals = self.calculate_intervals_between_participation(user_id)
        regularity = np.std(intervals) if intervals else 0
        trend = self.calculate_participation_trend(user_id)
        participation = self.task_participation_count.get(user_id, 0)
        average_interval = np.mean(intervals) if intervals else 0
        regInv = 1 - regularity  # Calculate 1 minus regularity
        avgInv = 1 - average_interval  # Calculate 1 minus average interval
        # Fetch classes_so_far for the user
        classes_so_far = self.user_classes.get(user_id, [])
        classes_so_far_length = len(classes_so_far)
        if round > 3:
            metrics = [frequency, regInv, trend, participation, avgInv, classes_so_far_length]
            return ([frequency, regularity, trend, participation, average_interval, classes_so_far_length],
                    metrics)  # Return both sets of metrics

        print(
            f"User ID {user_id}: Frequency: {frequency}, Regularity: {regularity:.2f}, Trend: {trend}, Participation: {participation}, Intervals: {intervals}, Average_intervals: {average_interval},  Length of Classes: {classes_so_far_length}")

        return ([frequency, regularity, trend, participation, average_interval, classes_so_far_length], None)

    def avg_of_feature_set(self, round):
        num_users = len(self.users)
        raw_participation_vectors = []
        user_averages = {}  # Store user averages separately

        for user_id in range(num_users):
            participation_metrics = None
            if round > 3:
                # Calculate participation metrics for active users
                _, participation_metrics = self.calculate_participation_metrics(user_id, round)

            raw_participation_vectors.append(participation_metrics)

        normalized_metrics = self.normalize_metrics(raw_participation_vectors)

        for i, normalized_metric in enumerate(normalized_metrics):
            average_metric = np.mean(normalized_metric)
            user_averages[i] = average_metric

        print("User Averages:", user_averages)

        self.user_averages = user_averages

    def normalize_metrics(self, raw_metrics):
        # Convert to NumPy array for mathematical operations
        raw_metrics_array = np.array(raw_metrics)

        # Compute minimum and maximum values
        min_val = np.min(raw_metrics_array, axis=0)
        max_val = np.max(raw_metrics_array, axis=0)

        # Avoid division by zero
        range_val = max_val - min_val
        range_val[range_val == 0] = 1  # Set zero ranges to 1 to avoid division by zero

        # Normalize the metrics using Min-Max scaling
        normalized_metrics = (raw_metrics_array - min_val) / range_val
        #print("hi", normalized_metrics)

        return normalized_metrics

    def normalize_and_append_metrics(self, raw_metrics, user_id):
        num_users = len(self.users)
        normalized_metrics = self.normalize_metrics(raw_metrics)
        average_metric = np.mean(normalized_metrics)
        #print("normalized", normalized_metrics)
        print("Avg", average_metric)
        self.user_averages[user_id] = average_metric
    def normalize_all_metrics(self, raw_metrics):
        # Find the maximum length of inner lists
        max_length = max(len(metrics) for metrics in raw_metrics)
        # Extend shorter lists with zeros
        extended_metrics = [metrics + [0] * (max_length - len(metrics)) for metrics in raw_metrics]
        # Convert to NumPy array for mathematical operations
        raw_metrics_array = np.array(extended_metrics)

        # Compute minimum and maximum
        min_val = np.min(raw_metrics_array, axis=0)
        max_val = np.max(raw_metrics_array, axis=0)

        # Prevent division by zero by setting min=max to 1 in the range
        range_val = max_val - min_val
        range_val[range_val == 0] = 1  # Avoid division by zero

        # Normalize the metrics using Min-Max scaling
        normalized_metrics = (raw_metrics_array - min_val) / range_val

        # Optional: If you want to ensure that all values are strictly positive and within [0, 1],
        # you can skip adjusting for zero ranges as the division would not be problematic given the non-negativity of the data.

        return normalized_metrics

    def update_user_classes(self, user_id, classes):
        self.user_classes[user_id] = classes

    def calculate_participation_trend(self, user_id):

        total_rounds = self.total_rounds
        mid_point = total_rounds // 2
        early_participation = sum(1 for round in self.user_participation_rounds.get(user_id, []) if round <= mid_point)
        later_participation = self.task_participation_count.get(user_id, 0) - early_participation
        trend = later_participation - early_participation

        return trend

    def update_participation_counts(self, participating_users):
        # Update participation counts after each training round
        print(f"Updating participation counts for round: Users involved: {[user.id for user in participating_users]}")
        for user in participating_users:
            if user.id not in self.user_participation_count:
                self.user_participation_count[user.id] = 1
            else:
                self.user_participation_count[user.id] += 1
        self.total_rounds += 1

    def check_participation_equity(self):
        # Check if users meet the minimum participation rate
        minimum_participation_rate = 0.75  # 75%
        minimum_required_rounds = self.total_rounds * minimum_participation_rate

        underrepresented_users = []
        for user_id, participation_count in self.user_participation_count.items():
            if participation_count < minimum_required_rounds:
                underrepresented_users.append(user_id)

        if underrepresented_users:
            print("Underrepresented Users (participated in less than 75% of rounds):")
            for user_id in underrepresented_users:
                print(f"User ID {user_id}")

        return underrepresented_users

    def init_loss_fn(self):
        self.loss = nn.NLLLoss()
        self.ensemble_loss = nn.KLDivLoss(reduction="batchmean")  # ,log_target=True)
        self.ce_loss = nn.CrossEntropyLoss()

    def save_results(self, args):
        alg = get_log_path(args, args.algorithm, self.seed, args.gen_batch_size)
        with h5py.File("./{}/{}.h5".format(self.save_path, alg), 'w') as hf:
            for key in self.metrics:
                hf.create_dataset(key, data=self.metrics[key])
            hf.close()

    def store_initial_accuracies_for_all_users(self):
        """
        Calls the _store_initial_accuracies method for each user to store the initial accuracies
        of classes after a task is completed.

        Args:
            task: The task that has just been completed.
        """
        for user in self.users:
            _, _, _, logits, labels = user.test_data_sofar(personal=True)
            user._store_initial_accuracies(logits, labels)

    def calculate_class_accuracies(self):
        old_accuracies = {}
        current_accuracies = {}
        forgetting_measures = {}
        bwt_per_user = {}
        cfi_per_user = {}
        total_acc = {}  # Keep this as a dictionary

        all_total_accuracies = []  # List to store all total accuracies
        task_accuracies = {}  # Dictionary to store accuracies for each task

        for user in self.users:
            _, _, _, logits, labels = user.test_data_sofar(personal=True)
            # Calculate accuracy for this user
            user_total_acc, old_acc, current_acc, avg_old_acc, avg_current_acc, old_class_accuracies, current_class_accuracies, avg_retention_rate, forgetting_measure, bwt= user._calculate_accuracy(
                logits, labels)

            # Update dictionaries with this user's data
            total_acc[user.id] = user_total_acc  # Use a different variable here
            all_total_accuracies.append(user_total_acc * 100)  # Add the total accuracy as percentage
            old_accuracies[user.id] = old_acc
            current_accuracies[user.id] = current_acc
            forgetting_measures[user.id] = forgetting_measure
            bwt_per_user[user.id] = bwt


            # Update task-wise accuracies
            for old_class, (acc, task) in old_class_accuracies.items():
                if task not in task_accuracies:
                    task_accuracies[task] = []
                task_accuracies[task].append(acc)
            for current_class, acc in current_class_accuracies.items():
                if 'current' not in task_accuracies:
                    task_accuracies['current'] = []
                task_accuracies['current'].append(acc)

            print(f"\nUser {user.id}:")
            print(f"Total Accuracy: {total_acc[user.id] * 100:.2f}%")
            print(f"Old Accuracy: {old_acc:.2f}%")
            for old_class, (acc, task) in old_class_accuracies.items():
                if isinstance(acc, tuple):
                    correct, total = acc
                    acc_percentage = correct / total if total != 0 else 0
                    print(f"Class {old_class} (Old): {acc_percentage:.2f}% (Correct: {correct}, Total: {total})")
                else:
                    print(f"Class {old_class} (Introduced in Task {task}): {acc:.2f}% ")
            print(f"Current Accuracy: {current_acc:.2f}%")
            for current_class, acc in current_class_accuracies.items():
                if isinstance(acc, tuple):
                    correct, total = acc
                    acc_percentage = 100 * correct / total if total != 0 else 0
                    print(
                        f"Class {current_class} (Current): {acc_percentage:.2f}% (Correct: {correct}, Total: {total})")
                else:
                    print(f"Class {current_class} (Current): {acc:.2f}%")

            print(f"User {user.id} Average Retention Rate: {avg_retention_rate:.2f}%")
            print(f"Forgetting Measure: {forgetting_measure:.2f}, BWT: {bwt:.2f}")

        # After collecting all accuracies, calculate and print overall mean and standard deviation
        mean_total_accuracy = np.mean(all_total_accuracies)
        std_total_accuracy = np.std(all_total_accuracies, ddof=1)  # Using sample standard deviation

        # Initialize task-wise statistics dictionary if not already initialized
        if not hasattr(self, 'task_acc_statistics'):
            self.task_acc_statistics = {}

        # Update task-wise statistics in the global dictionary
        task_mean_std = self.calculate_acc_statistics(task_accuracies)
        for task, (mean_acc, std_acc) in task_mean_std.items():
            if task in self.task_acc_statistics:
                # If the task already exists, append the new mean and std values to the existing ones
                self.task_acc_statistics[task]['mean'].append(mean_acc)
                self.task_acc_statistics[task]['std'].append(std_acc)

            else:
                # If the task does not exist in the dictionary, create a new entry
                self.task_acc_statistics[task] = {'mean': [mean_acc], 'std': [std_acc]}



        # Update overall statistics
        self.round_statistics.append((mean_total_accuracy, std_total_accuracy))
        print("Overall Stat", [(f"{mean_acc:.2f}", f"{std_acc:.2f}") for mean_acc, std_acc in self.round_statistics])

        print(f"\nOverall Mean Total Accuracy: {mean_total_accuracy:.2f}")
        print(f"Overall Standard Deviation of Total Accuracy: {std_total_accuracy:.2f}")

        self.total_accuracies = total_acc
        avg_forgetting_measure = sum(forgetting_measures.values()) / len(
            forgetting_measures) if forgetting_measures else 0
        avg_bwt = sum(bwt_per_user.values()) / len(bwt_per_user) if bwt_per_user else 0

        # Calculate standard deviation of forgetting measure and BWT
        std_forgetting_measure = np.std(list(forgetting_measures.values()), ddof=1)
        std_bwt = np.std(list(bwt_per_user.values()), ddof=1)

        # Append the average forgetting measure, BWT, and their standard deviations to the respective lists
        self.forgetting_measurements.append((avg_forgetting_measure, std_forgetting_measure))
        #print("Forgetting", self.forgetting_measurements)
        self.bwt_measurements.append((avg_bwt, std_bwt))
        print(
            f"Average Forgetting Measure: {avg_forgetting_measure:.2f}, Average BWT: {avg_bwt:.2f}")
        return total_acc, old_accuracies, current_accuracies, avg_forgetting_measure

    def calculate_acc_statistics(self, task_accuracies):
        task_mean_std = {}
        for task, accuracies in task_accuracies.items():
            if accuracies:
                mean_accuracy = np.mean(accuracies)
                std_accuracy = np.std(accuracies, ddof=1)
                task_mean_std[task] = (mean_accuracy, std_accuracy)
        return task_mean_std

    def plot_forgetting_measure_history(self, task):
        # Plotting average forgetting measure history
        print("Overall Forget",
              [(f"{mean_acc:.2f}", f"{std_acc:.2f}") for mean_acc, std_acc in self.forgetting_measurements])
        rounds = range(1, len(self.forgetting_measurements) + 1)
        avg_measurements = [stat[0] for stat in self.forgetting_measurements]
        std_measurements = [stat[1] for stat in self.forgetting_measurements]

        plt.figure(figsize=(16, 12))
        plt.errorbar(rounds, avg_measurements, yerr=std_measurements, fmt='-o', capsize=5)
        plt.xlabel('Round')
        plt.ylabel('Average Forgetting Measure')
        plt.title(f'Average Forgetting Measure per Round for Task {task}')
        plt.grid(True)

        # Save the plot
        save_dir = 'forgetting_measure_plots'
        os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist
        filename = os.path.join(save_dir, f'forgetting_measure_task{task}.png')
        plt.savefig(filename)
        plt.close()  # Close the plot to free up memory
        print(f"Forgetting measure plot saved to {filename}")
        self.forgetting_measurements.clear()

    def plot_bwt_history(self, task):
        # Plotting average BWT history
        print("Overall bwt", [(f"{mean_acc:.2f}", f"{std_acc:.2f}") for mean_acc, std_acc in self.bwt_measurements])
        rounds = range(1, len(self.bwt_measurements) + 1)
        avg_measurements = [stat[0] for stat in self.bwt_measurements]
        std_measurements = [stat[1] for stat in self.bwt_measurements]

        plt.figure(figsize=(16, 12))
        plt.errorbar(rounds, avg_measurements, yerr=std_measurements, fmt='-o', capsize=5)
        plt.xlabel('Round')
        plt.ylabel('Average BWT')
        plt.title(f'Average BWT per Round for Task {task}')
        plt.grid(True)

        # Save the plot
        save_dir = 'bwt_plots'
        os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist
        filename = os.path.join(save_dir, f'bwt_task{task}.png')
        plt.savefig(filename)
        plt.close()  # Close the plot to free up memory
        print(f"BWT plot saved to {filename}")
        self.bwt_measurements.clear()
    def plot_round_statistics(self, task):
        # This method should be called at the end of the task to plot the statistics
        rounds = range(1, len(self.round_statistics) + 1)
        means = [stat[0] for stat in self.round_statistics]
        std_devs = [stat[1] for stat in self.round_statistics]

        # Create the plot
        plt.figure(figsize=(16, 12))
        plt.errorbar(rounds, means, yerr=std_devs, fmt='-o', capsize=5)
        plt.xlabel('Round')
        plt.ylabel('Accuracy (%)')
        plt.title(f'Mean and Standard Deviation of Accuracies per Round for Task {task}')
        plt.xticks(rounds)
        plt.grid(True)

        # Save the plot
        save_dir = 'round_statistics_plots'
        os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist
        round_statistics_filename = os.path.join(save_dir, f'round_statistics_task{task}.png')
        plt.savefig(round_statistics_filename)
        plt.close()  # Close the plot to free up memory
        print(f"Round statistics plot saved to {round_statistics_filename}")
        self.round_statistics.clear()

    def build_similarity_matrix(self, round):
        num_users = len(self.users)
        raw_participation_vectors = []

        for user_id in range(num_users):
            if user_id in self.user_participation_count:
                # Calculate participation metrics for active users
                participation_metrics, _ = self.calculate_participation_metrics(user_id, round)
            else:
                # Assign a zero vector for inactive users
                participation_metrics = [0] * 5  # Assume there are 5 metrics

            raw_participation_vectors.append(participation_metrics)
        #print(raw_participation_vectors)
        normalized_metrics = self.normalize_all_metrics(raw_participation_vectors)
        #print(normalized_metrics)
        similarity_matrix = cosine_similarity(normalized_metrics)
        # Post-process the similarity matrix to set similarities involving inactive users to 0
        for user_id in range(num_users):
            if user_id not in self.user_participation_count:
                similarity_matrix[user_id, :] = 0
                similarity_matrix[:, user_id] = 0

        return similarity_matrix

    def calculate_cosine_metric(self, glob_iter, task):
        cosine_sim_matrix = self.build_similarity_matrix(glob_iter)
        user_indices = {user.id: i for i, user in enumerate(self.users)}

        # Calculate the cosine dissimilarity matrix
        # cosine_dissimilarity_matrix = 1 - cosine_sim_matrix
        # cosine_inv_matrix = 1 - cosine_sim_matrix
        cosine_inv_matrix = 1 - abs(cosine_sim_matrix)
        """
        for user_i in self.users:
            for user_j in self.users:
                i = user_indices[user_i.id]
                j = user_indices[user_j.id]

                cosine_similarity_score = cosine_sim_matrix[i][j]
                cosine_inv_score = cosine_inv_matrix[i][j]
                print(f"i: {i}, j: {j}, user_ids[i]: {user_indices[i]}, user_ids[j]: {user_indices[j]}")
                print(f"Users {user_i.id} and {user_j.id}: Cosine Similarity = {cosine_similarity_score:.2f}")
                print(f"Users {user_i.id} and {user_j.id}: Cosine Inverted Score = {cosine_inv_score:.2f}")
        """
        flatten_cosine = cosine_sim_matrix.flatten()
        #self.all_rounds_cosine_data.append(flatten_cosine)

        mean_cosine_sim_matrix = np.mean(flatten_cosine, axis=0)
        print("mean", mean_cosine_sim_matrix)
        median_cosine_sim_matrix = np.median(flatten_cosine, axis=0)

        # Store the cosine similarity matrix for this round
        print(self.mean_cosine_sim_matrix_history)
        self.mean_cosine_sim_matrix_history.append(mean_cosine_sim_matrix)
        self.median_cosine_sim_matrix_history.append(median_cosine_sim_matrix)
        """
        # Plot the boxplot of the cosine similarity matrix
        plt.figure(figsize=(10, 8))
        plt.boxplot(flatten_cosine, patch_artist=True,
                    boxprops=dict(facecolor='lightgreen', color='darkgreen'))
        plt.title(f'Boxplot of Cosine Similarity Scores, task{task}_round{glob_iter}')
        plt.ylabel('Cosine Similarity Score')
        plt.xlabel('All User Pairs')
        plt.grid(True)

        # Save the boxplot
        save_dir = 'boxplot_cosine_similarity_overall'
        os.makedirs(save_dir, exist_ok=True)
        boxplot_filename = os.path.join(save_dir, f'boxplot_cosine_similarity_overall_task{task}_round{glob_iter}.png')
        plt.savefig(boxplot_filename)
        plt.close()
        print(f"Boxplot saved to {boxplot_filename}")
        """

        return cosine_sim_matrix, cosine_inv_matrix

    def compute_delta_output_matrix(self, glob_iter, task):
        total_forgetting = {}
        user_indices = {user.id: i for i, user in enumerate(self.users)}
        for user in self.users:
            _, _, _, logits, labels = user.test_data_sofar(personal=True)
            _, _, _, _, _, _, _, _, forgetting_measure, _, = user._calculate_accuracy(logits, labels)
            total_forgetting[user.id] = forgetting_measure

        user_ids = list(total_forgetting.keys())
        n_users = len(user_ids)
        delta_output_matrix = np.zeros((n_users, n_users))

        for i in range(n_users):
            for j in range(n_users):
                if i != j:

                    delta_output_matrix[i, j] = abs(total_forgetting[user_ids[i]] - total_forgetting[user_ids[j]])

                    print(f"Users {user_ids[i]} and {user_ids[j]}: Forgetting Difference = {delta_output_matrix[i, j]:.2f} having forgetting measure {total_forgetting[user_ids[i]]} and {total_forgetting[user_ids[j]]}")
                    """
                    accuracy_difference = delta_output_matrix[i, j]
                    # Adjust the printed decimal places based on the magnitude of the difference
                    if accuracy_difference < 0.01:
                        print(
                            f"Users {user_ids[i]} and {user_ids[j]}: Forgetting Difference = {accuracy_difference:.4f} having forgetting measure {total_forgetting[user_ids[i]]:.6f} and {total_forgetting[user_ids[j]]:.6f}")
                    else:
                        print(
                            f"Users {user_ids[i]} and {user_ids[j]}: Forgetting Difference = {accuracy_difference:.2f} having forgetting measure {total_forgetting[user_ids[i]]:.6f} and {total_forgetting[user_ids[j]]:.6f}")
                    """
        # Min-Max normalization
        # Print the normalized matrix

        flatten_delta = delta_output_matrix.flatten()
        #self.all_rounds_delta_data.append(flatten_delta)
        mean_delta_output_matrix = np.mean(flatten_delta, axis=0)
        median_delta_output_matrix = np.median(flatten_delta, axis=0)

        # Store the delta output matrix for this round
        self.mean_delta_output_matrix_history.append(mean_delta_output_matrix)
        self.median_delta_output_matrix_history.append(median_delta_output_matrix)


        return delta_output_matrix

    def elementwise_multiply_matrices(self, glob_iter, task):
        # Initialize task in dictionary if not already present
        if task not in self.task_scores:
            self.task_scores[task] = []

        if task not in self.task_fairness_ratios:
            self.task_fairness_ratios[task] = {}
        # Print the data type of task_fairness_ratios[task]
        print(f"Data type of task_fairness_ratios[{task}]: {type(self.task_fairness_ratios[task])}")

        # Get user IDs
        user_ids = list(self.user_participation_count.keys())
        n_users = len(user_ids)
        user_indices = {user.id: i for i, user in enumerate(self.users)}
        users = len(self.users)
        print("users", users)

        # Compute the delta output matrix
        self.delta_output_matrix = self.compute_delta_output_matrix(glob_iter, task)

        # Compute the cosine similarity and dissimilarity matrices
        self.cosine_sim_matrix, cosine_inverted_matrix = self.calculate_cosine_metric(glob_iter, task)

        # Perform elementwise multiplication with the similarity matrix
        abs_multiplied_sim_matrix = np.multiply(self.delta_output_matrix, self.cosine_sim_matrix)
        self.abs_multiplied_sim_matrix = abs(abs_multiplied_sim_matrix)

        # Define the high similarity threshold
        high_similarity_threshold = 0.8
        # Define the high similarity thresholds
        thresholds = 0.1
        # Initialize empty lists to store high similarity scores
        high_similarity_cosine = []
        high_similarity_delta = []
        high_similarity_multiplied = []
        high_multiplied = []
        high_similarity_data = []
        # Initialize lists to store multiplied scores for the current task
        task_multiplied_scores = []
        # Iterate only over the upper triangle, excluding the diagonal
        for i, user_i in enumerate(self.users):
            for j, user_j in enumerate(self.users[i + 1:], start=i + 1):  # start from the next user after i
                # Retrieve cosine similarity
                similarity = self.cosine_sim_matrix[i, j]
                mul = self.abs_multiplied_sim_matrix[i, j]

                # Calculate the absolute difference in outcomes
                outcome_diff = abs(self.delta_output_matrix[i, j])
                print(
                    f"Users {user_i.id} and {user_j.id} - Cosine Similarity: {similarity:.2f}, Outcome Difference: {outcome_diff:.2f}, Multiplied: {mul:.2f}")

                # Check if similarity is above the threshold
                if similarity > high_similarity_threshold:
                    # Print the values for logging or analysis
                    print(
                        f"Users {user_i.id} and {user_j.id}: Cosine Similarity = {similarity:.2f}, Delta = {outcome_diff:.2f}")

                    # Store only high similarity scores
                    high_similarity_cosine.append(similarity)
                    high_similarity_delta.append(outcome_diff)
                    # Assuming that self.abs_multiplied_sim_matrix is already computed
                    high_multiplied.append(self.abs_multiplied_sim_matrix[i, j])
                    task_multiplied_scores.append(self.abs_multiplied_sim_matrix[i, j])
                    high_similarity_multiplied.append(self.abs_multiplied_sim_matrix[i, j])
                    # self.high_similarity_data.append((user_i.id, user_j.id, similarity, mul))
                    # Optionally print multiplied score as well
                    print(f"Multiplied Score = {high_similarity_multiplied[-1]:.2f}")
                    # Store the user IDs and high multiplication score
                    self.task_scores[task].append(mul)
                    high_similarity_data.append((user_i.id, user_j.id, glob_iter, mul))
                    #print("scores", task_multiplied_scores)
        filtered_task_multiplied_scores = [score for score in task_multiplied_scores if isinstance(score, float)]
        #print("hey", filtered_task_multiplied_scores)
        # Calculate fairness ratio for the current task using the single threshold
        self.fairness_ratio = self.calculate_round_specific_fairness_ratio(filtered_task_multiplied_scores, thresholds, task,
                                                                      glob_iter)
        print(f"Fairness Ratio for Task {task} with threshold {thresholds} in round {glob_iter}: {self.fairness_ratio:.2f}")
        # Update the class attribute for plotting
        self.high_similarity_data.extend(high_similarity_data)
        self.task_scores[task].extend(high_similarity_multiplied)
        # Append the flattened high similarity data for historical record
        self.all_rounds_cosine_data.append(np.array(high_similarity_cosine))
        self.all_rounds_delta_data.append(np.array(high_similarity_delta))
        self.all_rounds_mul_sim_data.append(np.array(high_similarity_multiplied))
        # Assuming self.abs_multiplied_sim_matrix contains NaN for low similarity pairs
        flatten_multiplied_sim = self.abs_multiplied_sim_matrix[~np.isnan(self.abs_multiplied_sim_matrix)].flatten()

        return self.abs_multiplied_sim_matrix

    def calculate_round_specific_fairness_ratio(self, task_multiplied_scores, threshold, task, round_num):
       # print("task_multiplied_scores:", task_multiplied_scores)

        if not task_multiplied_scores:
            print("No valid scores found for threshold", threshold)
            return 0

        task_multiplied_scores = np.array(task_multiplied_scores)  # Convert to NumPy array

        #print("round values", task_multiplied_scores)

        # Define your fairness criteria
        fair_counts = np.sum(task_multiplied_scores < threshold)
        unfair_counts = np.sum(task_multiplied_scores >= threshold)
        print("Threshold:", threshold)
        print("fair counts", fair_counts)
        print("unfair counts", unfair_counts)
        print("Fair values:", task_multiplied_scores[task_multiplied_scores < threshold])
        print("Unfair values:", task_multiplied_scores[task_multiplied_scores >= threshold])
        """
        # Calculate fairness ratio using fair counts / unfair counts
        if unfair_counts == 0:
            fairness_ratio = fair_counts  # Assigning a default value of 10 for unfair_counts being zero
        else:
            fairness_ratio = fair_counts / unfair_counts
        """
        fairness_ratio = fair_counts / (unfair_counts + fair_counts)
        print("fairness ratio", fairness_ratio)
        # Check if the task exists in the dictionary
        if task not in self.task_fairness_ratios:
            self.task_fairness_ratios[task] = {}  # Initialize an empty dictionary if the task doesn't exist

        # Check if the threshold exists for the current task
        if threshold not in self.task_fairness_ratios[task]:
            self.task_fairness_ratios[task][threshold] = []  # Initialize an empty list for the threshold

        # Append the fairness ratio for the current round to the list for the threshold
        self.task_fairness_ratios[task][threshold].append((round_num, fairness_ratio))

        # Print information for debugging
        #print(f"Stored fairness ratio for Task {task} with threshold {threshold}: {fairness_ratio}")
        print(f"After storing, Data of task_fairness_ratios[{task}]: {self.task_fairness_ratios[task]}")
        #print(f"Data type of task_fairness_ratios[{task}]: {type(self.task_fairness_ratios[task])}")

        return fairness_ratio

    def plot_round_fairness(self, task):
        # Define thresholds
        threshold = 0.1
        high_threshold_count = 0

        # Check if task is in task_fairness_ratios dictionary
        if int(task) in self.task_fairness_ratios:  # Convert task to int here
            # Check if task_fairness_ratios[task] is a dictionary
            if isinstance(self.task_fairness_ratios[int(task)], dict):  # Convert task to int here
                # Create separate plots for each threshold
                    if threshold in self.task_fairness_ratios[int(task)]:  # Convert task to int here
                        fairness_data = self.task_fairness_ratios[int(task)][threshold]  # Convert task to int here

                        # Extract rounds and fairness ratios for plotting
                        rounds = [data[0] for data in fairness_data]
                        fairness_ratios = [data[1] for data in fairness_data]

                        # Count the number of fairness ratios greater than 0.8
                        high_threshold_count = sum(value > 0.8 for value in fairness_ratios)
                        total_ratio = high_threshold_count / len(fairness_ratios) if fairness_ratios else 0
                        print(f"length: {len(fairness_ratios) if fairness_ratios else 0}")
                        print(f"Number of fairness ratios greater than 0.8: {high_threshold_count}")
                        print(f"Ratio of fairness ratios greater than 0.8: {total_ratio:.2f}")

                        if len(set(fairness_ratios)) == 1:  # Check if all fairness ratios are the same
                            print(
                                f"All fairness ratios are the same for threshold {threshold} in task {task}. Skipping normalization and plotting.")

                            # Plot original fairness ratios without normalization
                            plt.figure(figsize=(16, 12))
                            plt.plot(rounds, fairness_ratios, marker='o', label='Original')
                            plt.xlabel('Round')
                            plt.ylabel('Fairness Ratio')
                            plt.title(f'Fairness Ratio per Round for Task {task} (Threshold: {threshold}) - Original')
                            plt.xticks(rounds)
                            plt.grid(True)
                            plt.legend()

                            # Save the plot
                            save_dir = 'fairness_ratio_plots'
                            os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist
                            fairness_ratio_filename = os.path.join(save_dir,
                                                                   f'fairness_ratio_task{task}_threshold{threshold}_original.png')
                            plt.savefig(fairness_ratio_filename)
                            plt.close()  # Close the plot to free up memory
                            print(
                                f"Original fairness ratio plot with threshold {threshold} saved to {fairness_ratio_filename}")
                        else:
                            # Normalize fairness ratios using min-max normalization
                            min_ratio = min(fairness_ratios)
                            max_ratio = max(fairness_ratios)
                            min_max_normalized_ratios = [(r - min_ratio) / (max_ratio - min_ratio) for r in
                                                         fairness_ratios]

                            # Normalize fairness ratios using z-score normalization
                            #z_score_normalized_ratios = zscore(fairness_ratios)

                            # Create the plot for original fairness ratios
                            plt.figure(figsize=(16, 12))
                            plt.plot(rounds, fairness_ratios, marker='o', label='Original')
                            plt.xlabel('Round')
                            plt.ylabel('Fairness Ratio')
                            plt.title(f'Fairness Ratio per Round for Task {task} (Threshold: {threshold}) - Original')
                            plt.xticks(rounds)
                            plt.grid(True)
                            plt.legend()

                            # Save the plot
                            save_dir = 'fairness_ratio_plots'
                            os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist
                            fairness_ratio_filename = os.path.join(save_dir,
                                                                   f'fairness_ratio_task{task}_threshold{threshold}_original.png')
                            plt.savefig(fairness_ratio_filename)
                            plt.close()  # Close the plot to free up memory
                            print(
                                f"Original fairness ratio plot with threshold {threshold} saved to {fairness_ratio_filename}")

                            # Create the plot for min-max normalized fairness ratios
                            plt.figure(figsize=(16, 12))
                            plt.plot(rounds, min_max_normalized_ratios, marker='x', label='Min-Max Normalized')
                            plt.xlabel('Round')
                            plt.ylabel('Fairness Ratio')
                            plt.title(
                                f'Fairness Ratio per Round for Task {task} (Threshold: {threshold}) - Min-Max Normalized')
                            plt.xticks(rounds)
                            plt.grid(True)
                            plt.legend()

                            # Save the plot
                            min_max_normalized_filename = os.path.join(save_dir,
                                                                       f'fairness_ratio_task{task}_threshold{threshold}_min_max_normalized.png')
                            plt.savefig(min_max_normalized_filename)
                            plt.close()  # Close the plot to free up memory
                            print(
                                f"Min-Max normalized fairness ratio plot with threshold {threshold} saved to {min_max_normalized_filename}")
                    else:
                        print(f"No fairness ratios found for threshold {threshold} in task {task}")
            else:
                print(f"task_fairness_ratios[{task}] is not a dictionary")
        else:
            print(f"No fairness ratios found for task {task}")


    def calculate_roundwise_statistics(self, task):
        # This method is called after the last round of the task
        scores = np.array(self.task_scores[task])
        if scores.size > 0:
            mean_value = np.mean(scores)
            std_dev_value = np.std(scores)
            variance_value = np.var(scores)
            count = scores.size

            print(f"\nStatistics for Task {task}:")
            print(f"Mean of Multiplied Scores: {mean_value:.2f}")
            print(f"Standard Deviation of Multiplied Scores: {std_dev_value:.2f}")
            print(f"Variance of Multiplied Scores: {variance_value:.2f}")
            print(f"Count of High Similarity Multiplied Scores: {count}")
        else:
            print(f"\nStatistics for Task {task}: No high similarity scores found.")

        self.task_scores[task] = []

    def calculate_task_statistics(self, task):
        # This method is called after the last round of the task
        scores = np.array(self.high_similarity_multiplied)
        if scores.size > 0:
            mean_value = np.mean(scores)
            std_dev_value = np.std(scores)
            variance_value = np.var(scores)
            count = scores.size

            print(f"\nStatistics for Task {task}:")
            print(f"Mean of Multiplied Scores: {mean_value:.2f}")
            print(f"Standard Deviation of Multiplied Scores: {std_dev_value:.2f}")
            print(f"Variance of Multiplied Scores: {variance_value:.2f}")
            print(f"Count of High Similarity Multiplied Scores: {count}")
        else:
            print(f"\nStatistics for Task {task}: No high similarity scores found.")

        self.high_similarity_multiplied.clear()



    def plot_and_save_std_matrices(self, glob_iter, task):

        plt.figure(figsize=(12, 8))

        plt.plot(self.multiplied_dif_std_matrix_history, marker='o', linestyle='-',
                 label='Standard Deviation of Difference')

        # Set labels and title
        plt.xlabel('Rounds')
        plt.ylabel('Standard Deviation ')
        plt.title('Standard Deviation Over Time')

        # Add legend
        plt.legend()
        plt.show()
        # Save the plot
        save_dir = f'std_matrices_plots'
        os.makedirs(save_dir, exist_ok=True)
        plot_filename = os.path.join(save_dir, f'std_matrices_task{task}_round{glob_iter}.png')
        plt.savefig(plot_filename)
        plt.close()
        print(f"STD Matrices saved to {plot_filename}")

    def plot_and_save_mean_matrices(self, glob_iter, task):
        print(f"Length of mean_cosine_sim_matrix_history: {len(self.mean_cosine_sim_matrix_history)}")
        print(f"Length of mean_delta_output_matrix_history: {len(self.mean_delta_output_matrix_history)}")
        print(f"Length of mean_multiplied_sim_matrix_history: {len(self.mean_multiplied_sim_matrix_history)}")

        # mean_cosine_sim_matrix = np.mean(self.mean_cosine_sim_matrix_history, axis=0)
        mean_delta_output_matrix = np.mean(self.delta_output_matrix, axis=0)
        # mean_multiplied_sim_matrix = np.mean(self.mean_multiplied_sim_matrix_history, axis=0)
        plt.figure(figsize=(12, 8))
        # flattened_array_cosine = [item for sublist in self.mean_cosine_sim_matrix_history for item in sublist]
        # Plot mean cosine similarity matrix as a line plot
        print("print", self.mean_cosine_sim_matrix_history)
        print("print", self.mean_delta_output_matrix_history)
        plt.plot(self.mean_cosine_sim_matrix_history, marker='o', linestyle='-', label='Mean Cosine Similarity')

        # flattened_array_delta = [item for sublist in self.mean_delta_output_matrix_history for item in sublist]
        # Plot mean delta output matrix as a line plot
        plt.plot(self.mean_delta_output_matrix_history, marker='^', linestyle='-', label='Mean Delta Output')
        # flattened_array_multiplied = [item for sublist in self.mean_multiplied_sim_matrix_history for item in sublist]

        plt.plot(self.mean_multiplied_sim_matrix_history, marker='o', linestyle='-', label='Mean Multiplied Output')

        # Set labels and title
        plt.xlabel('Rounds')
        plt.ylabel('Mean ')
        plt.title('Mean Matrices Over Time')

        # Add legend
        plt.legend()
        plt.show()
        # Save the plot
        save_dir = f'mean_matrices_plots'
        os.makedirs(save_dir, exist_ok=True)
        plot_filename = os.path.join(save_dir, f'mean_matrices_task{task}_round{glob_iter}.png')
        plt.savefig(plot_filename)
        plt.close()
        print(f"Mean Matrices saved to {plot_filename}")

    def plot_and_save_median_matrices(self, glob_iter, task):
        # median_cosine_sim_matrix = np.median(self.median_cosine_sim_matrix_history, axis=0)
        # median_delta_output_matrix = np.median(self.median_delta_output_matrix_history, axis=0)
        # median_multiplied_sim_matrix = np.median(self.median_multiplied_sim_matrix_history, axis=0)
        plt.figure(figsize=(12, 8))
        # flattened_array_cosine_median = [item for sublist in self.median_cosine_sim_matrix_history for item in sublist]
        # Plot mean cosine similarity matrix as a line plot
        plt.plot(self.median_cosine_sim_matrix_history, marker='o', linestyle='-', label='Median Cosine Similarity')
        # flattened_array_delta_median = [item for sublist in self.median_delta_output_matrix_history for item in sublist]
        # Plot mean delta output matrix as a line plot
        plt.plot(self.median_delta_output_matrix_history, marker='o', linestyle='-', label='Median Delta Output')
        # flattened_array_delta_median = [item for sublist in self.median_delta_output_matrix_history for item in sublist]
        plt.plot(self.median_multiplied_sim_matrix_history, marker='o', linestyle='-', label='Median Multiplied Output')

        # Set labels and title

        plt.xlabel('Rounds')
        plt.ylabel('Median ')
        plt.title('Median Matrices Over Time')

        # Add legend
        plt.legend()

        # Save the plot
        save_dir = f'median_matrices_plots'
        os.makedirs(save_dir, exist_ok=True)
        plot_filename = os.path.join(save_dir, f'median_matrices_plots{task}_round{glob_iter}.png')
        plt.savefig(plot_filename)
        plt.close()
        print(f"Median Matrices saved to {plot_filename}")

    def collect_forgetting_measures(self):
        forgetting_measures = {}
        for user in self.users:
            _, _, _, logits, labels = user.test_data_sofar(personal=True)
            _, _, _, _, _, _, _, _, forgetting_measure, _, = user._calculate_accuracy(logits, labels)
            forgetting_measures[user.id] = forgetting_measure
        return forgetting_measures



    def calculate_taskwise_distances(self, forgetting_measures):
        task_distances = {}
        user_ids = list(forgetting_measures.keys())
        for i in range(len(user_ids)):
            for j in range(i + 1, len(user_ids)):
                user_i = user_ids[i]
                user_j = user_ids[j]
                distance = np.abs(forgetting_measures[user_i] - forgetting_measures[user_j])
                task_distances[f"{user_i}-{user_j}"] = distance
        return task_distances

    def calculate_users_below_thresholds(self, forgetting_measures):
        adjusted_measures = {user: measure / 100 for user, measure in forgetting_measures.items()}
        thresholds = [i / 10 for i in range(1, 11)]  # This creates a list [0.1, 0.2, ..., 1.0]
        results = {}
        total_users = len(adjusted_measures)
        print("Total number of users:", total_users)
        print("Forgetting measures for each user:", forgetting_measures)
        print("Adjusted forgetting measures for each user:", adjusted_measures)

        last_percentage = -1
        for threshold in thresholds:
            users_below_threshold = [user for user, measure in adjusted_measures.items() if measure < threshold]
            percentage_below = (len(users_below_threshold) / total_users) * 100 if total_users > 0 else 0

            if percentage_below != last_percentage:
                print(f"\nThreshold: {threshold}")
                print(f"Users below threshold {threshold}: {users_below_threshold}")
                print(f"Percentage of users below threshold {threshold}: {percentage_below:.2f}%")
                last_percentage = percentage_below

            results[threshold] = percentage_below
        return results

    def calculate_users_above_thresholds(self, forgetting_measures):
        adjusted_measures = {user: measure / 100 for user, measure in forgetting_measures.items()}
        thresholds = [i / 10 for i in range(1, 11)]  # This creates a list [0.1, 0.2, ..., 1.0]
        results = {}
        total_users = len(adjusted_measures)
        print("Total number of users:", total_users)
        print("Forgetting measures for each user:", forgetting_measures)
        print("Adjusted forgetting measures for each user:", adjusted_measures)
        last_percentage = -1
        for threshold in thresholds:
            users_above_threshold = sum(measure > threshold for measure in adjusted_measures.values())
            percentage_above = (users_above_threshold / total_users) * 100 if total_users > 0 else 0
            if percentage_above != last_percentage:
                # Print statements for debugging
                print(f"Threshold: {threshold}")
                print(f"Number of users above threshold {threshold}: {users_above_threshold}")
                print(f"Percentage of users above threshold {threshold}: {percentage_above:.2f}%")
                last_percentage = percentage_above
            results[threshold] = percentage_above
        return results

    def check_forgetting_threshold(self, forgetting_measures, threshold):
        users_above_threshold = {}
        for user_id, measure in forgetting_measures.items():
            if measure > threshold:
                users_above_threshold[user_id] = measure
        return users_above_threshold

    def test(self, selected=False):
        '''
        tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        losses = []
        users = self.selected_users if selected else self.users
        for c in users:
            ct, c_loss, ns = c.test()
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
            losses.append(c_loss)
        ids = [c.id for c in self.users]

        return ids, num_samples, tot_correct, losses

    def test_all(self, selected=False):
        '''
        tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        losses = []
        users = self.selected_users if selected else self.users
        for c in users:
            ct, c_loss, ns = c.test_all()
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
            losses.append(c_loss)
        ids = [c.id for c in self.users]

        return ids, num_samples, tot_correct, losses

    def test_per_task(self, selected=False):
        '''
        tests latest model on leanrt tasks
        '''
        accs = {}

        users = self.selected_users if selected else self.users
        for c in users:
            accs[c.id] = []

            ct, c_loss, ns = c.test_per_task()

            # per past task:
            for task in range(len(ct)):
                acc = ct[task] / ns[task]
                accs[c.id].append(acc)

        return accs

    def test_(self, selected=False, personal=False):
        '''
        tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        losses = []
        users = self.selected_users if selected else self.users
        print("Results from test_ function:")
        for c in users:
            ct, c_loss, ns, *_ = c.test_(personal=personal)
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
            losses.append(c_loss)
            print(f"User ID: {c.id}")
            print(f"Number of Samples: {ns}")
            print(f"Total Correct: {ct}")
            print(f"Loss: {c_loss}\n")
        ids = [c.id for c in users]
        return ids, num_samples, tot_correct, losses

    def test_data_sofar(self, selected=False, personal=False):
        '''
        tests self.latest_model on the accumulated test data so far for given clients
        '''
        num_samples = []
        tot_correct = []
        losses = []
        all_logits = []
        all_labels = []

        users = self.selected_users if selected else self.users
        for c in users:
            ct, c_loss, ns, c_logits, c_labels = c.test_data_sofar(personal=personal)
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
            losses.append(c_loss)
            all_logits.append(c_logits)
            all_labels.append(c_labels)

        ids = [c.id for c in users]

        # Concatenate all logits and labels from all users
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        return ids, num_samples, tot_correct, losses, all_logits, all_labels

    def test_all_(self, selected=False, personal=False, matrix=False):
        '''
        tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        losses = []
        preds = []
        labels = []

        users = self.selected_users if selected else self.users
        print("Results from test_all_ function:")
        for c in users:
            if matrix == False:
                ct, c_loss, ns = c.test_all_(personal=personal)
            else:
                ct, c_loss, ns, pred, label = c.test_all_(personal=personal, matrix=True)
                preds += pred
                labels += label
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
            losses.append(c_loss)
            print(f"User ID: {c.id}")
            print(f"Number of Samples: {ns}")
            print(f"Total Correct: {ct}")
            print(f"Loss: {c_loss}")

        ids = [c.id for c in users]

        if matrix == False:
            return ids, num_samples, tot_correct, losses
        else:
            return ids, num_samples, tot_correct, losses, preds, labels

    def test_per_task_(self, selected=False, personal=True):
        '''
        Tests the latest model on learned tasks and collects forgetting measures.
        '''
        accs = {}
        forgetting_measures = {}  # Store forgetting measures per user

        # Select users based on the 'selected' flag
        users = self.selected_users if selected else self.users

        for c in users:
            accs[c.id] = []
            forgetting_measures[c.id] = []  # Initialize list to store forgetting measures for this user

            # Updated to expect two return values: test_acc and forgetting_measures
            ct, forgetting_measures_ct = c.test_per_task_(personal=personal)

            # Iterate over each task
            for task in range(len(ct)):
                acc = ct[task]
                accs[c.id].append(acc)

                # Collect the forgetting measure for the current task
                forgetting_measure = forgetting_measures_ct[task]
                forgetting_measures[c.id].append(forgetting_measure)

                # Optional: Print task-specific accuracy and forgetting measure
                # print(f"User ID: {c.id}, Task {task + 1}: Accuracy: {acc:.2f}%, Forgetting Measure: {forgetting_measure:.2f}")

        # Optional: Return forgetting measures alongside accuracies
        return accs, forgetting_measures

    def test_personalized_model(self, selected=True):
        '''
        tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        losses = []
        users = self.selected_users if selected else self.users
        for c in users:
            ct, ns, loss = c.test_personalized_model()
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
            losses.append(loss)
        ids = [c.id for c in self.users]

        return ids, num_samples, tot_correct, losses

    def evaluate_personalized_model(self, selected=True, save=True):
        stats = self.test_personalized_model(selected=selected)
        test_ids, test_num_samples, test_tot_correct, test_losses = stats[:4]
        glob_acc = np.sum(test_tot_correct) * 1.0 / np.sum(test_num_samples)
        test_loss = np.sum([x * y for (x, y) in zip(test_num_samples, test_losses)]).item() / np.sum(test_num_samples)
        if save:
            self.metrics['per_acc'].append(glob_acc)
            self.metrics['per_loss'].append(test_loss)
        print("Average Global Accuracy = {:.4f}, Loss = {:.2f}.".format(glob_acc, test_loss))

    def evaluate_ensemble(self, selected=True):
        self.model.eval()
        users = self.selected_users if selected else self.users
        test_acc = 0
        loss = 0
        for x, y in self.testloaderfull:
            target_logit_output = 0
            for user in users:
                # get user logit
                user.model.eval()
                user_result = user.model(x, logit=True)
                target_logit_output += user_result['logit']
            target_logp = F.log_softmax(target_logit_output, dim=1)
            test_acc += torch.sum(torch.argmax(target_logp, dim=1) == y)  # (torch.sum().item()
            loss += self.loss(target_logp, y)
        loss = loss.detach().numpy()
        test_acc = test_acc.detach().numpy() / y.shape[0]
        self.metrics['glob_acc'].append(test_acc)
        self.metrics['glob_loss'].append(loss)
        print("Average Accuracy = {:.4f}, Loss = {:.2f}.".format(test_acc, loss))

    def evaluate(self, save=True, selected=False):
        # override evaluate function to log vae-loss.
        test_ids, test_samples, test_accs, test_losses = self.test(selected=selected)

        glob_acc = np.sum(test_accs) * 1.0 / np.sum(test_samples)

        glob_loss = np.sum(
            [x * y.detach().to(torch.device('cpu')) for (x, y) in zip(test_samples, test_losses)]).item() / np.sum(
            test_samples)
        if save:
            self.metrics['glob_acc'].append(glob_acc)
            self.metrics['glob_loss'].append(glob_loss)
        print("Average Global Accuracy = {:.4f}, Loss = {:.2f}.".format(glob_acc, glob_loss))

    def evaluate_all(self, save=True, selected=False):
        # override evaluate function to log vae-loss.
        test_ids, test_samples, test_accs, test_losses = self.test_all(selected=selected)

        glob_acc = np.sum(test_accs) * 1.0 / np.sum(test_samples)
        glob_loss = np.sum(
            [x * y.detach().to(torch.device('cpu')) for (x, y) in zip(test_samples, test_losses)]).item() / np.sum(
            test_samples)

        if save:
            self.metrics['glob_acc'].append(glob_acc)
            self.metrics['glob_loss'].append(glob_loss)
        print("Average Global Accuracy (classes so far) = {:.4f}, Loss = {:.2f}.".format(glob_acc, glob_loss))

    def evaluate_per_client_per_task(self, save=True, selected=False):
        accs = self.test_per_task()

        for k, v in accs.items():
            print(k)
            print(v)

    def evaluate_(self, save=True, selected=False, personal=False):
        # override evaluate function to log vae-loss.
        test_ids, test_samples, test_accs, test_losses = self.test_(selected=selected, personal=personal)

        glob_acc = np.sum(test_accs) * 1.0 / np.sum(test_samples)
        # Since test_losses are already floats, we can directly use them for computation
        glob_loss = np.sum([x * y for (x, y) in zip(test_samples, test_losses)]) / np.sum(test_samples)
        """
        if save:
            self.metrics['glob_acc'].append(glob_acc)
            self.metrics['glob_loss'].append(glob_loss)
        """
        # Assuming std_dev calculation needs adjustment as well. Ensure test_accs and test_samples are numpy arrays or lists of numbers.
        std_dev = np.std(np.array(test_accs) / np.array(test_samples))
        print("Standard Deviation of Accuracies: {:.4f}".format(std_dev))

        print("Average Global Accuracy = {:.4f}, Loss = {:.2f}.".format(glob_acc, glob_loss))

    def write(self, accuracy, file=None, mode='a'):
        with open(file, mode) as f:
            line = str(accuracy) + '\n'
            f.writelines(line)

    def evaluate_all_(self, save=True, selected=False, personal=False, matrix=False):
        '''
        test_all_() returns lists of a certain info. of all Clients. [data of client_1, d_o_c_2, ...]
        '''

        if matrix == False:
            test_ids, test_samples, test_accs, test_losses = self.test_all_(selected=selected, personal=personal)
        else:
            test_ids, test_samples, test_accs, test_losses, preds, labels = self.test_all_(selected=selected,
                                                                                           personal=personal,
                                                                                           matrix=True)
            # save pdf
            save_matrix(preds, labels)

        glob_acc = np.sum(test_accs) * 1.0 / np.sum(test_samples)
        glob_loss = np.sum(
            [x * y.detach().to(torch.device('cpu')) for (x, y) in zip(test_samples, test_losses)]).item() / np.sum(
            test_samples)
        """
        if save:
            self.metrics['glob_acc'].append(glob_acc)
            self.metrics['glob_loss'].append(glob_loss)
        """
        # Modified: Calculate and print the standard deviation of accuracies
        std_dev = np.std(test_accs / np.array(test_samples))
        print("Standard Deviation of Accuracies: {:.4f}".format(std_dev))

        if personal == False:
            print("Average Accuracy (classes so far) = {:.4f}, Loss = {:.2f}.".format(glob_acc, glob_loss))
        else:
            print("Average Personal Accuracy (classes so far) = {:.4f}, Loss = {:.2f}.".format(glob_acc, glob_loss))

    def evaluate_per_client_per_task_(self, save=True, selected=False, personal=True):
        accs, forgetting_measures = self.test_per_task_(selected=selected, personal=personal)

        std_devs_per_client = {}
        avg_accs_per_client = {}
        avg_forgetting_measures_per_client = {}

        for client_id, accuracies in accs.items():
            std_devs_per_client[client_id] = np.std(accuracies)
            avg_accs_per_client[client_id] = np.mean(accuracies)
            client_forgetting_measures = forgetting_measures.get(client_id, [])
            avg_forgetting_measures_per_client[client_id] = np.mean(
                client_forgetting_measures) if client_forgetting_measures else 0

            print(f'Client-{client_id}: ', end="")

            for task_idx, task_accuracy in enumerate(accuracies):
                print(f"Task {task_idx + 1} Accuracy: {task_accuracy:.4f} ", end="")

            print(f"(Overall Std Dev: {std_devs_per_client[client_id]:.4f}, "
                  f"Average Accuracy: {avg_accs_per_client[client_id]:.4f}, "
                  f"Average Forgetting Measure: {avg_forgetting_measures_per_client[client_id]:.4f})", end="\n")

        # Calculate and print the average forgetting measure across all users
        all_forgetting_measures = [measure for measures in forgetting_measures.values() for measure in measures]
        avg_forgetting_measure_across_users = np.mean(all_forgetting_measures) if all_forgetting_measures else 0
        print(f"Average Forgetting Measure Across All Users: {avg_forgetting_measure_across_users:.4f}")

    def calculate_task_accuracies(self, selected=False, personal=True):
        accs, _ = self.test_per_task_(selected=selected, personal=personal)
        task_accuracies = {}
        for client_accs in accs.values():
            for task_idx, task_accuracy in enumerate(client_accs):
                task_accuracies.setdefault(task_idx, []).append(task_accuracy)

        # Calculate, print average accuracy per task, and calculate standard deviation
        average_task_accuracies = {}
        standard_deviation_task_accuracies = {}  # Dictionary to store the standard deviation
        for task, accs in task_accuracies.items():
            average_acc = np.mean(accs)  # Use np.mean for consistency
            standard_deviation = np.std(accs, ddof=1)  # Calculate standard deviation for task accuracies using NumPy
            print(f"Task {task + 1} Accuracies: {accs}")
            print(f"Task {task + 1} Average Accuracy: {average_acc:.2f}%")
            print(f"Task {task + 1} Standard Deviation: {standard_deviation:.2f}")  # Print standard deviation
            average_task_accuracies[task] = average_acc
            standard_deviation_task_accuracies[task] = standard_deviation  # Store standard deviation

        return average_task_accuracies, standard_deviation_task_accuracies


def save_matrix(preds, labels):
    p = []
    for item in preds:
        p.append(item.cpu().numpy())

    l = []
    for item in labels:
        l.append(item.cpu().numpy())

    s = set()
    for item in l:
        s.add(int(item))

    s = list(s)

    sns.set()
    f, ax = plt.subplots()
    df = confusion_matrix(l, p, labels=s)

    min_ = 0
    max_ = 0

    for row in df:
        for v in row:
            if v >= max_:
                max_ = v
            if v <= min_:
                min_ = v

    df_n = (df - min_) / (max_ - min_)

    sns.heatmap(df_n, annot=False, ax=ax, yticklabels=True, xticklabels=True, )  # 画热力图
    name = 'None'
    plt.savefig('matrix/' + name)