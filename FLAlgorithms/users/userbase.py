import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import copy
from utils.model_utils import get_dataset_name
from utils.model_config import RUNCONFIGS
from FLAlgorithms.optimizers.fedoptimizer import pFedIBOptimizer
from torch import optim
from collections import defaultdict


class User:
    """
    Base class for users in federated learning.
    """

    def __init__(
            self, args, id, model, train_data, test_data, use_adam=False, my_model_name=None, unique_labels=None):

        self.model = copy.deepcopy(model[0])
        self.model_name = model[1]
        self.id = id  # integer
        self.train_data = train_data
        self.test_data = test_data
        self.train_samples = len(self.train_data)
        self.test_samples = len(self.test_data)
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.beta = args.beta
        self.lamda = args.lamda
        self.local_epochs = args.local_epochs
        self.algorithm = args.algorithm
        self.K = args.K
        self.dataset = args.dataset
        self.trainloader = DataLoader(self.train_data, self.batch_size, drop_last=True, shuffle=True)
        self.testloader = DataLoader(self.test_data, self.batch_size, drop_last=True)

        self.testloaderfull = DataLoader(self.test_data, self.test_samples)
        self.trainloaderfull = DataLoader(self.train_data, self.train_samples, shuffle=True)
        self.iter_trainloader = iter(self.trainloader)
        self.iter_testloader = iter(self.testloader)

        self.test_data_so_far = []
        self.test_data_so_far += self.test_data
        self.test_data_so_far_loader = DataLoader(self.test_data_so_far, len(self.test_data_so_far))

        self.test_data_per_task = []
        self.test_data_per_task.append(self.test_data)

        dataset_name = get_dataset_name(self.dataset)
        self.unique_labels = unique_labels
        self.generative_alpha = RUNCONFIGS[dataset_name]['generative_alpha']
        self.generative_beta = RUNCONFIGS[dataset_name]['generative_beta']

        # those parameters are for personalized federated learning.
        self.local_model = copy.deepcopy(list(self.model.parameters()))
        self.personalized_model_bar = copy.deepcopy(list(self.model.parameters()))
        self.prior_decoder = None
        self.prior_params = None

        # continual federated learning
        self.classes_so_far = []  # all labels of a client so far
        self.available_labels_current = []  # labels from all clients on T (current)
        self.current_labels = []  # current labels for itself
        self.classes_past_task = []  # classes_so_far (current labels excluded)
        self.available_labels_past = []  # labels from all clients on T-1
        self.current_task = 0
        self.previous_task = 0
        self.init_loss_fn()
        self.label_counts = {}
        self.available_labels = []  # l from all c from 0-T
        self.label_set = [i for i in range(10)]
        self.my_model_name = my_model_name
        self.last_copy = None
        self.if_last_copy = False
        self.args = args
        self.class_to_last_task = {}
        self.initial_class_accuracies = {}
        self.task_accuracies_after_learning = {}
        # self.old_classes = []
        # self.new_classes = []

    def next_task(self, train, test, label_info=None, if_label=True):

        if "CIFAR10" in self.args.dataset:
            optimizerD = optim.Adam(self.generator.critic.parameters(), lr=0.0002, betas=(0.5, 0.999))
            optimizerG = optim.Adam(self.generator.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

            self.generator.set_generator_optimizer(optimizerG)
            self.generator.set_critic_optimizer(optimizerD)
            print('optimizers updated!')

        # update last model:
        self.last_copy = copy.deepcopy(self.generator).to(
            torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.if_last_copy = True

        # update generator:
        if self.my_model_name == 'fedcl':
            self.last_generator = copy.deepcopy(self.generator)

        # update dataset:
        self.train_data = train
        self.test_data = test

        self.train_samples = len(self.train_data)
        self.test_samples = len(self.test_data)

        self.trainloader = DataLoader(self.train_data, self.batch_size, drop_last=True, shuffle=True)
        self.testloader = DataLoader(self.test_data, self.batch_size, drop_last=True)

        self.testloaderfull = DataLoader(self.test_data, len(self.test_data))
        self.trainloaderfull = DataLoader(self.train_data, len(self.train_data), shuffle=True)
        self.iter_trainloader = iter(self.trainloader)
        self.iter_testloader = iter(self.testloader)

        # update classes_past_task
        self.classes_past_task = copy.deepcopy(self.classes_so_far)

        # update classes_so_far
        if if_label:
            self.classes_so_far.extend(label_info['labels'])

            self.current_labels.clear()
            self.current_labels.extend(label_info['labels'])

        # update test data for CL: (classes so far)
        self.test_data_so_far += self.test_data
        self.test_data_so_far_loader = DataLoader(self.test_data_so_far, len(self.test_data_so_far))

        # update test data for CL: (test per task)
        self.test_data_per_task.append(self.test_data)

        for class_id in self.current_labels:
            # if class_id not in self.class_to_last_task:
            # Only update if the class is new and was not introduced in previous tasks
            self.class_to_last_task[class_id] = self.current_task
        self.previous_task = self.current_task
        # update class recorder:
        self.current_task += 1

        return

    def init_loss_fn(self):
        self.loss = nn.NLLLoss()
        self.dist_loss = nn.MSELoss()
        self.ensemble_loss = nn.KLDivLoss(reduction="batchmean")
        self.ce_loss = nn.CrossEntropyLoss()

    # kd_loss
    def kd_loss(self, teacher, gen_output, generative_alpha, selected, T=2):

        # user output logp
        student_output = self.model(gen_output, start_layer_idx=self.latent_layer_idx, logit=True)['logit']
        student_output = student_output[:, selected]
        student_output_logp = F.log_softmax(student_output / T, dim=1)

        # global output p
        teacher_output = teacher(gen_output, start_layer_idx=self.latent_layer_idx, logit=True)['logit']
        teacher_output = teacher_output[:, selected]
        teacher_outputp = F.softmax(teacher_output / T, dim=1).clone().detach()

        # kd loss
        kd_loss = self.ensemble_loss(student_output_logp, teacher_outputp)
        kd_loss = generative_alpha * kd_loss  # debug

        return kd_loss

    def set_parameters(self, model, beta=1):
        '''
        self.model: old user model
        model: the global model on the server (new model)
        '''
        for old_param, new_param, local_param in zip(self.model.parameters(), model.parameters(), self.local_model):
            if beta == 1:
                old_param.data = new_param.data.clone()
                local_param.data = new_param.data.clone()
            else:
                old_param.data = beta * new_param.data.clone() + (1 - beta) * old_param.data.clone()
                local_param.data = beta * new_param.data.clone() + (1 - beta) * local_param.data.clone()

    def set_parameters_(self, model, only_critic, beta=1, mode=None, gr=False, classifier=None):
        '''
        At the beginning of this round:
        self.model: old user model, note trained yet
        model: the global model on the server (new model)
        '''
        if gr == True:
            for old_param, new_param in zip(self.classifier.critic.parameters(), classifier.critic.parameters()):
                if beta == 1:
                    old_param.data = new_param.data.clone()
                else:
                    old_param.data = beta * new_param.data.clone() + (1 - beta) * old_param.data.clone()

        else:
            if only_critic == True:
                for old_param, new_param in zip(self.generator.critic.parameters(), model.critic.parameters()):
                    if beta == 1:
                        old_param.data = new_param.data.clone()
                    else:
                        old_param.data = beta * new_param.data.clone() + (1 - beta) * old_param.data.clone()

            else:
                for old_param, new_param in zip(self.generator.parameters(), model.parameters()):
                    if beta == 1:
                        old_param.data = new_param.data.clone()
                    else:
                        old_param.data = beta * new_param.data.clone() + (1 - beta) * old_param.data.clone()

    def set_prior_decoder(self, model, beta=1):
        for new_param, local_param in zip(model.personal_layers, self.prior_decoder):
            if beta == 1:
                local_param.data = new_param.data.clone()
            else:
                local_param.data = beta * new_param.data.clone() + (1 - beta) * local_param.data.clone()

    def set_prior(self, model):
        for new_param, local_param in zip(model.get_encoder() + model.get_decoder(), self.prior_params):
            local_param.data = new_param.data.clone()

    # only for pFedMAS
    def set_mask(self, mask_model):
        for new_param, local_param in zip(mask_model.get_masks(), self.mask_model.get_masks()):
            local_param.data = new_param.data.clone()

    def set_shared_parameters(self, model, mode='decode'):
        # only copy shared parameters to local
        for old_param, new_param in zip(
                self.model.get_parameters_by_keyword(mode),
                model.get_parameters_by_keyword(mode)
        ):
            old_param.data = new_param.data.clone()

    def get_parameters(self):
        for param in self.model.parameters():
            param.detach()
        return self.model.parameters()

    def clone_model_paramenter(self, param, clone_param):
        with torch.no_grad():
            for param, clone_param in zip(param, clone_param):
                clone_param.data = param.data.clone()
        return clone_param

    def get_updated_parameters(self):
        return self.local_weight_updated

    def update_parameters(self, new_params, keyword='all'):
        for param, new_param in zip(self.model.parameters(), new_params):
            param.data = new_param.data.clone()

    def get_grads(self):
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad.data)
        return grads

    def _store_initial_accuracies(self, logits, labels):
        _, predicted = torch.max(logits.data, 1)
        unique_classes = labels.unique()
        for class_id_tensor in unique_classes:
            class_id = class_id_tensor.item()  # Convert to Python int
            if class_id not in self.initial_class_accuracies:
                class_mask = labels == class_id_tensor  # Use the tensor version for indexing
                correct = (predicted[class_mask] == class_id_tensor).sum().item()
                total = class_mask.sum().item()
                self.initial_class_accuracies[class_id] = 100 * correct / total if total > 0 else 0
                print("acc", self.initial_class_accuracies)

    def _calculate_accuracy(self, logits, labels, mask_logits=True):
        _, predicted = torch.max(logits.data, 1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()

        # Separate old and current classes
        old_classes = [c for c in self.classes_past_task if c not in self.current_labels]
        current_classes = self.current_labels

        # Calculate class accuracies for average method
        old_class_accuracies_avg = []
        current_class_accuracies_avg = []
        for class_id in old_classes:
            # task_introduced = self._infer_task_for_class(class_id)
            class_accuracy = self._calculate_class_accuracy(predicted, labels, class_id, mask_logits)
            old_class_accuracies_avg.append(class_accuracy)
            # old_class_accuracies_avg[class_id] = (class_accuracy, task_introduced)
        for class_id in current_classes:
            class_accuracy = self._calculate_class_accuracy(predicted, labels, class_id, mask_logits)
            current_class_accuracies_avg.append(class_accuracy)

        # Compute average accuracies
        avg_old_accuracy = sum(old_class_accuracies_avg) / len(
            old_class_accuracies_avg) if old_class_accuracies_avg else 0
        avg_current_accuracy = sum(current_class_accuracies_avg) / len(
            current_class_accuracies_avg) if current_class_accuracies_avg else 0

        # Create masks based on the updated class lists
        old_classes_mask = torch.zeros_like(labels, dtype=torch.bool)
        for old_class in old_classes:
            old_classes_mask |= (labels == old_class)

        current_classes_mask = torch.zeros_like(labels, dtype=torch.bool)
        for current_class in current_classes:
            current_classes_mask |= (labels == current_class)

        old_total = old_classes_mask.sum().item()
        current_total = current_classes_mask.sum().item()

        old_correct = (predicted[old_classes_mask] == labels[old_classes_mask]).sum().item()
        current_correct = (predicted[current_classes_mask] == labels[current_classes_mask]).sum().item()

        orig_old_accuracy = old_correct / old_total if old_total != 0 else 0
        orig_current_accuracy = current_correct / current_total if current_total != 0 else 0

        total_accuracy = correct / total

        # Calculate and return accuracy for each old and current class
        # old_class_accuracies = {old_class: self._calculate_class_accuracy(predicted, labels, old_class, mask_logits) for old_class in old_classes}
        # current_class_accuracies = {current_class: self._calculate_class_accuracy(predicted, labels, current_class, mask_logits) for current_class in current_classes}

        # Calculate and return accuracy for each old and current class with task information
        old_class_accuracies = {old_class: (self._calculate_class_accuracy(predicted, labels, old_class, mask_logits),
                                            self._infer_task_for_class(old_class)) for old_class in old_classes}
        current_class_accuracies = {
            current_class: self._calculate_class_accuracy(predicted, labels, current_class, mask_logits) for
            current_class in current_classes}

        # Calculate retention rate
        retention_rates = []
        for class_id, initial_acc in self.initial_class_accuracies.items():
            if class_id in old_classes:
                current_acc = self._calculate_class_accuracy(predicted, labels, class_id, mask_logits)
                retention_rate = current_acc / initial_acc if initial_acc != 0 else 0
                retention_rates.append(retention_rate)

        avg_retention_rate = sum(retention_rates) / len(retention_rates) if retention_rates else 0

        # forgetting measure
        total_forgetting = 0
        for class_id, initial_acc in self.initial_class_accuracies.items():
            current_acc = self._calculate_class_accuracy(predicted, labels, class_id, mask_logits)
            # total_forgetting += max(0, initial_acc - current_acc)
            # Calculate the decrease in accuracy as a percentage of the initial accuracy
            forgetting_percentage = max(0, initial_acc - current_acc) / initial_acc if initial_acc != 0 else 0
            total_forgetting += forgetting_percentage

        forgetting_measure = total_forgetting / len(
            self.initial_class_accuracies) if self.initial_class_accuracies else 0

        # bwt
        bwt = 0
        for class_id in self.classes_past_task:
            initial_acc = self.initial_class_accuracies.get(class_id, 0)
            current_acc = self._calculate_class_accuracy(predicted, labels, class_id, mask_logits)
            bwt += (current_acc - initial_acc)

        bwt /= len(self.classes_past_task) if self.classes_past_task else 1


        return total_accuracy, orig_old_accuracy, orig_current_accuracy, avg_old_accuracy, avg_current_accuracy, old_class_accuracies, current_class_accuracies, avg_retention_rate, forgetting_measure, bwt

    def _calculate_class_accuracy(self, predicted, labels, class_id, mask_logits=True):
        mask = labels == class_id
        correct = (predicted[mask] == labels[mask]).sum().item()
        total = mask.sum().item()

        if mask_logits:
            # Skip logits masking for old classes
            return 100 * correct / total if total != 0 else 0
        else:
            return correct, total

    def _calculate_forgetting_measure(self, predicted, labels):
        total_forgetting = 0
        for class_id, initial_acc in self.initial_class_accuracies.items():
            current_acc = self._calculate_class_accuracy(predicted, labels, class_id, mask_logits)
            total_forgetting += max(0, initial_acc - current_acc)

        forgetting_measure = total_forgetting / len(
            self.initial_class_accuracies) if self.initial_class_accuracies else 0
        return forgetting_measure

    def _calculate_backward_transfer(self, predicted, labels):
        bwt = 0
        for class_id in self.classes_past_task:
            initial_acc = self.initial_class_accuracies.get(class_id, 0)
            current_acc = self._calculate_class_accuracy(predicted, labels, class_id, mask_logits)
            bwt += (current_acc - initial_acc)

        bwt /= len(self.classes_past_task) if self.classes_past_task else 1
        return bwt

    def _calculate_catastrophic_forgetting_index(self, predicted, labels, threshold):
        num_classes_forgotten = 0
        for class_id, initial_acc in self.initial_class_accuracies.items():
            current_acc = self._calculate_class_accuracy(predicted, labels, class_id, mask_logits)
            if initial_acc - current_acc > threshold:
                num_classes_forgotten += 1

        cfi = num_classes_forgotten / len(self.initial_class_accuracies) if self.initial_class_accuracies else 0
        return cfi

    def _infer_task_for_class(self, class_id):
        # Check if the class_id is in the dictionary that tracks the last task of each class
        # if class_id in self.class_to_last_task:
        #   return self.class_to_last_task[class_id]
        # else:
        # If the class is not found, it might be a new class introduced in the current task
        # Alternatively, you can return a default value or raise an error
        # return "Unknown"  # or return a default value like 'Unknown'
        return self.class_to_last_task.get(class_id, -1)

    def test(self, personal=True):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.model.to(device)  # Ensure the model is on the same device as the data
        self.model.eval()
        test_acc = 0
        loss = 0

        if personal == True:
            for x, y in self.testloaderfull:
                x = x.to(device)
                y = y.to(device)
                output = self.model(x)['output']
                loss += self.loss(output, y)
                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()

        else:
            for x, y in self.testloaderfull:
                x = x.to(device)
                y = y.to(device)
                output = self.model(x)['output']
                loss += self.loss(output, y)
                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()

        return test_acc, loss, y.shape[0]

    def test_a_dataset(self, dataloader):
        '''
        test_acc: total correct samples
        loss: total loss (on a dataset)
        y_shape: total tested samples
        '''
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model.eval()
        test_acc = 0
        loss = 0
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            output = self.model(x)['output']
            loss += self.loss(output, y)
            test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()  # counts: how many correct samples
        return test_acc, loss, y.shape[0]

    def test_per_task(self):

        self.model.eval()
        test_acc = []
        loss = []
        y_shape = []

        # evaluate per task:
        for test_data in self.test_data_per_task:
            test_data_loader = DataLoader(test_data, len(test_data))
            test_acc_, loss_, y_shape_ = self.test_a_dataset(test_data_loader)

            test_acc.append(test_acc_)
            loss.append(loss_)
            y_shape.append(y_shape_)

        return test_acc, loss, y_shape

    def test_all(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model.eval()
        test_acc = 0
        loss = 0
        for x, y in self.test_data_so_far_loader:
            x = x.to(device)
            y = y.to(device)

            output = self.model(x)['output']
            loss += self.loss(output, y)
            test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
        return test_acc, loss, y.shape[0]

    def test_personalized_model(self):
        self.model.eval()
        test_acc = 0
        loss = 0
        self.update_parameters(self.personalized_model_bar)
        for x, y in self.testloaderfull:
            output = self.model(x)['output']
            loss += self.loss(output, y)
            test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
        self.update_parameters(self.local_model)
        return test_acc, y.shape[0], loss

    def get_next_train_batch(self, count_labels=True):
        try:
            # Samples a new batch for personalizing
            (X, y) = next(self.iter_trainloader)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_trainloader = iter(self.trainloader)
            (X, y) = next(self.iter_trainloader)
        result = {'X': X, 'y': y}
        if count_labels:
            unique_y, counts = torch.unique(y, return_counts=True)
            unique_y = unique_y.detach().numpy()
            counts = counts.detach().numpy()
            result['labels'] = unique_y
            result['counts'] = counts
        return result

    def get_next_test_batch(self):
        try:
            # Samples a new batch for personalizing
            (X, y) = next(self.iter_testloader)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_testloader = iter(self.testloader)
            (X, y) = next(self.iter_testloader)
        return (X, y)

    def save_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model, os.path.join(model_path, "user_" + self.id + ".pt"))

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        self.model = torch.load(os.path.join(model_path, "server" + ".pt"))

    @staticmethod
    def model_exists():
        return os.path.exists(os.path.join("models", "server" + ".pt"))

    def test_(self, personal=False):
        if self.my_model_name == 'fedcl':
            model = self.generator.critic
            if self.args.algorithm == 'FedGR':
                model = self.classifier.critic

        if self.my_model_name == 'fedlwf':
            model = self.model

        model.cuda()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.eval()
        test_acc = 0
        loss = 0
        all_logits = []
        all_labels = []

        if personal:
            for x, y in self.testloaderfull:
                x, y = x.to(device), y.to(device)

                if self.my_model_name == 'fedcl':
                    _, p, _ = model.forward(x)
                else:
                    p = model.forward(x)['output']

                loss += self.loss(torch.log(p), y).item()

                # mask irrelevant output logits:
                mask = list(set(self.label_set).difference(set(self.classes_so_far)))
                p[:, mask] = -9999

                test_acc += (torch.argmax(p, dim=1) == y).sum().item()
                all_logits.append(p.cpu().detach())
                all_labels.append(y.cpu().detach())

        else:
            for x, y in self.testloaderfull:
                x, y = x.to(device), y.to(device)

                if self.my_model_name == 'fedcl':
                    _, p, _ = model.forward(x)
                else:
                    p = model.forward(x)['output']

                loss += self.loss(torch.log(p), y).item()
                test_acc += (torch.argmax(p, dim=1) == y).sum().item()
                all_logits.append(p.cpu().detach())
                all_labels.append(y.cpu().detach())

        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        return test_acc, loss, all_labels.size(0), all_logits, all_labels

    def test_data_sofar(self, personal=False):
        if self.my_model_name == 'fedcl':
            model = self.generator.critic
            if self.args.algorithm == 'FedGR':
                model = self.classifier.critic

        if self.my_model_name == 'fedlwf':
            model = self.model

        model.cuda()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.eval()
        test_acc = 0
        loss = 0
        all_logits = []
        all_labels = []

        for x, y in self.test_data_so_far_loader:
            x, y = x.to(device), y.to(device)

            if self.my_model_name == 'fedcl':
                _, p, _ = model.forward(x)
            else:
                p = model.forward(x)['output']

            loss += self.loss(torch.log(p), y).item()

            if personal:
                mask = list(set(self.label_set).difference(set(self.classes_so_far)))
                p[:, mask] = -9999

            test_acc += (torch.argmax(p, dim=1) == y).sum().item()
            all_logits.append(p.cpu().detach())
            all_labels.append(y.cpu().detach())

        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        return test_acc, loss, all_labels.size(0), all_logits, all_labels

    """
    def test_a_dataset_(self, dataloader, personal = True):
        '''
        test_acc: total correct samples
        loss: total loss (on a dataset) 
        y_shape: total tested samples
        '''
        if self.my_model_name == 'fedcl':
            model = self.generator.critic
            if self.args.algorithm == 'FedGR':
                model = self.classifier.critic

        if self.my_model_name == 'fedlwf':
            model = self.model

        model.cuda()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model.eval()
        test_acc = 0
        loss = 0
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)


            if self.my_model_name == 'fedcl':
                _, p, _ = model.forward(x)
            else:
                p = model.forward(x)['output']  

            loss += self.loss(torch.log(p), y)

            if personal == True:
                # mask irrevelent output logits:
                # p: 1000 * 10
                mask = list(set(self.label_set).difference(set(self.classes_so_far)))
                p[:,mask] = -9999

            else:
                pass

            test_acc += (torch.sum(torch.argmax(p, dim=1) == y)).item() # counts: how many correct samples
        return test_acc, loss, y.shape[0]
    """

    def test_a_dataset_(self, dataloader, personal=True):
        if self.my_model_name == 'fedcl':
            model = self.generator.critic
            if self.args.algorithm == 'FedGR':
                model = self.classifier.critic

        if self.my_model_name == 'fedlwf':
            model = self.model

        model.cuda()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model.eval()

        test_acc = 0
        total_loss = 0
        total_samples = 0
        criterion = torch.nn.CrossEntropyLoss()  # Assuming CrossEntropyLoss, adjust according to your setup

        all_logits = []
        all_labels = []

        for x, y in dataloader:
            x = x.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
            y = y.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

            if self.my_model_name == 'fedcl':
                _, logits, _ = model.forward(x)
            else:
                logits = model.forward(x)['output']

            if personal:
                # Assuming self.label_set is the set of all labels and self.classes_so_far is the set of introduced classes
                mask = torch.tensor([label not in self.classes_so_far for label in range(logits.size(1))],
                                    device=logits.device)
                logits[:, mask] = float('-inf')  # Mask irrelevant logits

            loss = criterion(logits, y)
            total_loss += loss.item()

            predicted = torch.argmax(logits, dim=1)
            test_acc += (predicted == y).sum().item()
            total_samples += y.size(0)

            all_logits.append(logits.detach().cpu())
            all_labels.append(y.cpu())

        all_logits = torch.cat(all_logits, 0)
        all_labels = torch.cat(all_labels, 0)

        return test_acc, total_loss, total_samples, all_logits, all_labels

    """
    def test_per_task_(self, personal=True, mask_logits=True):
        if self.my_model_name == 'fedcl':
            model = self.generator.critic
            if self.args.algorithm == 'FedGR':
                model = self.classifier.critic

        if self.my_model_name == 'fedlwf':
            model = self.model

        model.cuda()

        self.generator.eval()
        # Ensure the model is in evaluation mode

        test_acc = []
        forgetting_measures = []

        # Evaluate per task
        for task_index, test_data in enumerate(self.test_data_per_task):
            test_data_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)
            _, _, _, all_logits, all_labels = self.test_a_dataset_(test_data_loader, personal=personal)

            # Calculate accuracy for the current task
            correct_predictions = torch.argmax(all_logits, dim=1) == all_labels
            task_accuracy = correct_predictions.float().mean().item()
            test_acc.append(task_accuracy)

            # Initialize forgetting measure calculation

            total_forgetting = 0
            for class_id, initial_acc in self.initial_class_accuracies.items():
                # Calculate current accuracy for the class
                class_mask = all_labels == class_id
                if class_mask.any():
                    class_correct = (torch.argmax(all_logits[class_mask], dim=1) == all_labels[
                        class_mask]).float().mean().item() * 100
                    forgetting_percentage = max(0, initial_acc - class_correct)
                else:
                    # If there are no samples for this class in the current task, we assume no forgetting
                    forgetting_percentage = 0
                total_forgetting += forgetting_percentage

            if task_index == 0:
                # No previous task to forget about, so forgetting measure is 0
                forgetting_measures.append(0)
                continue

                # For tasks after the first, calculate forgetting for previous tasks
            total_forgetting = 0
            for prev_task_index in range(task_index):
                prev_task_accuracy = self.task_accuracies_after_learning[prev_task_index]
                # Re-evaluate the model on previous task's test data to get current accuracy
                prev_test_data_loader = DataLoader(self.test_data_per_task[prev_task_index],
                                                   batch_size=len(self.test_data_per_task[prev_task_index]),
                                                   shuffle=False)
                _, _, _, prev_all_logits, prev_all_labels = self.test_a_dataset_(prev_test_data_loader,
                                                                                 personal=personal)
                prev_correct_predictions = torch.argmax(prev_all_logits, dim=1) == prev_all_labels
                current_task_accuracy = prev_correct_predictions.float().mean().item()

                forgetting_percentage = max(0, (prev_task_accuracy - current_task_accuracy) * 100)
                total_forgetting += forgetting_percentage

            # Calculate average forgetting across all previous tasks
            forgetting_measure = total_forgetting / task_index if task_index > 0 else 0
            forgetting_measures.append(forgetting_measure)

        return test_acc, forgetting_measures
    """

    def test_per_task_(self, personal=True, mask_logits=True):
        if self.my_model_name == 'fedcl':
            model = self.generator.critic
        elif self.args.algorithm == 'FedGR':
            model = self.classifier.critic
        elif self.my_model_name == 'fedlwf':
            model = self.model

        model.cuda()
        model.eval()  # Ensure the model is in evaluation mode

        test_acc = []
        forgetting_measures = []
        loss = []
        y_shape = []

        # Iterate over tasks with enumerate for index
        for task_index, test_data in enumerate(self.test_data_per_task):
            test_data_loader = DataLoader(test_data, len(test_data))
            test_acc_, loss_, y_shape_, all_logits, all_labels = self.test_a_dataset_(test_data_loader,
                                                                                      personal=personal)

            # Calculate accuracy for the current task
            correct_predictions = torch.argmax(all_logits, dim=1) == all_labels
            task_accuracy = correct_predictions.float().mean().item() * 100  # Convert to percentage
            task_accuracy_percentage = (test_acc_ / y_shape_) * 100
            test_acc.append(task_accuracy_percentage)
            loss.append(loss_)
            y_shape.append(y_shape_)
            # test_acc.append(task_accuracy)

            # Store the accuracy for the current task
            self.task_accuracies_after_learning[task_index] = task_accuracy

            # For the first task, there's no previous task to forget, hence forgetting measure is 0
            if task_index == 0:
                forgetting_measures.append(0)
                continue

            # For tasks after the first, calculate forgetting for previous tasks
            total_forgetting = 0
            for prev_task_index in range(task_index):
                prev_task_accuracy = self.task_accuracies_after_learning.get(prev_task_index)
                # Assuming prev_task_accuracy is correctly stored; if not, default to current task accuracy
                current_task_accuracy = self.task_accuracies_after_learning.get(task_index, task_accuracy)

                # Calculate forgetting as the decrease in performance on previous tasks
                forgetting_percentage = max(0, prev_task_accuracy - current_task_accuracy)
                total_forgetting += forgetting_percentage

            # Average forgetting measure across all previous tasks
            forgetting_measure = total_forgetting / task_index if task_index > 0 else 0
            forgetting_measures.append(forgetting_measure)

        return test_acc, forgetting_measures

    def test_all_(self, personal=False, matrix=False):

        if self.my_model_name == 'fedcl':
            model = self.generator.critic
            if self.args.algorithm == 'FedGR':
                model = self.classifier.critic
        if self.my_model_name == 'fedlwf':
            model = self.model

        model.cuda()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model.eval()
        test_acc = 0
        loss = 0
        predicts = []
        labels = []

        for x, y in self.test_data_so_far_loader:
            x = x.to(device)
            y = y.to(device)

            if self.my_model_name == 'fedcl':
                _, p, _ = model.forward(x)
            else:
                p = model.forward(x)['output']

            loss += self.loss(torch.log(p), y)  # if the probability of 'Y' is too small, log(p) -> INF

            #             # debug:
            #             if (self.loss(torch.log(p), y).item())**2 == float('-inf')**2:
            #                 np.set_printoptions(threshold=9999)
            #                 lo = nn.NLLLoss(reduction = 'none')

            #                 t = 0
            #                 for num, sample in enumerate(lo( torch.log(p), y )):
            #                     if (sample.item())**2 == float('-inf')**2:
            #                         t = num

            #                 print(p[t])
            #                 print(y[t])
            #                 exit()

            if personal == True:
                mask = list(set(self.label_set).difference(set(self.classes_so_far)))
                p[:, mask] = -9999

            else:
                pass

            test_acc += (torch.sum(torch.argmax(p, dim=1) == y)).item()

            if matrix == True:
                # confusion matrix
                predicts += torch.argmax(p, dim=1)
                labels += y

        if matrix == True:

            return test_acc, loss, y.shape[0], predicts, labels
        else:
            return test_acc, loss, y.shape[0]