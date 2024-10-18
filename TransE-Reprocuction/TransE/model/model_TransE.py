import codecs
import copy
import random
import time
import numpy as np


class LxDistance:
    @staticmethod
    def l1_distance(h, r, t):
        return np.sum(np.fabs(h + r - t))

    @staticmethod
    def l2_distance(h, r, t):
        return np.sum(np.square(h + r - t))

    @staticmethod
    def l2_grad(h, r, t):
        return 2 * (h + r - t)

    @staticmethod
    def l1_grad(h, r, t):
        temp_d = h + r - t
        grad = temp_d
        for i in range(len(temp_d)):
            if (temp_d[i] > 0):
                grad[i] = 1
            else:
                grad[i] = -1
        return grad


class TransE(LxDistance):
    def __init__(self, entity_set, relation_set, triple_list, embedding_dim=50,
                 learning_rate=0.01, margin=1, L1=True):
        super(LxDistance, self).__init__()
        self.entity_set = entity_set
        self.relation_set = relation_set
        self.triple_list = triple_list
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.margin = margin
        self.L1 = L1
        self.loss = 0

    def embedding_init(self):
        relation_dict = {}
        entity_dict = {}

        # initialize the relation
        for r in self.relation_set:
            r_embedding = np.random.uniform(
                low=(-6 / np.sqrt(self.embedding_dim)),
                high=(6 / np.sqrt(self.embedding_dim)),
                size=self.embedding_dim,
            )
            # L2 normalization
            relation_dict[r] = r_embedding / np.linalg.norm(r_embedding, ord=2)

        # initialize the entity
        for e in self.entity_set:
            e_embedding = np.random.uniform(
                low=(-6 / np.sqrt(self.embedding_dim)),
                high=(6 / np.sqrt(self.embedding_dim)),
                size=self.embedding_dim,
            )
            entity_dict[e] = e_embedding
        return relation_dict, entity_dict

    def triple_corrupt(self, triple):
        corrupted_triple = copy.deepcopy(triple)
        rand_seed = random.random()

        # corrupt triple with head or tail replaced by a random entity
        # (head, tail, relation)
        if rand_seed > 0.5:
            head = triple[0]
            rand_head = head
            while (rand_head == head):
                rand_head = random.randint(0, len(self.entity_set) - 1)
            corrupted_triple[0] = rand_head
        else:
            tail = triple[1]
            rand_tail = tail
            while (rand_tail == tail):
                rand_tail = random.randint(0, len(self.entity_set) - 1)
            corrupted_triple[1] = rand_tail
        return corrupted_triple

    def hinge_loss(self, dist_correct, dist_corrupt):
        return max(0, dist_correct - dist_corrupt + self.margin)

    def gen_update_dict(self, t_batch):
        entity_update = {}
        relation_update = {}

        for t, ct in t_batch:
            if t[0] in entity_update.keys():
                pass
            else:
                entity_update[t[0]] = (
                    copy.copy(self.entity_set[t[0]]))

            if t[1] in entity_update.keys():
                pass
            else:
                entity_update[t[1]] = (
                    copy.copy(self.entity_set[t[1]]))

            if t[2] in relation_update.keys():
                pass
            else:
                relation_update[t[2]] = (
                    copy.copy(self.relation_set[t[2]]))

            if ct[0] in entity_update.keys():
                pass
            else:
                entity_update[ct[0]] = (
                    copy.copy(self.entity_set[ct[0]]))

            if ct[1] in entity_update.keys():
                pass
            else:
                entity_update[ct[1]] = (
                    copy.copy(self.entity_set[ct[1]]))
        return entity_update, relation_update

    def update_embeddings(self, t_batch):
        entity_update, relation_update = self.gen_update_dict(t_batch)

        for triple, corrupted_triple in t_batch:
            h_correct = self.entity_set[triple[0]]
            t_correct = self.entity_set[triple[1]]

            r = self.relation_set[triple[2]]

            h_corrupt = self.entity_set[corrupted_triple[0]]
            t_corrupt = self.entity_set[corrupted_triple[1]]

            if self.L1:
                dist_correct = self.l1_distance(h_correct, r, t_correct)
                dist_corrupt = self.l1_distance(h_corrupt, r, t_corrupt)
            else:
                dist_correct = self.l2_distance(h_correct, r, t_correct)
                dist_corrupt = self.l2_distance(h_corrupt, r, t_corrupt)

            err = self.hinge_loss(dist_correct, dist_corrupt)

            if err > 0:
                self.loss += err
                if self.L1:
                    grad_pos = self.l1_grad(h_correct, r, t_correct)
                    grad_neg = self.l1_grad(h_corrupt, r, t_corrupt)
                else:
                    grad_pos = self.l2_grad(h_correct, r, t_correct)
                    grad_neg = self.l2_grad(h_corrupt, r, t_corrupt)

                entity_update[triple[0]] -= (
                        self.learning_rate * grad_pos)
                entity_update[triple[1]] -= (
                        (-1) * self.learning_rate * grad_pos)

                entity_update[corrupted_triple[0]] -= (
                        (-1) * self.learning_rate * grad_neg)
                entity_update[corrupted_triple[1]] -= (
                        self.learning_rate * grad_neg)

                relation_update[triple[2]] -= (
                        self.learning_rate * grad_pos)
                relation_update[triple[2]] -= (
                        (-1) * self.learning_rate * grad_neg)

        # batch norm
        # prevent loss from minimizing during training
        for i in entity_update.keys():
            entity_update[i] /= np.linalg.norm(entity_update[i])
            self.entity_set[i] = entity_update[i]

        for i in relation_update.keys():
            relation_update[i] /= np.linalg.norm(relation_update[i])
            self.relation_set[i] = relation_update[i]
        return None

    def train(self, epochs, batch_size):
        global_start_time = time.time()
        # initialization
        self.relation_set, self.entity_set = self.embedding_init()

        n_batch = len(self.triple_list) // batch_size
        print(f'Number of batch: {n_batch}')

        for epoch in range(epochs):
            start_time = time.time()

            self.loss = 0

            # batch update
            for i in range(n_batch):
                s_batch = random.sample(self.triple_list, batch_size)
                t_batch = []  # negative samples

                # create negative samples
                for triple in s_batch:
                    corrupted_triple = self.triple_corrupt(triple)
                    t_batch.append((triple, corrupted_triple))

                # update embeddings
                self.update_embeddings(t_batch)

            end_time = time.time()

            # console log information
            print(f'Epoch: {epoch}, '
                  f'Epoch loss: {self.loss}, '
                  f'Running time: {round((end_time - start_time), 3)}')
        global_end_time = time.time()
        total_time = global_end_time - global_start_time
        print(f'Training Done! Total Time: {total_time}')

        # Save Model
        print('Saving Model...')
        with codecs.open(
                f"../results/entity_{self.embedding_dim}dim_batch{epochs}",
                "w") as f1:
            for e in self.entity_set.keys():
                f1.write(str(e) + "\t")
                f1.write(str(list(self.entity_set[e])))
                f1.write("\n")

        with codecs.open(
                f"../results/relation_{self.embedding_dim}dim_batch{epochs}",
                "w") as f2:
            for r in self.relation_set.keys():
                f2.write(str(r) + "\t")
                f2.write(str(list(self.relation_set[r])))
                f2.write("\n")
        print('Done!')
        return None
