import codecs


class DataLoader:
    @staticmethod
    def train_dataloader(file_path, kb_set):
        entity2id = {}
        relation2id = {}

        entity_set = set()  # use to set remove duplicate values
        relation_set = set()  # use to set remove duplicate values
        triple_list = []

        entity_file = file_path + 'entity2id.txt'
        relation_file = file_path + 'relation2id.txt'
        train_file = file_path + kb_set

        with codecs.open(entity_file, 'r') as e:
            e_lines = e.readlines()
            for line in e_lines:
                line = line.strip().split('\t')
                if len(line) != 2:
                    continue
                entity2id[line[0]] = int(line[1])

        with codecs.open(relation_file, 'r') as r:
            r_lines = r.readlines()
            for line in r_lines:
                line = line.strip().split('\t')
                if len(line) != 2:
                    continue
                relation2id[line[0]] = int(line[1])

        with codecs.open(train_file, 'r') as f:
            content = f.readlines()
            for line in content:
                triple = line.strip().split("\t")
                if len(triple) != 3:
                    continue

                h_ = entity2id[triple[0]]
                t_ = entity2id[triple[1]]
                r_ = relation2id[triple[2]]

                triple_list.append([h_, t_, r_])

                entity_set.add(h_)
                entity_set.add(t_)

                relation_set.add(r_)

        return entity_set, relation_set, triple_list
