from data import data_loader
from model import model_TransE

if __name__ == '__main__':
    file_path = "../datasets/FB15k/FB15k/"
    entity_set, relation_set, triple_list = (
        data_loader.DataLoader.train_dataloader(file_path, 'train.txt'))
    print("load file...")
    print("Complete load. entity : %d , relation : %d , triple : %d" % (
        len(entity_set), len(relation_set), len(triple_list)))

    transE = model_TransE.TransE(
        entity_set=entity_set,
        relation_set=relation_set,
        triple_list=triple_list,
        embedding_dim=50,
        learning_rate=0.01,
        margin=1,
        L1=True,
    )
    transE.train(epochs=1000, batch_size=128)
