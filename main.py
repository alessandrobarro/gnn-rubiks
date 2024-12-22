import torch
import heapq
import random
import time
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from cube import RubiksCube
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from model import GCNModel
from utils import *

np.set_printoptions(threshold=np.inf)

#/------------------------/HYPERPARAMS/-----------------------/
NUM_TRAIN_NODES = 100000
LEN_TRAIN_RAND_WALKS = 7
NUM_CLASSES = LEN_TRAIN_RAND_WALKS + 1
LEARNING_RATE = 0.02
EPOCHS = 500

#/---------------/GRAPH AND STATE GENERATION/-----------------/
def generate_graph_from_random_walks(cube, num_nodes, len_random_walks):
    graph = nx.DiGraph()
    features = []
    distances = []
    rand_walks = []
    state_to_node = {}

    start_state = cube.copy_state()
    start_state_feature = state_to_feature(cube.state)

    features.append(start_state_feature)
    distances.append(0)
    graph.add_node(0)
    
    state_to_node[hash_state(start_state_feature)] = 0

    while (len(graph.nodes)) < num_nodes:
        rand_walk = []
        cube.restore_state(start_state)
        current_state = cube.copy_state()
        current_node = 0
        
        for t in range(len_random_walks):
            current_feature = state_to_feature(current_state)
            rand_move = random.sample(cube.move_keys, 1)[0]
            cube.move(rand_move[0], rand_move[1])
            next_state = cube.copy_state()
            next_state_feature = state_to_feature(next_state)
            next_state_hash = hash_state(next_state_feature)
            
            if next_state_hash not in state_to_node:
                new_node = len(graph.nodes)
                graph.add_node(new_node)
                features.append(next_state_feature)
                distances.append(t + 1)
                state_to_node[next_state_hash] = new_node
                target_node = new_node
            else:
                target_node = state_to_node[next_state_hash]
                if t + 1 < distances[target_node]:
                    distances[target_node] = t + 1
            
            graph.add_edge(current_node, target_node, action=f'{rand_move[0]}{rand_move[1]}')
            graph.add_edge(target_node, current_node, action=f'{rand_move[0]}{inverse_direction(rand_move[1])}')
            
            current_node = target_node
            current_state = next_state
            rand_walk.append(current_node)
        
        rand_walks.append(rand_walk)
    print(len(rand_walks))
    return graph, np.array(features), np.array(distances), rand_walks

def create_local_graph_from_configuration(cube):
    current_state = cube.copy_state()
    features = [state_to_feature(current_state)]
    edge_index = []
    neighbors = []
    
    for i, move in enumerate(cube.move_keys):
        cube.move(move[0], move[1])
        next_state = cube.copy_state()
        next_feature = state_to_feature(next_state)
        features.append(next_feature)
        neighbors.append(next_state)
        edge_index.append([0, i + 1])
        edge_index.append([i + 1, 0])
        cube.restore_state(current_state)
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    features = torch.tensor(np.array(features), dtype=torch.float)
    
    return features, edge_index, neighbors,

#/-------------------/DATASET CONSTRUCTION/--------------------/
def create_dataset(graph, features, labels):
    edge_index = torch.tensor(list(graph.edges), dtype=torch.long).t().contiguous()
    x = torch.tensor(features, dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y)
    return data

#/----------------------/TRAIN AND EVAL/-----------------------/
def train(model, data, optimizer, epochs=200):
    model.train()
    loader = DataLoader([data], shuffle=False)
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            optimizer.zero_grad()
            
            out = model(batch)
            loss = loss_function(out, batch.y)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 20 == 0:
            print(f'EPOCH {epoch} - LOSS: {total_loss:.6f}')
    
    return model

#/----------------------/PRED AND SEARCH/----------------------/
def compute_heuristic(model, cube):
    model.eval()
    features, edge_index, neighbors = create_local_graph_from_configuration(cube)
    data = Data(x=features, edge_index=edge_index)
    with torch.no_grad():
        out = model(data)
        predicted_distances = torch.argmax(out, dim=1)
    return predicted_distances.tolist(), neighbors

def a_star_search(cube, model):
    start_state = cube.copy_state()
    start_hash = hash_state_(start_state)

    open_set = []
    heapq.heappush(open_set, (0, start_hash))
    state_data = {start_hash: (start_state, [], 0)}
    closed_set = set()
    nodes_expanded = 0

    start_time = time.time()

    while open_set:
        _, current_hash = heapq.heappop(open_set)
        current_state, path, g_cost = state_data[current_hash]
        nodes_expanded += 1
        
        if is_solved(current_state):
            search_time = time.time() - start_time
            return len(path), nodes_expanded, search_time

        closed_set.add(current_hash)

        h_score, neighbors = compute_heuristic(model, cube)

        for i, move in enumerate(cube.move_keys): # ATTENZIONE: ORDINAMENTO EURISTICA E ORDINE MOSSE
            cube.restore_state(current_state)
            cube.move(*move)
            next_state = cube.copy_state()
            next_hash = hash_state_(next_state)

            if next_hash in closed_set:
                continue

            f_cost = g_cost + 1 + (1 / 7) * (h_score[i + 1] + 1) # [0] E' IL NODO STESSO 

            if next_hash not in state_data or g_cost + 1 < state_data[next_hash][2]:
                state_data[next_hash] = (next_state, path + [move], g_cost + 1)
                heapq.heappush(open_set, (f_cost, next_hash))
            
    return None, nodes_expanded, time.time() - start_time

#/----------------------------/TEST/---------------------------/
def test():
    comp_time = []
    path_lengths = []
    nums_nodes = []

    for iter in range(0, 1):
        n_shuffle = random.randint(5, 7)
        test_cube = RubiksCube()
        test_cube.shuffle_state(n_shuffle)
        start_time = time.time()
        path_len, num_visited_nodes, elapsed_time = a_star_search(test_cube, model)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f'COMPL TEST {iter} - LENGTH: {path_len}, NUM_NODES: {num_visited_nodes}, ELAPSED_TIME: {elapsed_time}')

        comp_time.append(elapsed_time)
        nums_nodes.append(num_visited_nodes)
        path_lengths.append(path_len)
        iter += 1

    print(f'AVG TIME {sum(comp_time)/len(comp_time)} - AVG LEN {sum(path_lengths)/len(path_lengths)} - AVG VIS_NODES {sum(num_visited_nodes)/len(num_visited_nodes)}\n')

if __name__ == "__main__":    

    cube = RubiksCube()
    cube.reset_state()
    
    print("/----------------/DATA PREP PHASE/----------------/")
    train_graph, train_features, train_distances, train_rand_walks = generate_graph_from_random_walks(cube, NUM_TRAIN_NODES, LEN_TRAIN_RAND_WALKS)
    print(f'GENERATED GRAPH OF {NUM_TRAIN_NODES} NODES FROM {LEN_TRAIN_RAND_WALKS} LONG RWs')
    unique, counts = np.unique(train_distances, return_counts=True)
    freqs = dict(zip(unique, counts))
    print("DISTANCE DISTRIBUTION:", freqs)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    class_weights = torch.tensor([1.0 / freqs[i] for i in range(len(freqs))], dtype=torch.float).to(device)
    class_weights = class_weights / class_weights.sum()

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    def loss_function(output, target):
        return criterion(output, target)

    train_data = create_dataset(train_graph, train_features, train_distances)

    model = GCNModel(in_channels=48, out_channels=NUM_CLASSES)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("/-----------------/TRAINING PHASE/----------------/")
    trained_model = train(model, train_data, optimizer, epochs=EPOCHS)
    torch.save(model.state_dict(), 'weights.pth')

    model.load_state_dict(torch.load('weights.pth'))
    model.eval()

    print("\n/------------------/TEST PHASE/-------------------/")
    eval()
    test()
    

