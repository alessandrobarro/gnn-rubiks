import torch
import heapq
import random
import time

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from cube import RubiksCube
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from model_edge_class import GATModel, loss_function
from utils import *

np.set_printoptions(threshold=np.inf)
move_keys = [('F', '+'), ('F', '-'), ('B', '+'), ('B', '-'), ('U', '+'), ('U', '-'), ('D', '+'), ('D', '-'), ('R', '+'), ('R', '-'), ('L', '+'), ('L', '-')]
train_keys = [('F', '+'), ('U', '+'), ('R', '+')]

#-----------------------Graph and state generation-----------------------#
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
            rand_move = random.sample(move_keys, 1)[0]
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
    
    return graph, np.array(features), np.array(distances), rand_walks

def generate_random_test_configuration(cube, num_moves):
    cube.reset_state()
    random_moves = random.choices(move_keys, k=num_moves)
    for move in random_moves:
        cube.move(move[0], move[1])
    return cube.copy_state(), random_moves

def create_local_graph_from_configuration(cube):
    current_state = cube.copy_state()
    features = [state_to_feature(current_state)]
    edge_index = []
    neighbors = []
    
    for i, move in enumerate(move_keys):
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

#---------------------Dataset construction---------------------#
def create_dataset(graph, features, edge_labels):
    edge_index = torch.tensor(list(graph.edges), dtype=torch.long).t().contiguous()
    x = torch.tensor(features, dtype=torch.float)
    edge_attr = torch.tensor(edge_labels, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data

def calculate_optimal_actions(graph, distances):
    edge_labels = []

    for u, v, edge_data in graph.edges(data=True):
        if distances[v] < distances[u]:
            edge_labels.append(1.0)
        else:
            edge_labels.append(0.0)
    
    return np.array(edge_labels)

#------------------------Train and Eval------------------------#
def train_model(model, data, optimizer, epochs=100):
    model.train()
    loader = DataLoader([data], batch_size=1)

    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            optimizer.zero_grad()

            # Forward pass
            edge_logits = model(batch)
            loss = loss_function(edge_logits, batch.edge_attr)  # Loss sugli edge
            
            # Backward pass
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Epoch {epoch + 1}, Loss: {total_loss:.4f}')

    return model

#---------------------Prediction and Search--------------------#
def predict_edge_probabilities(model, data):
    model.eval()
    with torch.no_grad():
        edge_probs = model(data)
    return edge_probs

def compute_heuristic(model, cube, lambda_=1.0):
    model.eval()

    features, edge_index, neighbors = create_local_graph_from_configuration(cube)
    data = Data(x=features, edge_index=edge_index)

    with torch.no_grad():
        edge_probs = model(data)
    
    heuristic = lambda_ * (1 - edge_probs)

    return heuristic.tolist(), neighbors

def a_star_search(cube, model, lambda_, num_moves_limit=5000):
    open_set = []
    heapq.heappush(open_set, (0, 0, cube.copy_state(), []))
    
    came_from = {}
    g_score = {hash_state(state_to_feature(cube.state)): 0}
    unique_counter = 1

    while open_set:
        _, _, current_state, current_moves = heapq.heappop(open_set)
        
        if is_solved(current_state):
            return current_moves
        
        if len(current_moves) >= num_moves_limit:
            continue
        
        temp_cube = RubiksCube()
        temp_cube.restore_state(current_state)
        
        h_score, neighbors = compute_heuristic(model, temp_cube, lambda_)
        #print(h_score)
        
        for i, neighbor in enumerate(neighbors):
            next_cube = RubiksCube()
            next_cube.restore_state(neighbor)
            
            next_state_feature = state_to_feature(next_cube.state)
            next_state_hash = hash_state(next_state_feature)
            
            tentative_g_score = g_score[hash_state(state_to_feature(current_state))] + 1
            
            if next_state_hash not in g_score or tentative_g_score < g_score[next_state_hash]:
                g_score[next_state_hash] = tentative_g_score
                f_score = tentative_g_score + h_score[i]  # f(s) = g(s) + h(s)
                heapq.heappush(open_set, (f_score, unique_counter, next_cube.copy_state(), current_moves + [move_keys[i]]))
                unique_counter += 1

    return None

def retrieve_action_moves(graph, solution_path):
    action_moves = []
    
    for i in range(len(solution_path) - 1):
        current_node = solution_path[i]
        next_node = solution_path[i + 1]
        
        edge_data = graph.get_edge_data(current_node, next_node)
        
        if edge_data is not None:
            action = edge_data['action']
            action_tuple = (action[0], action[1])
            action_moves.append(action_tuple)
        else:
            action_moves.append(None)
    
    return action_moves

#-----------------------------Plot------------------------------#
def plot_graph(graph):
    pos = nx.spring_layout(graph, k=0.5)
    node_sizes = [5] * len(graph.nodes())
    node_sizes[0] = 7
    node_colors = ['lightblue'] * len(graph.nodes())
    node_colors[0] = 'lime'

    nx.draw_networkx_nodes(graph, pos, node_size=node_sizes, node_color=node_colors, label=False)
    nx.draw_networkx_edges(graph, pos, edge_color='grey', arrows=False)
    plt.show()

if __name__ == "__main__":

    # Parameters
    train_optimal_paths = {1: [('L', '+'), ('L', '+'), ('D', '-'), ('F', '+'), ('F', '+'), ('L', '+'), ('L', '+'), ('F', '+'), ('F', '+'), ('U', '-'), ('B', '+'), ('B', '+'), ('F', '-'), ('D', '+'), ('D', '+'), ('F', '+'), ('F', '+'), ('L', '+'), ('F', '+'), ('U', '+'), ('L', '-'), ('R', '+'), ('B', '+'), ('R', '-'), ('F', '+'), ('F', '+'), ('R', '+'), ('R', '+'), ('U', '+'), ('U', '+')]}
    num_train_nodes = 10000
    len_train_random_walks = 10
    lambda_ = 10.0

    # Init Rubik's Cube
    cube = RubiksCube()
    cube.reset_state()
    
    # Generate the train subgraph
    #train_graph, train_features, train_distances, train_rand_walks = generate_graph_from_random_walks(cube, num_train_nodes, len_train_random_walks)
    #print(train_distances)
    #plot_graph(train_graph)

    # Calculate the ground truth probability vector from distances
    #train_optimal_actions = calculate_optimal_actions(train_graph, train_distances)
    #print(train_optimal_actions)

    # Construct a train dataset
    #train_data = create_dataset(train_graph, train_features, train_optimal_actions)
    #print(train_data)

    # Define model and optimizer
    model = GATModel(in_channels=48)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    #trained_model = train_model(model, train_data, optimizer, epochs=150)
    #torch.save(model.state_dict(), 'weights_edge_class_10k.pth')
    #print("Model weights saved successfully")
    model.load_state_dict(torch.load('weights_edge_class_10k.pth'))
    model.eval()
    print("Model weights loaded successfully")

    # Construct an test graph and dataset
    comptime = []
    lengths = []
    numnodes = []
    for iter in range(0, 10):
        print(iter)
        test_cube = RubiksCube()
        test_state, random_moves = generate_random_test_configuration(test_cube, num_moves=5)
        print("Test configuration generated from random moves:", random_moves)

        start_time = time.time()
        solution_moves, num_nodes = a_star_search(test_cube, model, lambda_)
        end_time = time.time()

        if solution_moves:
            print("Solution found:", solution_moves)
        else:
            print("No solution was found")

        elapsed_time = end_time - start_time
        print(f"Execution time: {elapsed_time:.4f} seconds")
        numnodes.append(num_nodes)
        lengths.append(len(solution_moves))
        comptime.append(elapsed_time)
        iter += 1

    print(f'avg time {sum(comptime)/len(comptime)} - avg length {sum(lengths)/len(lengths)} - avg visited {sum(numnodes)/len(numnodes)}')



