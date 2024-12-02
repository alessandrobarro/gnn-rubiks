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
import torch.nn.functional as F
from model_node_class import GCNModel, loss_function
from utils import *

np.set_printoptions(threshold=np.inf)
move_keys = [('F', '+'), ('F', '-'), ('B', '+'), ('B', '-'), ('U', '+'), ('U', '-'), ('D', '+'), ('D', '-'), ('R', '+'), ('R', '-'), ('L', '+'), ('L', '-')]

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
def create_dataset(graph, features, labels):
    edge_index = torch.tensor(list(graph.edges), dtype=torch.long).t().contiguous()
    x = torch.tensor(features, dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.long)
    
    data = Data(x=x, edge_index=edge_index, y=y)
    return data

#------------------------Train and Eval------------------------#
def train(model, data, optimizer, epochs=100):
    model.train()
    loader = DataLoader([data], batch_size=1)
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            optimizer.zero_grad()
            
            # Forward pass
            out = model(batch)
            loss = loss_function(out, batch.y)
            
            # Backward pass
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        
        print(f'Epoch {epoch + 1}, Loss: {total_loss:.4f}')
    
    return model

#---------------------Prediction and Search--------------------#
def compute_heuristic(model, cube):
    model.eval()
    
    # Crea il grafo locale dalla configurazione del cubo
    features, edge_index, neighbors = create_local_graph_from_configuration(cube)
    data = Data(x=features, edge_index=edge_index)
    
    with torch.no_grad():
        # Ottieni le predizioni dal modello
        out = model(data)  # [num_nodes, num_classes] - log-probabilities o raw logits
        
        # Predici la distanza come classe più probabile
        predicted_distances = torch.argmax(out, dim=1)  # Restituisce l'indice (distanza) più probabile
        #print(predicted_distances)
    
    # Restituisce le distanze predette per ogni nodo e i vicini
    return predicted_distances.tolist(), neighbors

def a_star_search(cube, model, num_moves_limit=5000):
    open_set = []
    heapq.heappush(open_set, (0, 0, cube.copy_state(), []))
    
    came_from = {}
    g_score = {hash_state(state_to_feature(cube.state)): 0}
    unique_counter = 1
    nodes_visited = 0  # Contatore per i nodi visitati

    while open_set:
        _, _, current_state, current_moves = heapq.heappop(open_set)
        
        # Incrementa il contatore dei nodi visitati
        nodes_visited += 1

        if is_solved(current_state):
            print(f"Nodes visited: {nodes_visited}")  # Stampa il numero di nodi visitati
            return current_moves, nodes_visited
        
        if len(current_moves) >= num_moves_limit:
            continue
        
        temp_cube = RubiksCube()
        temp_cube.restore_state(current_state)
        
        h_score, neighbors = compute_heuristic(model, temp_cube)
        #print(h_score)
        
        for i, neighbor in enumerate(neighbors):
            next_cube = RubiksCube()
            next_cube.restore_state(neighbor)
            
            next_state_feature = state_to_feature(next_cube.state)
            next_state_hash = hash_state(next_state_feature)
            
            tentative_g_score = g_score[hash_state(state_to_feature(current_state))] + 1
            
            if next_state_hash not in g_score or tentative_g_score < g_score[next_state_hash]:
                g_score[next_state_hash] = tentative_g_score
                f_score = tentative_g_score + (1/26) * h_score[i]  # f(s) = g(s) + h(s)
                #print(f_score)
                heapq.heappush(open_set, (f_score, unique_counter, next_cube.copy_state(), current_moves + [move_keys[i]]))
                unique_counter += 1

    print(f"Nodes visited: {nodes_visited}")  # Stampa anche se non risolto
    return nodes_visited

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
    num_train_nodes = 100000
    len_train_random_walks = 11
    num_classes = len_train_random_walks + 1

    # Init Rubik's Cube
    cube = RubiksCube()
    cube.reset_state()
    
    # Generate the train subgraph, train features, train distances from the solved state
    #train_graph, train_features, train_distances, train_rand_walks = generate_graph_from_random_walks(cube, num_train_nodes, len_train_random_walks)
    #print(train_distances)
    #plot_graph(train_graph)

    # Construct a train dataset
    #train_data = create_dataset(train_graph, train_features, train_distances)
    #print(train_data)

    # Define model and optimizer
    model = GCNModel(in_channels=48, out_channels=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    #trained_model = train(model, train_data, optimizer, epochs=250)
    #torch.save(model.state_dict(), 'weights_node_class_GCN_200k.pth')
    #print("Model weights saved successfully")
    model.load_state_dict(torch.load('weights_node_class_GCN_100k.pth'))
    model.eval()
    print("Model weights loaded successfully")

    # Construct an test graph and dataset
    comptime = []
    lengths = []
    numnodes = []
    with open("results.txt", "w") as file:
        for iter in range(0, 100):
            print(f"iter - {iter}")
            nshuffle = random.randint(5, 9)
            test_cube = RubiksCube()
            test_state, random_moves = generate_random_test_configuration(test_cube, num_moves=nshuffle)
            print(f"Test configuration generated from {nshuffle} random moves: {random_moves}\n")
            file.write(f"Test configuration generated from {nshuffle} random moves: {random_moves}\n")

            start_time = time.time()
            solution_moves, num_nodes = a_star_search(test_cube, model)
            end_time = time.time()

            if solution_moves:
                print(f"Solution found: {solution_moves}\n")
                file.write(f"Solution found: {solution_moves}\n")
            else:
                print("No solution was found\n")
                file.write("No solution was found\n")

            elapsed_time = end_time - start_time
            print(f"Execution time: {elapsed_time:.4f} seconds\n")
            file.write(f"Execution time: {elapsed_time:.4f} seconds\n")
            numnodes.append(num_nodes)
            lengths.append(len(solution_moves))
            comptime.append(elapsed_time)

        avg_time = sum(comptime) / len(comptime)
        avg_length = sum(lengths) / len(lengths)
        avg_nodes = sum(numnodes) / len(numnodes)
        file.write(f'\nAverage time: {avg_time:.4f} seconds\n')
        file.write(f'Average solution length: {avg_length:.4f}\n')
        file.write(f'Average nodes visited: {avg_nodes:.4f}\n')

        file.write(f'\nAll nodes visited: {numnodes}\n')
        file.write(f'All solution lengths: {lengths}\n')
        file.write(f'All computation times: {comptime}\n')

    print("Results saved to results.txt")

