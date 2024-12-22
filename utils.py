import torch
import numpy as np

def state_to_feature(state):
    feature = []
 
    for face in state.values():
        feature.extend(face[0, :])
        feature.extend(face[2, :])
        feature.extend(face[1, 0:2])

    return np.array(feature, dtype=int)

def feature_to_state(feature):
    face_keys = ['F', 'B', 'U', 'D', 'R', 'L']
    color_keys = [1, 2, 3, 4, 5, 6]
    state = {}
    idx = 0

    for i, face in enumerate(face_keys):
        face_array = np.zeros((3, 3), dtype=int)
        face_array[0, :] = feature[idx:idx + 3]
        face_array[2, :] = feature[idx + 3:idx + 6]
        face_array[1, 0] = feature[idx + 6]
        face_array[1, 2] = feature[idx + 7]
        face_array[1, 1] = color_keys[i]
        state[face] = face_array
        idx += 8

    return state

def hash_state(state_feature):
    return tuple(state_feature.tolist())

def hash_state_(state):
    return tuple(tuple(face.flatten().tolist()) for face in state.values())

def inverse_direction(direction):
    return '-' if direction == '+' else '+'

def arrays_are_equal(array_1, array_2):
    return np.array_equal(array_1, array_2) and array_1.shape == array_2.shape

def remove_sparse_rows(matrix):
    while matrix.shape[0] > 0 and np.all(matrix[-1] == -1):
        matrix = matrix[:-1]
    return matrix

def reconstruct_path(came_from, current_node):
    path = [current_node]

    while current_node in came_from:
        current_node = came_from[current_node]
        path.append(current_node)
    path.reverse()
    return path

def is_inverse(last_move, current_move):
    return last_move[0] == current_move[0] and last_move[1] != current_move[1]

def is_solved(state):
    for face_key in state.values():
        if not np.all(face_key == face_key[0, 0]):
            return False
    return True
