# Require the hpyhex module from the python end of game HappyHex


'''
A simple helper for saving, loading, and generating machine learning data for the HappyHex Autoplay feature.
Contains the following functions:

- save_training_dataset: save training data to a text file
- load_training_data: load training data from a text file
- generate_training_data: generate training data based on a certain algorithm,
  which will be provided in its sample directory as a function, and optionally store or return that data.
'''

from random import random
from hpyhex.hex import HexEngine, Piece, Hex
from hpyhex.game import Game


__all__ = ["save_training_dataset", "load_training_data", "generate_training_data"]


def save_training_dataset(data: list[tuple[HexEngine, list[Piece], list[tuple[int, Hex]]]], filename: str, print_err: bool = False) -> None:
    '''
    Save training data to a file.
    
    Parameters:
        data (list): A list of tuples containing the engine, queue, and best options.
        filename (str): The file to save the training data to.
        print_err (bool): Whether to print errors if they occur during saving.
    Returns:
        None
    '''
    try:
        with open(filename, 'w') as f:
            for engine, queue, best_options in data:
                try:
                    # Engine booleans as string of 0s and 1s
                    engine_data = repr(engine)
                    # Queue as comma-separated bytes
                    queue_data = ','.join(repr(piece) for piece in queue)
                    # Best options as index:line:pos
                    result_data = ','.join(f"{idx}:{coord.i}:{coord.k}" for idx, coord in best_options)
                    # Write line
                    f.write(f"{engine_data} | {queue_data} | {result_data}\n")
                except Exception as e:
                    if print_err:
                        print(f"Error saving data for engine {engine}: {e}")
    except IOError as e:
        if print_err:
            print(f"Error writing to file {filename}: {e}")


def load_training_data(filename: str, print_err: bool = False) -> list[tuple[HexEngine, list[Piece], list[tuple[int, Hex]]]]:
    """
    Load training data from a file.

    Parameters:
        filename (str): The file to load the training data from.
        print_err (bool): Whether to print errors if they occur during loading.
    Returns:
        data (list[tuple]): A list of tuples containing the engine, queue, and best options.
    """
    dataset = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                engine_str, queue_str, result_str = [part.strip() for part in line.strip().split('|')]
                # Reconstruct engine
                engine = HexEngine(engine_str)
                # Reconstruct queue
                queue = [Piece(int(b)) for b in queue_str.split(',')]
                # Reconstruct results
                results = []
                for res in result_str.split(','):
                    idx, line_no, pos_no = map(int, res.split(':'))
                    results.append((idx, Hex(int(line_no), int(pos_no))))
                dataset.append((engine, queue, results))
    except IOError as e:
        if print_err:
            print(f"Error reading file {filename}: {e}")
    return dataset


def generate_training_data(num_samples: int, algorithm,
                           engine_radius: int = 5, queue_size: int = 3, significant_choices: int = 7,
                           remove_head: float = 0.0, remove_tail: float = 0.05, move_dropout: float = 0.05,
                           verbose: bool = True) -> list[tuple[HexEngine, list[Piece], list[tuple[int, Hex]]]]:
    '''
    Generate a training dataset with random game states and save it to a file if filename is given.
    
    Parameters:
        num_samples (int): The number of samples to generate.
        algorithm (callable): The algorithm to use for generating the game states.
        engine_radius (int): The radius of the HexEngine.
        queue_size (int): The size of the queue of pieces.
        significant_choices (int): The number of significant choices to consider.
        remove_head (float): The fraction of the first part of the game data to remove.
        remove_tail (float): The fraction of the last part of the game data to remove.
        move_dropout (float): The probability of dropping a move.
        return_data (bool): Whether to return the generated data instead of saving it.
        verbose (bool): Whether to print progress messages.
    Returns:
        data (list[tuple]): A list of tuples containing the engine, queue, and best options.
    '''
    data = []
    sample = 0
    while len(data) < num_samples:
        inner_data = []
        game = Game(engine_radius, queue_size)
        while not game.end:
            # Run the algorithm to get the best moves
            best_moves = algorithm(game.engine, game.queue, significant_choices)
            if not best_moves:
                break
            # Make the first best move
            piece_index, coord = best_moves[0]
            copy_engine = game.engine.__copy__()
            copy_queue = game.queue.copy()
            if not game.add_piece(piece_index, coord):
                break
            # Collect the engine, queue, and best options
            if random() > move_dropout:
                # Only add the data if the move is not dropped
                inner_data.append((copy_engine, copy_queue, best_moves))
        # Strip the last 5% of the game data as it is not meaningful, saving the other 95% to data
        data_len = len(inner_data)
        if data_len > 0:
            data.extend(inner_data[int(data_len * remove_head):int(data_len * (1 - remove_tail))])
        turn, score = game.result
        sample += 1
        if verbose:
            print(f"Game {sample} ends with {turn}, {score}, {100*len(data)/num_samples:.2f}% complete.")
    # Strip the extra data to the specified number of samples
    data = data[:num_samples]
    if verbose:
        print(f"Generated {len(data)} samples.")
    return data