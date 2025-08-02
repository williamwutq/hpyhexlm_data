# Require the hpyhex module from the python end of game HappyHex

from random import random
from hpyhex.hex import HexEngine, Piece, Hex
from hpyhex.game_env import Game


__all__ = ["save_training_dataset", "load_training_data", "generate_training_data"]


"""
A simple helper for saving, loading, and generating machine learning data for the HappyHex Autoplay feature.
Contains the following functions:

- save_training_dataset: save training data to a text file
- load_training_data: load training data from a text file
- generate_training_data: generate training data based on a certain algorithm,
  which will be provided in its sample directory as a function, and optionally store or return that data.
"""


def save_training_dataset(data: list[tuple[HexEngine, list[Piece], list[tuple[int, Hex]]]], filename: str, print_err: bool = False) -> None:
    """
    Save training data to a file.
    :param data: A list of tuples containing the engine, queue, and best options.
    :param filename: The file to save the training data to.
    :param print_err: Whether to print error if occured.
    """
    try:
        with open(filename, 'w') as f:
            for engine, queue, best_options in data:
                try:
                    # Engine booleans as string of 0s and 1s
                    engine_data = ''.join(['1' if b else '0' for b in engine.to_booleans()])
                    # Queue as comma-separated bytes
                    queue_data = ','.join(str(piece.to_byte()) for piece in queue)
                    # Best options as index:line:pos
                    result_data = ','.join(f"{idx}:{coord.line_tuple()[0]}:{coord.line_tuple()[1]}" for idx, coord in best_options)
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
    :param filename: The file to load the training data from.
    :return: A list of tuples containing the engine, queue, and best options.
    """
    dataset = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                engine_str, queue_str, result_str = [part.strip() for part in line.strip().split('|')]
                # Reconstruct engine
                engine_bools = [c == '1' for c in engine_str]
                engine = HexEngine.engine_from_booleans(engine_bools)
                # Reconstruct queue
                queue = [Piece.piece_from_byte(int(b)) for b in queue_str.split(',')]
                # Reconstruct results
                results = []
                for res in result_str.split(','):
                    idx, line_no, pos_no = map(int, res.split(':'))
                    results.append((idx, Hex.hex(int(line_no), int(pos_no))))
                dataset.append((engine, queue, results))
    except IOError as e:
        if print_err:
            print(f"Error reading file {filename}: {e}")
    return dataset


def generate_training_data(num_samples: int, algorithm, filename: str = "",
                           engine_radius: int = 5, queue_size: int = 3, significant_choices: int = 7,
                           remove_head: float = 0.0, remove_tail: float = 0.05, move_dropout: float = 0.05,
                           return_data: bool = False, verbose: bool = True) -> list[tuple[HexEngine, list[Piece], list[tuple[int, Hex]]]] | None:
    """
    Generate a training dataset with random game states and save it to a file if filename is given.
    :param num_samples: The number of samples to generate.
    :param algorithm: The algorithm to generate the samples.
    :param filename: The file to save the training data to.
    :param engine_radius: The radius of the HexEngine.
    :param queue_size: The size of the queue of pieces.
    :param significant_choices: The number of significant choices to consider for each game state.
    :param remove_head: The fraction of the game data to remove from the start.
    :param remove_tail: The fraction of the game data to remove from the end.
    :param move_dropout: The probability of dropping a move from the game data.
    :param save_file: Whether to save the generated data to a file.
    :param return_data: Whether to return the generated data.
    :param verbose: Whether to print progress messages.
    """
    data = []
    sample = 0
    while len(data) < num_samples:
        inner_data = []
        game = Game(engine_radius, queue_size, None)
        while not game.is_game_end():
            # Run the algorithm to get the best moves
            best_moves = algorithm(game._engine, game._queue, significant_choices)
            if not best_moves:
                break
            # Make the first best move
            piece_index, coord = best_moves[0]
            if not game.add_piece(piece_index, coord):
                break
            # Collect the engine, queue, and best options
            if random() > move_dropout:
                # Only add the data if the move is not dropped
                inner_data.append((game._engine.__copy__(), game._queue.copy(), best_moves))
        # Strip the last 5% of the game data as it is not meaningful, saving the other 95% to data
        data_len = len(inner_data)
        if data_len > 0:
            data.extend(inner_data[int(data_len * remove_head):int(data_len * (1 - remove_tail))])
        turn, score = game.game_result()
        sample += 1
        if verbose:
            print(f"Game {sample} ends with {turn}, {score}, {100*len(data)/num_samples:.2f}% complete.")
    # Strip the extra data to the specified number of samples
    data = data[:num_samples]
    if verbose:
        print(f"Generated {len(data)} samples.")
    if filename: # If the filename is specified, save the data
        save_training_dataset(data, filename)
    if return_data:
        return data
    else:
        return None
