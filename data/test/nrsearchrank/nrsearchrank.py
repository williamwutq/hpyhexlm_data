# This is the algorithm used to generate the data contained in this directory.
# Require the hpyhex module from the python end of game HappyHex

# Algorithm: nrsearchrank
# Also named: nrsearcheven, nrsearch


from hpyhex.hex import HexEngine, Piece, Hex
from hpyhex.game_env import Game


__all__ = ["nrsearchrank"]


def nrsearchrank(engine: HexEngine, queue: list[Piece], significant_choices: int = 9) -> list[tuple[int, Hex]]:
    """
    A heuristic algorithm that selects the best pieces and positions based on the dense index, and score gain of the game state.
    :param engine: The game engine
    :param queue: The queue of pieces available for placement
    :return: A list of tuples containing the index of the best pieces and the best positions to place it
    """
    options = []
    seen_pieces = {}
    for piece_index, piece in enumerate(queue):
        key = piece.to_byte()
        if key in seen_pieces: continue
        seen_pieces[key] = piece_index
        for coord in engine.check_positions(piece):
            score = engine.compute_dense_index(coord, piece) + piece.length()
            copy_engine = engine.__copy__()
            copy_engine.add(coord, piece)
            score += len(copy_engine.eliminate()) / engine._radius
            options.append((piece_index, coord, score))
    sorted_options = sorted(options, key=lambda item: item[2], reverse=True)
    return [(item[0], item[1]) for item in sorted_options[:significant_choices]]
