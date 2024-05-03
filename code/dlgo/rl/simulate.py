from dlgo import rl
from dlgo import scoring
from dlgo import goboard_fast as goboard
from dlgo.gotypes import Player

from collections import namedtuple
import datetime


class GameRecord(namedtuple('GameRecord', 'moves winner margin')):
    pass


def simulate_game(black_player, white_player):
    moves = []
    game = goboard.GameState.new_game(19)
    agents = {
        Player.black: black_player,
        Player.white: white_player,
    }
    while not game.is_over():
        #print("before select_move")
        next_move = agents[game.next_player].select_move(game)
        #print("after select_move ")
        moves.append(next_move)
        game = game.apply_move(next_move)

    game_result = scoring.compute_game_result(game)
    print(f"结果：{game_result}，总步数：{len(moves)}")

    return GameRecord(
        moves=moves,
        winner=game_result.winner,
        margin=game_result.winning_margin,
    )


def experience_simulation(num_games, agent1, agent2):
    collector1 = rl.ExperienceCollector()
    collector2 = rl.ExperienceCollector()

    color1 = Player.black
    for i in range(num_games):
        print(f"{i}begin，{datetime.datetime.now()}")
        collector1.begin_episode()
        agent1.set_collector(collector1)
        collector2.begin_episode()
        agent2.set_collector(collector2)

        if color1 == Player.black:
            black_player, white_player = agent1, agent2
        else:
            white_player, black_player = agent2, agent1
        game_record = simulate_game(black_player, white_player)
        if game_record.winner == color1:
            collector1.complete_episode(reward=1)
            collector2.complete_episode(reward=-1)
        else:
            collector2.complete_episode(reward=1)
            collector1.complete_episode(reward=-1)
        color1 = color1.other

    return rl.combine_experience([collector1, collector2])
