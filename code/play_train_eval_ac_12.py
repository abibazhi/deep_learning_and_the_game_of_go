import argparse
import datetime
import multiprocessing
import os
import random
import shutil
import time
import tempfile
from collections import namedtuple

import h5py
import numpy as np

from dlgo import kerasutil
from dlgo import scoring
from dlgo import rl
from dlgo.goboard_fast import GameState, Player, Point


def load_agent(filename):
    with h5py.File(filename, 'r') as h5file:
        return rl.load_ac_agent(h5file)


COLS = 'ABCDEFGHJKLMNOPQRST'
STONE_TO_CHAR = {
    None: '.',
    Player.black: 'x',
    Player.white: 'o',
}


def avg(items):
    if not items:
        return 0.0
    return sum(items) / float(len(items))


def print_board(board):
    for row in range(board.num_rows, 0, -1):
        line = []
        for col in range(1, board.num_cols + 1):
            stone = board.get(Point(row=row, col=col))
            line.append(STONE_TO_CHAR[stone])
        print('%2d %s' % (row, ''.join(line)))
    print('   ' + COLS[:board.num_cols])


class GameRecord(namedtuple('GameRecord', 'moves winner margin')):
    pass


def name(player):
    if player == Player.black:
        return 'B'
    return 'W'


def simulate_game(black_player, white_player, board_size):
    moves = []
    game = GameState.new_game(board_size)
    agents = {
        Player.black: black_player,
        Player.white: white_player,
    }
    num_moves = 0
    while not game.is_over():
        if num_moves < 16:
            # Pick randomly.
            agents[game.next_player].set_temperature(1.0)
        else:
            # Favor the best-looking move.
            agents[game.next_player].set_temperature(0.05) # 0.05
        next_move = agents[game.next_player].select_move(game)
        moves.append(next_move)
        game = game.apply_move(next_move)
        num_moves += 1

    #print_board(game.board)
    game_result = scoring.compute_game_result(game)
    print(game_result)

    return GameRecord(
        moves=moves,
        winner=game_result.winner,
        margin=game_result.winning_margin,
    )


def get_temp_file():
    fd, fname = tempfile.mkstemp(prefix='dlgo-train')
    os.close(fd)
    return fname


def do_self_play(board_size, agent1_filename, agent2_filename,
                 num_games,
                 experience_filename,
                 gpu_frac):
    kerasutil.set_gpu_memory_target(gpu_frac)

    random.seed(int(time.time()) + os.getpid())
    np.random.seed(int(time.time()) + os.getpid())

    agent1 = load_agent(agent1_filename)
    agent2 = load_agent(agent2_filename)

    collector1 = rl.ExperienceCollector()

    color1 = Player.black
    for i in range(num_games):
        print('Simulating game %d/%d...' % (i + 1, num_games))
        collector1.begin_episode()
        agent1.set_collector(collector1)

        if color1 == Player.black:
            black_player, white_player = agent1, agent2
        else:
            white_player, black_player = agent1, agent2
        game_record = simulate_game(black_player, white_player, board_size)
        if game_record.winner == color1:
            print('Agent 1 wins.')
            collector1.complete_episode(reward=1)
        else:
            print('Agent 2 wins.')
            collector1.complete_episode(reward=-1)
        color1 = color1.other

    experience = rl.combine_experience([collector1])
    print('Saving experience buffer to %s\n' % experience_filename)
    with h5py.File(experience_filename, 'w') as experience_outf:
        experience.serialize(experience_outf)


def generate_experience(learning_agent, reference_agent, exp_file,
                        num_games, board_size, num_workers):
    experience_files = []
    workers = []
    gpu_frac = 0.95 / float(num_workers)
    games_per_worker = num_games // num_workers
    for i in range(num_workers):
        filename = get_temp_file()
        experience_files.append(filename)
        worker = multiprocessing.Process(
            target=do_self_play,
            args=(
                board_size,
                learning_agent,
                reference_agent,
                games_per_worker,
                filename,
                gpu_frac,
            )
        )
        worker.start()
        workers.append(worker)

    # Wait for all workers to finish.
    print('Waiting for workers...')
    for worker in workers:
        worker.join()

    # Merge experience buffers.
    print('Merging experience buffers...')
    first_filename = experience_files[0]
    other_filenames = experience_files[1:]
    with h5py.File(first_filename, 'r') as expf:
        combined_buffer = rl.load_experience(expf)
    for filename in other_filenames:
        with h5py.File(filename, 'r') as expf:
            next_buffer = rl.load_experience(expf)
        combined_buffer = rl.combine_experience([combined_buffer, next_buffer])
    print('Saving into %s...' % exp_file)
    with h5py.File(exp_file, 'w') as experience_outf:
        combined_buffer.serialize(experience_outf)

    # Clean up.
    for fname in experience_files:
        os.unlink(fname)


def train_worker(learning_agent, output_file, experience_file,
                 lr, batch_size):
    learning_agent = load_agent(learning_agent)
    with h5py.File(experience_file, 'r') as expf:
        exp_buffer = rl.load_experience(expf)
    learning_agent.train(exp_buffer, lr=lr, batch_size=batch_size)

    with h5py.File(output_file, 'w') as updated_agent_outf:
        learning_agent.serialize(updated_agent_outf)


def train_on_experience(learning_agent, output_file, experience_file,
                        lr, batch_size):
    # Do the training in the background process. Otherwise some Keras
    # stuff gets initialized in the parent, and later that forks, and
    # that messes with the workers.
    worker = multiprocessing.Process(
        target=train_worker,
        args=(
            learning_agent,
            output_file,
            experience_file,
            lr,
            batch_size
        )
    )
    worker.start()
    worker.join()


def play_games(args):
    agent1_fname, agent2_fname, num_games, board_size, gpu_frac = args

    kerasutil.set_gpu_memory_target(gpu_frac)

    random.seed(int(time.time()) + os.getpid())
    np.random.seed(int(time.time()) + os.getpid())

    agent1 = load_agent(agent1_fname)
    agent2 = load_agent(agent2_fname)

    wins, losses = 0, 0
    color1 = Player.black
    for i in range(num_games):
        print('Simulating game %d/%d...' % (i + 1, num_games))
        if color1 == Player.black:
            black_player, white_player = agent1, agent2
        else:
            white_player, black_player = agent1, agent2
        game_record = simulate_game(black_player, white_player, board_size)
        if game_record.winner == color1:
            print('Agent 1 wins')
            wins += 1
        else:
            print('Agent 2 wins')
            losses += 1
        print('Agent 1 record: %d/%d' % (wins, wins + losses))
        color1 = color1.other
    return wins, losses


def evaluate(learning_agent, reference_agent,
             num_games, num_workers, board_size):
    games_per_worker = num_games // num_workers
    gpu_frac = 0.95 / float(num_workers)
    pool = multiprocessing.Pool(num_workers)
    worker_args = [
        (
            learning_agent, reference_agent,
            games_per_worker, board_size, gpu_frac,
        )
        for _ in range(num_workers)
    ]
    game_results = pool.map(play_games, worker_args)

    total_wins, total_losses = 0, 0
    for wins, losses in game_results:
        total_wins += wins
        total_losses += losses
    print('FINAL RESULTS:')
    print('Learner: %d' % total_wins)
    print('Refrnce: %d' % total_losses)
    pool.close()
    pool.join()
    return total_wins


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', required=True)
    parser.add_argument('--games-per-batch', '-g', type=int, default=500)
    parser.add_argument('--work-dir', '-d')
    parser.add_argument('--num-workers', '-w', type=int, default=4)
    parser.add_argument('--board-size', '-b', type=int, default=11)
    parser.add_argument('--lr', type=float, default=0.007)
    parser.add_argument('--bs', type=int, default=512)
    parser.add_argument('--log-file', '-l')

    args = parser.parse_args()

    logf = open(args.log_file, 'a')
    logf.write('\n')
    logf.write('----------------------\n')
    logf.write('---------开始了--------\n')
    logf.write('----------------------\n')

    logf.write('Starting from %s at %s\n' % (
        args.agent, datetime.datetime.now()))

    learning_agent = args.agent
    reference_agent = args.agent
    experience_file = os.path.join(args.work_dir, 'exp_temp50.hdf5')
    tmp_agent = os.path.join(args.work_dir, 'agent_temp50.hdf5')
    working_agent = os.path.join(args.work_dir, 'agent_cur50.hdf5')
    total_games = 0

    #his_wons = np.array([])
    his_wons = [] # 开始是空数据太好。如果第一个合格的是35。有点大。所以加个32。这样只要33就可以合格，门槛第一点。
    avg_move_wons = 0 #初始化平均值，只有大于这个平均值的，才才采纳为继续训练。
    #avg_wons = np.mean(his_wons) #这是算法，实际等有了数据再算


    while True:
        print('Reference: %s' % (reference_agent,))
        logf.write('Total games so far %d\n' % (total_games,))
        generate_experience(
            learning_agent, reference_agent,
            experience_file,
            num_games=args.games_per_batch,
            board_size=args.board_size,
            num_workers=args.num_workers)
        train_on_experience(
            learning_agent, tmp_agent, experience_file,
            lr=args.lr, batch_size=args.bs)
        total_games += args.games_per_batch
        wins = evaluate(
            learning_agent, reference_agent,
            num_games=100,
            num_workers=args.num_workers,
            board_size=args.board_size)
        print('Won %d / 100 games (%.3f)' % (
            wins, float(wins) / 100.0))
        logf.write('Won %d / 100 games (%.3f)\n' % (
            wins, float(wins) / 100.0))
        shutil.copy(tmp_agent, working_agent)

        # 只有大于过去的平均成绩，才能有效升级
        # 但也不能大太多，太多是随机过头了。可能。需要抑制随机性。
        #his_wons.append(wins)
        #print(his_wons)
        #if len(his_wons) > 15:
        #    avg_move_wons = sum(his_wons[-15:]) / 15
        #else:
        #    avg_move_wons = sum(his_wons) / len(his_wons)
        
        #logf.write("当前轮胜负情况\n")
        #logf.write(f"wins={wins},avg_move_wons={avg_move_wons},总轮次:{len(his_wons)}\n\n")

        
        #if wins > 62: # and wins <= (avg_move_wons + 3):
        #    #print(f"当前平均胜局数为{avg_move_wons}")
        #    logf.write("升级！\n")
        learning_agent = working_agent
        if wins >= 62:
            next_filename = os.path.join(
                args.work_dir,
                'agent50_%08d.hdf5' % (total_games,))
            shutil.move(tmp_agent, next_filename)
            reference_agent = next_filename
            logf.write('New reference is %s\n' % next_filename)
            
            # 统计数据清零。重新初始化。
            his_wons = []
            avg_move_wons = 0
        else:
            print('Keep learning\n')
        logf.flush()


if __name__ == '__main__':
    main()
