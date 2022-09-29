import argparse
from river_raid import RiverRaid

# Hyperparameters
NUM_FRAME = 1000000
#NUM_FRAME = 10000

parser = argparse.ArgumentParser(description="Train and test different networks on RiverRaid")

# Parse arguments
parser.add_argument("-n", "--network", type=str, action='store', help=" DQN or DDQN", required=True)
parser.add_argument("-m", "--mode", type=str, action='store', help="train or test", required=True)
parser.add_argument("-l", "--load", type=str, action='store', help="load weights", required=False)
parser.add_argument("-s", "--save", type=str, action='store', help="simulation of network in", required=False)
parser.add_argument("-x", "--statistics", action='store_true', help="Specify to calculate statistics", required=False)
parser.add_argument("-v", "--view", action='store_true', help="playing a game of RiverRaid. ", required=False)

args = parser.parse_args()
print(args)

game_instance = RiverRaid(args.network)

if args.load:
    game_instance.load_network(args.load)

if args.mode == "train":
    game_instance.train(NUM_FRAME)

if args.statistics:
    stat = game_instance.calculate_mean()
    print("Game Statistics")
    print(stat)

if args.save:
    game_instance.simulate(path=args.save, save=True)
elif args.view:
    game_instance.simulate()

