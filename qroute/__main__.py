import os
import logging
import argparse

import wandb
import torch

from .environment.device import IBMqx20TokyoDevice, GridComputerDevice
from .models.graph_dual import GraphDualModel
from .algorithms.deepmcts import MCTSAgent
from .memory.list import MemorySimple
from .environment.circuits import circuit_from_qasm, CircuitRepDQN, \
    circuit_generated_randomly, circuit_generated_full_layer
from .engine import train_step
from .visualizers.greedy_schedulers import cirq_routing, qiskit_routing, tket_routing

logging.basicConfig(level=logging.DEBUG)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataset', default="small",
                        help='Choose training and test dataset from small, large, full, random')
    parser.add_argument('--gates', default=100, type=int,
                        help='Size of circuit if not from a file dataset')
    parser.add_argument('--hardware', default="qx20",
                        help='Device to run on, eg. qx20, grid/6, grid/4, etc.')
    parser.add_argument('--iterations', default=10, type=int,
                        help='Number of iterations to train for on generated circuits.')
    parser.add_argument('--train', action='store_const', default=False, const=True,
                        help='Whether the training loop should be run or just evaluation.')
    parser.add_argument('--wandb', action='store_const', default=False, const=True,
                        help='Whether to use WandB to log the results of experiments.')
    parser.add_argument('--search', default=100, type=int,
                        help='Number of iterations to search for before making a move.')
    args = parser.parse_args()

    device = IBMqx20TokyoDevice() if args.hardware == "qx20" else \
        GridComputerDevice(int(args.hardware.split("/")[-1]))
    model = GraphDualModel(device, True)
    memory = MemorySimple(0)
    agent = MCTSAgent(model, device, memory, search_depth=args.search)

    if args.wandb:
        os.system("wandb login d43f6dc5f4f9981ac8b6bffd1ab5db7d9ac45480")
        wandb.init(project='qroute-rl', name='mcts-small-qx20-1', save_code=False)

    if os.path.exists("model-weights.h5"):
        model.load_state_dict(torch.load("model-weights.h5"))

    if args.dataset == "small":
        small_files = ["graycode6_47", "xor5_254", "ex1_226", "ham3_102", "mod5d1_63",
                       "mod5mils_65", "alu-v0_27"]
        for e, file in enumerate(small_files):
            cirq = circuit_from_qasm(
                os.path.join("./test/circuit_qasm", file + "_onlyCX.qasm"))
            circuit = CircuitRepDQN(cirq, len(device))
            train_step(agent, device, circuit, episode_name=file, use_wandb=args.wandb, train_model=args.train)
            print("Cirq Routing Distance: ", cirq_routing(circuit, device))
            print("Qiskit Routing Distance: ", qiskit_routing(circuit, device))
            print("PyTket Routing Distance: ", tket_routing(circuit, device))
    elif args.dataset == "large":
        large_files = ["rd84_142", "adr4_197", "radd_250", "z4_268", "sym6_145", "misex1_241",
                       "rd73_252", "cycle10_2_110", "square_root_7", "sqn_258", "rd84_253"]
        for e, file in enumerate(large_files):
            cirq = circuit_from_qasm(
                os.path.join("./test/circuit_qasm", file + "_onlyCX.qasm"))
            circuit = CircuitRepDQN(cirq, len(device))
            train_step(agent, device, circuit, episode_name=file, use_wandb=args.wandb, train_model=args.train)
            print("Qiskit Routing Distance: ", qiskit_routing(circuit, device))
            print("PyTket Routing Distance: ", tket_routing(circuit, device))
    elif args.dataset == "random":
        for e in range(args.iterations):
            cirq = circuit_generated_randomly(len(device), args.gates)
            circuit = CircuitRepDQN(cirq, len(device))
            print("Layers in input circuit: ", len(cirq.moments))
            train_step(agent, device, circuit, episode_name=f"random_{e}", use_wandb=args.wandb, train_model=args.train)
            print("Cirq Routing Distance: ", cirq_routing(circuit, device))
            print("Qiskit Routing Distance: ", qiskit_routing(circuit, device))
            print("PyTket Routing Distance: ", tket_routing(circuit, device))
    elif args.dataset == "full":
        for e in range(args.iterations):
            cirq = circuit_generated_full_layer(len(device), args.gates)
            circuit = CircuitRepDQN(cirq, len(device))
            train_step(agent, device, circuit, episode_name=f"full_{e}", use_wandb=args.wandb, train_model=args.train)
