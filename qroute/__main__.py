import os
import logging

# import wandb
import torch

from .environment.device import IBMqx20TokyoDevice
from .models.graph_dual import GraphDualModel
from .algorithms.deepmcts import MCTSAgent
from .memory.list import MemorySimple
from .environment.circuits import circuit_from_qasm, CircuitRepDQN
from .engine import train_step
from .visualizers.greedy_schedulers import cirq_routing

logging.basicConfig(level=logging.DEBUG)

# os.system("wandb login d43f6dc5f4f9981ac8b6bffd1ab5db7d9ac45480")
# wandb.init(project='qroute-rl', name='dqn-basic-1', save_code=False)


if __name__ == '__main__':
    device = IBMqx20TokyoDevice()
    model = GraphDualModel(device, True)
    agent = MCTSAgent(model, device)
    memory = MemorySimple(500)

    if os.path.exists("model-weights.h5"):
        model.load_state_dict(torch.load("model-weights.h5"))

    small_files = ["graycode6_47", "xor5_254", "ex1_226", "ham3_102", "mod5d1_63",
                   "mod5mils_65", "alu-v0_27"]
    large_files = ["rd84_142", "adr4_197", "radd_250", "z4_268", "sym6_145", "misex1_241", "rd73_252",
                   "cycle10_2_110", "square_root_7", "sqn_258", "rd84_253", "rd84_142"]
    for e, file in enumerate(large_files):
        cirq = circuit_from_qasm(
            os.path.join("./test/circuit_qasm", file + "_onlyCX.qasm"))
        circuit = CircuitRepDQN(cirq, len(device))
        train_step(agent, device, circuit, memory, episode_id=file)
        print("Cirq Routing Distance: ",
              len(cirq_routing(circuit, device).circuit.moments))
