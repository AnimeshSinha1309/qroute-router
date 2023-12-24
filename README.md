# quantum-rl
Attempts at using RL in optimizations on Quantum Algorithms

## How to Run

Following is an example command, you can tune all the arguments to choose whatever benchmark, harware, search depth, etc. is needed. Use --train to train the model, don't if you want to just compile your circuits.
```shell
python -m qroute --dataset small --hardware qx20 --search 200 --train
```

## Code Structure

The code in the qroute library looks as follows:
* algorithms - The combining methods, MCTS, Annealers, etc.
* models - The Neural networks that evaluate a state and the actions from that state, edit for new architectures.
* environment - Basic code managing the circuit and updates to the state, can be edited to add state representations.
* memory - Storing data for replay, linearly, or with Prioritized Experience Replay.
* visualizers - Checking if the result is correct, making videos and plots.

The rest of the package contains code for the paper, plotting stuff, illustrator files, etc.

## Explaination

Following is demonstrative video:
![Routing Video](paper/video/example_routing_video.gif)

A presentation with details of the routing process are found here:
https://docs.google.com/presentation/d/1Q-Y84ltoNbW15tKF_Nuh4MOcfkwyFIr6loAM7aGyJHk/edit?usp=sharing
