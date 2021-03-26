# quantum-rl
Attempts at using RL in optimizations on Quantum Algorithms

## How to Run

Following is an example command, you can tune all the arguments to choose whatever benchmark, harware, search depth, etc. is needed. Use --train to train the model, don't if you want to just compile your circuits.
```shell
python -m qroute --dataset small --hardware qx20 --search 200 --train
```

## Code Structure

* algorithms - The combining methods, MCTS, Annealers, etc.
* models - The Neural networks that evaluate a state
* 
* memory - Storing data, linearly, or with Prioritized Experience Replay

