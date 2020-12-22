import abc
import collections
import qroute


MemoryItem = collections.namedtuple('MemoryItem', ['state', 'reward', 'action', 'next_state', 'done'])


class CombinerAgent(abc.ABC):
    """
    Class to train and act using the model, by combining the actions taken by the model in a single step
    """

    @abc.abstractmethod
    def act(self, state: qroute.environment.state.CircuitStateDQN):
        """
        Chooses an action to perform in the environment and returns it
        (i.e. does not alter environment state)
        :param state: the state of the environment
        :return: np.array of shape (len(device),), the chosen action mask after annealing
        """

    @abc.abstractmethod
    def replay(self):
        """
        Learns from past experiences
        """


class TransformationState(abc.ABC):
    """
    Represents the State of the system when transforming a circuit. This holds the reference
    copy of the environment and the state of the transformation (even within a step).
    """

    @abc.abstractmethod
    def execute_swap(self, solution):
        """
        Updates the state of the system with whatever swaps are executed in the solution.
        This function MUTATES the state.
        :param solution: boolean np.array, whether to take each edge on the device
        :return list of pairs, pairs of nodes representing gates which will be executed
        """

    @abc.abstractmethod
    def execute_cnot(self):
        """
        Updates the state of the system with whatever interactions can be executed on the hardware.
        This function MUTATES the state.
        :return list of pairs, pairs of nodes representing gates which will be executed
        """

    @abc.abstractmethod
    def is_done(self):
        """
        Returns True iff each qubit has completed all of its interactions
        :return: bool, True if the entire circuit is executed
        """

    # Other utility functions and properties

    @abc.abstractmethod
    def __copy__(self):
        """
        Makes a copy, keeping the reference to the same environment, but
        instantiating the rest of the state again.

        :return: State, a copy of the original, but independent of the first one, except env
        """

    @abc.abstractmethod
    def __eq__(self, other):
        """
        Checks whether two state are identical

        :param other: State, the other state to compare against
        :return: True if they are the same, False otherwise
        """


class ReplayMemory(abc.ABC):
    """
    Memory object to keep track of the entire history as a replay buffer, then
    read it and learn from it.
    """

    @abc.abstractmethod
    def store(self, experience):
        """
        Stores a MemoryItem in the data structure
        :param experience: the item to store
        """

    @abc.abstractmethod
    def __getitem__(self, item):
        """
        Returns the i-th item in memory
        :param item: index of item to get
        :return: the MemoryItem
        """

    @abc.abstractmethod
    def sample(self, batch_size):
        """
        Gets a random sample from the memory
        :param batch_size: number of samples to get
        :return: MemoryItem, or list of them of len batch_size
        """

    @abc.abstractmethod
    def __iter__(self):
        """
        Returns an iterator over the items in memory
        :return: the iterator
        """
