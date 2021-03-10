import abc


class ReplayMemory(abc.ABC):
    """
    Memory object to keep track of the entire history as a replay buffer, then
    read it and learn from it.
    """

    @abc.abstractmethod
    def store(self, *args):
        """
        Stores a MemoryItem in the data structure
        :param args: the items to store
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

    @abc.abstractmethod
    def clear(self):
        """
        Clears out the memory
        """


class CombinerAgent(abc.ABC):
    """
    Class to train and act using the model, by combining the actions taken by the model in a single step
    """

    def __init__(self, model, device):
        """
        Sets up the state and the model for the agent
        :param model: The neural network for deciding each step
        :param device: The device on which to compile
        """
        self.model = model
        self.device = device

    @abc.abstractmethod
    def act(self, state):
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
