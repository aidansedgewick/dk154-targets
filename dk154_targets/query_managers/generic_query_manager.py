import abc

class GenericQueryManager(abc.ABC):

    @property
    @abc.abstractmethod
    def name(self,):
        pass


    @abc.abstractmethod
    def perform_all_tasks(self,):
        pass