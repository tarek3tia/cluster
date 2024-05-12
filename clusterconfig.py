import os
from abc import ABC

from pytorch_lightning.plugins.environments import ClusterEnvironment


class WakebClusterEnvironment(ClusterEnvironment, ABC):
    def __init__(self, world_size=2, master_node='192.168.126.228', master_port=8080, node_rank=0, device_rank=0,):
        self.world_size = world_size
        self.master_node = master_node
        self.master_port = master_port
        self.node_rank = node_rank
        self.device_rank = device_rank
        super().__init__()

    @property
    def creates_processes_externally(self) -> bool:
        """Return True if the cluster is managed (you don't launch processes yourself)"""
        return True

    def world_size(self) -> int:
        return self.world_size

    def global_rank(self) -> int:
        return self.node_rank

    def local_rank(self) -> int:
        return self.device_rank

    def node_rank(self) -> int:
        return self.node_rank

    def main_address(self) -> str:
        return self.master_node

    def main_port(self) -> int:
        return self.master_port
