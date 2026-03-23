from core.basic_object import BasicObject


class Edge(BasicObject):
    def __init__(self, id, source_node, terminal_node, bandwidth):
        super().__init__(id)
        self.source_node = source_node
        self.terminal_node = terminal_node
        self.bandwidth = bandwidth

    def test_function(self):
        print(1)
        return "test function: Edge"