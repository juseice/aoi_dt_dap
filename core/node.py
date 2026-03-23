from core.basic_object import BasicObject


class Node(BasicObject):
    def __init__(self, id):
        super().__init__(id)

    def test_function(self):
        print(1)
        return "test function: Node"


class EdgeNode(Node):
    def __init__(self, id, compute_power, cost, memory):
        super().__init__(id)
        self.compute_power = compute_power
        self.cost = cost
        self.available_time = 0
        self.available_memory = memory

    def test_function(self):
        print(1)
        return "test function: EdgeNode"
