from core.node import Node


class Sensor(Node):
    def __init__(self, id, data_size, period=1.0):
        super().__init__(id)
        self.data_size = data_size
        self.period = period
        self.last_generation_time = 0

    def test_function(self):
        print(1)
        return "test function: Sensor"

    def set_id(self, id):
        self.id = id

    def get_id(self):
        return self.id

    # def set_node_id(self, node_id):
    #     self.node_id = node_id

    def get_node_id(self):
        return self.node_id

    def generate_update(self, current_time):
        self.last_generation_time = current_time
        return current_time
