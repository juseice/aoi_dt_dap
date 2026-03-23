from core.node import Node


class UserNode(Node):
    def __init__(self, id):
        super().__init__(id)

    def test_function(self):
        print(1)
        return "test function: User"
