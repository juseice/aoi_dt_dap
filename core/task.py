from core.basic_object import BasicObject


class Task(BasicObject):
    def __init__(self, id, workload, deployment_cost, memory_requirement, output_data_size=0.0):
        super().__init__(id)
        self.workload = workload  # (FLOPs)
        self.deployment_cost = deployment_cost
        self.memory_requirement = memory_requirement

        self.output_data_size = output_data_size

    def test_function(self):
        print(1)
        return "test: Service"


class TaskChain(BasicObject):
    def __init__(self, id, tasks, sensor_id, required_bandwidth=2.0):
        super().__init__(id)
        self.tasks = tasks  # list of Task
        self.sensor_id = sensor_id
        # TODO: update required_bandwidth if placement modified
        self.required_bandwidth = required_bandwidth
