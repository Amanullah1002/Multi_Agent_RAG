from collections import defaultdict, deque


class DependencyResolver:
    """
    Mini Dependency Resolver using Kahn's Topological Sort.
    
    Features:
    - Handles missing root tasks
    - Detects cycles
    - Returns deterministic order
    """

    def __init__(self, dependencies):
        self.dependencies = dependencies
        self.graph = defaultdict(list)
        self.in_degree = defaultdict(int)
        self.tasks = set()

    
    # Build Graph
    
    def build_graph(self):
        for item in self.dependencies:
            task = item["task"]
            depends_on = item["depends_on"]

            # Track all tasks
            self.tasks.add(task)
            self.tasks.add(depends_on)

            # Build adjacency list
            self.graph[depends_on].append(task)

            # Update in-degree
            self.in_degree[task] += 1

            # Ensure root exists in in-degree map
            if depends_on not in self.in_degree:
                self.in_degree[depends_on] = 0

    
    # Topological Sort
    
    def resolve(self):
        self.build_graph()

        # Queue for tasks with no dependencies
        queue = deque(
            [task for task in self.tasks if self.in_degree[task] == 0]
        )

        execution_order = []

        while queue:
            current = queue.popleft()
            execution_order.append(current)

            for neighbor in self.graph[current]:
                self.in_degree[neighbor] -= 1

                if self.in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Cycle Detection
        if len(execution_order) != len(self.tasks):
            raise ValueError(
                "Cycle detected! No valid execution order exists."
            )

        return execution_order


if __name__ == "__main__":

    dependencies = [
        {"task": "B", "depends_on": "A"},
        {"task": "C", "depends_on": "B"},
        {"task": "D", "depends_on": "B"},
        {"task": "E", "depends_on": "C"}
    ]

    resolver = DependencyResolver(dependencies)

    order = resolver.resolve()

    print("Execution Order:")
    print(order)
