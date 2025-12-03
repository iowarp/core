@CLAUDE.md Let's add the concept of task graphs.


## Task Definition
We will add a new method to the admin chimod called ProcessTaskGraph.

```cpp
struct TaskNode {
  chi::ipc::vector<Pointer> tasks_;
};

struct TaskGraph {
  chi::ipc::vector<TaskNode> graph_;
}
```

A task graph is a chi::ipc::vector<TaskNode> graph_. Each TaskNode represents a batch
of tasks to execute independently.

```cpp
class ProcessTaskGraph : public Task {
  IN TaskGraph graph_;

  
}
```
