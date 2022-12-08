#include <cassert>
#include <functional>
#include "tasksys.h"


IRunnable::~IRunnable() {}

ITaskSystem::ITaskSystem(int num_threads) {}
ITaskSystem::~ITaskSystem() {}

/*
 * ================================================================
 * Serial task system implementation
 * ================================================================
 */

const char* TaskSystemSerial::name() {
    return "Serial";
}

TaskSystemSerial::TaskSystemSerial(int num_threads): ITaskSystem(num_threads) {
}

TaskSystemSerial::~TaskSystemSerial() {}

void TaskSystemSerial::run(IRunnable* runnable, int num_total_tasks) {
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }
}

TaskID TaskSystemSerial::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                          const std::vector<TaskID>& deps) {
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }

    return 0;
}

void TaskSystemSerial::sync() {
    return;
}

/*
 * ================================================================
 * Parallel Task System Implementation
 * ================================================================
 */

const char* TaskSystemParallelSpawn::name() {
    return "Parallel + Always Spawn";
}

TaskSystemParallelSpawn::TaskSystemParallelSpawn(int num_threads): ITaskSystem(num_threads) {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelSpawn in Part B.
}

TaskSystemParallelSpawn::~TaskSystemParallelSpawn() {}

void TaskSystemParallelSpawn::run(IRunnable* runnable, int num_total_tasks) {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelSpawn in Part B.
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }
}

TaskID TaskSystemParallelSpawn::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                 const std::vector<TaskID>& deps) {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelSpawn in Part B.
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }

    return 0;
}

void TaskSystemParallelSpawn::sync() {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelSpawn in Part B.
    return;
}

/*
 * ================================================================
 * Parallel Thread Pool Spinning Task System Implementation
 * ================================================================
 */

const char* TaskSystemParallelThreadPoolSpinning::name() {
    return "Parallel + Thread Pool + Spin";
}

TaskSystemParallelThreadPoolSpinning::TaskSystemParallelThreadPoolSpinning(int num_threads): ITaskSystem(num_threads) {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelSpawn in Part B.
}

TaskSystemParallelThreadPoolSpinning::~TaskSystemParallelThreadPoolSpinning() {}

void TaskSystemParallelThreadPoolSpinning::run(IRunnable* runnable, int num_total_tasks) {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelSpawn in Part B.
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }
}

TaskID TaskSystemParallelThreadPoolSpinning::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                              const std::vector<TaskID>& deps) {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelSpawn in Part B.
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }

    return 0;
}

void TaskSystemParallelThreadPoolSpinning::sync() {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelSpawn in Part B.
    return;
}

/*
 * ================================================================
 * Parallel Thread Pool Sleeping Task System Implementation
 * ================================================================
 */

const char* TaskSystemParallelThreadPoolSleeping::name() {
    return "Parallel + Thread Pool + Sleep";
}

TaskSystemParallelThreadPoolSleeping::TaskSystemParallelThreadPoolSleeping(int num_threads)
    : ITaskSystem(num_threads), pool_(num_threads) {
    //
    // TODO: CS149 student implementations may decide to perform setup
    // operations (such as thread pool construction) here.
    // Implementations are free to add new class member variables
    // (requiring changes to tasksys.h).
    //
    // printf("[TaskSystemParallelThreadPoolSleeping] construct done!\n");
}

TaskSystemParallelThreadPoolSleeping::~TaskSystemParallelThreadPoolSleeping() {
    //
    // TODO: CS149 student implementations may decide to perform cleanup
    // operations (such as thread pool shutdown construction) here.
    // Implementations are free to add new class member variables
    // (requiring changes to tasksys.h).
    //
}

void TaskSystemParallelThreadPoolSleeping::run(IRunnable* runnable, int num_total_tasks) {
    //
    // TODO: CS149 students will modify the implementation of this
    // method in Parts A and B.  The implementation provided below runs all
    // tasks sequentially on the calling thread.
    //
    std::vector<TaskID> no_deps;
    runAsyncWithDeps(runnable, num_total_tasks, no_deps);
    sync();
}

// 调用该方法时需要已经持有锁
bool TaskSystemParallelThreadPoolSleeping::try_run(TaskID taskid) {
    if (states_.count(taskid) == 0 || states_[taskid].state_ != TaskState::Ready) {
        return false;
    }
    assert(states_[taskid].num_total_tasks_ != 0);
    pool_.reset_cnt(int(taskid), states_[taskid].num_total_tasks_);
    for (int i = 0; i < states_[taskid].num_total_tasks_; i++) {
        using namespace std::placeholders;
        auto job = std::bind(&IRunnable::runTask, states_[taskid].runnable_, i, states_[taskid].num_total_tasks_);
        pool_.add_job(int(taskid), job);
    }
    states_[taskid].state_ = TaskState::Running;
    return true;
}

TaskID TaskSystemParallelThreadPoolSleeping::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                    const std::vector<TaskID>& deps) {
    //
    // TODO: CS149 students will implement this method in Part B.
    //
    std::lock_guard<std::mutex> guard(mtx_);
    TaskID task_id = next_task_id_++;
    states_.insert({task_id, TaskWrapper(TaskState::Not_Ready, runnable, num_total_tasks, deps)});
    
    bool this_task_ready = true;
    if (!deps.empty()) {
        for (const auto &dep_id: deps) {
            if (states_.count(dep_id) != 0 && states_[dep_id].state_ != TaskState::Finish) {
                this_task_ready = false;
                wait_queue_[dep_id].push_back(task_id);
            }
        }
    }
    if (this_task_ready) {
        // printf("[TaskSystemParallelThreadPoolSleeping::runAsyncWithDeps] %d is ready to run!\n", task_id);
        states_[task_id].state_ = TaskState::Ready;
        ready_set_.insert(task_id);
        assert(try_run(task_id));
    }
    
    return task_id;
}

void TaskSystemParallelThreadPoolSleeping::sync() {
    //
    // TODO: CS149 students will modify the implementation of this method in Part B.
    //
    mtx_.lock();
    while (!ready_set_.empty()) {
        for (const auto &ready_id: ready_set_) {
            if (states_[ready_id].state_ == TaskState::Running) {
                continue;
            }
            assert(try_run(ready_id));
        }
        // block here
        mtx_.unlock();
        std::vector<TaskID> finish_ids = pool_.wait_finish();
        mtx_.lock();
        for (const auto &finish_id: finish_ids) {
            states_[finish_id].state_ = TaskState::Finish;
            ready_set_.erase(finish_id);

            for (const auto &wait_id: wait_queue_[finish_id]) {
                states_[wait_id].unfinish_deps_.erase(finish_id);
                if (states_[wait_id].unfinish_deps_.empty()) {
                    ready_set_.insert(wait_id);
                    states_[wait_id].state_ = TaskState::Ready;
                }
            }
            wait_queue_.erase(finish_id);
        }
    }
    return;
}
