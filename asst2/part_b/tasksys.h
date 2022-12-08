#ifndef _TASKSYS_H
#define _TASKSYS_H

#include <condition_variable>
#include <map>
#include <mutex>
#include <queue>
#include <set>
#include <vector>

#include "itasksys.h"
#include "thread_pool.h"

/*
 * TaskSystemSerial: This class is the student's implementation of a
 * serial task execution engine.  See definition of ITaskSystem in
 * itasksys.h for documentation of the ITaskSystem interface.
 */
class TaskSystemSerial: public ITaskSystem {
    public:
        TaskSystemSerial(int num_threads);
        ~TaskSystemSerial();
        const char* name();
        void run(IRunnable* runnable, int num_total_tasks);
        TaskID runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                const std::vector<TaskID>& deps);
        void sync();
};

/*
 * TaskSystemParallelSpawn: This class is the student's implementation of a
 * parallel task execution engine that spawns threads in every run()
 * call.  See definition of ITaskSystem in itasksys.h for documentation
 * of the ITaskSystem interface.
 */
class TaskSystemParallelSpawn: public ITaskSystem {
    public:
        TaskSystemParallelSpawn(int num_threads);
        ~TaskSystemParallelSpawn();
        const char* name();
        void run(IRunnable* runnable, int num_total_tasks);
        TaskID runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                const std::vector<TaskID>& deps);
        void sync();
};

/*
 * TaskSystemParallelThreadPoolSpinning: This class is the student's
 * implementation of a parallel task execution engine that uses a
 * thread pool. See definition of ITaskSystem in itasksys.h for
 * documentation of the ITaskSystem interface.
 */
class TaskSystemParallelThreadPoolSpinning: public ITaskSystem {
    public:
        TaskSystemParallelThreadPoolSpinning(int num_threads);
        ~TaskSystemParallelThreadPoolSpinning();
        const char* name();
        void run(IRunnable* runnable, int num_total_tasks);
        TaskID runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                const std::vector<TaskID>& deps);
        void sync();
};

/*
 * TaskSystemParallelThreadPoolSleeping: This class is the student's
 * optimized implementation of a parallel task execution engine that uses
 * a thread pool. See definition of ITaskSystem in
 * itasksys.h for documentation of the ITaskSystem interface.
 */
class TaskSystemParallelThreadPoolSleeping: public ITaskSystem {
    public:
        enum class TaskState: int8_t {
            UnKnown,
            Not_Ready,
            Ready,
            Running,
            Finish
        };

        class TaskWrapper {
        public:
            TaskState state_ = TaskState::UnKnown;
            IRunnable *runnable_ = nullptr;;
            int num_total_tasks_ = {0};
            std::vector<TaskID> deps_ = {};
            std::set<TaskID> unfinish_deps_ = {};
            TaskWrapper() = default;

            TaskWrapper(TaskState state, IRunnable *runnable, int num_total_tasks, const std::vector<TaskID>& deps): state_(state)
                , runnable_(runnable), num_total_tasks_(num_total_tasks), deps_(deps) {
                for (const auto &dep_id: deps) {
                    unfinish_deps_.insert(dep_id);
                }
            }
        };

        TaskSystemParallelThreadPoolSleeping(int num_threads);
        ~TaskSystemParallelThreadPoolSleeping();
        const char* name();
        void run(IRunnable* runnable, int num_total_tasks);
        TaskID runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                const std::vector<TaskID>& deps);
        void sync();
    private:
        // 调用该方法时需要已经持有锁
        bool try_run(TaskID taskid);

        SleepThreadPool pool_;
        int next_task_id_ = 0;
        std::map<TaskID, std::vector<TaskID>> wait_queue_ = {}; // wait_queue_[1] = {2, 3} means task 2, 3 depend on task 1.
        std::set<TaskID> ready_set_ = {};

        std::map<TaskID, TaskWrapper> states_ = {};

        std::mutex mtx_ = {};
        std::condition_variable cv_ = {};
};

#endif
