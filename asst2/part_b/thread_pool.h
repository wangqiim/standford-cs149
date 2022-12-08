#include <vector>
#include <map>
#include <queue>
#include <atomic>
#include <functional>
#include <condition_variable>
#include <queue>
#include <thread>
#include <mutex>
#include <cstdio>

class SleepThreadPool {
public:
    SleepThreadPool(int thread_num) {
        is_start_ = true;
        for (int i = 0; i < thread_num; i++) {
            ths_.emplace_back(std::thread(std::bind(&SleepThreadPool::worker, this)));
        }
    }

    ~SleepThreadPool() {
        is_start_ = false;
        cv_.notify_all();
        for (int i = 0; i < int(ths_.size()); i++) {
            ths_[i].join();
        }
    }

    /**
     * group_id: tag of task_group
     * job: work
    */
    void add_job(int group_id, std::function<void()> job) {
        std::unique_lock<std::mutex> guard(mtx_);
        jobs_.push({group_id, job});
        cv_.notify_one();
    }

    void worker() {
        // printf("[SleepThreadPool::worker] launch!\n");
        std::pair<int, std::function<void ()>> job_pair = {};
        while (is_start_) {
            {
                std::unique_lock<std::mutex> guard(mtx_);
                while (jobs_.empty() && is_start_) {
                    cv_.wait(guard);
                }
                if (!is_start_) {
                    return;
                }
                job_pair = jobs_.front();
                jobs_.pop();
            }
            job_pair.second();
            // printf("[SleepThreadPool::worker] finish a job\n");

            std::unique_lock<std::mutex> guard(cnt_mtx_);
            if (++finish_cnt_[job_pair.first] == need_finish_cnt_[job_pair.first]) {
                finish_queue_.insert(job_pair.first);
                un_finish_queue_.erase(job_pair.first);
                finish_cnt_.erase(job_pair.first);
                need_finish_cnt_.erase(job_pair.first);
                cnt_cv_.notify_one();
                // printf("[SleepThreadPool::worker] cnt_cv_.notify_one()\n");
            }

        }
    }
    
    void reset_cnt(int group_id, int job_cnt) {
        std::lock_guard<std::mutex> guard(cnt_mtx_);
        finish_cnt_[group_id] = 0;
        need_finish_cnt_[group_id] = job_cnt;
        un_finish_queue_.insert(group_id);
        // debug("reset_cnt");
    }

    std::vector<TaskID> wait_finish() {
        std::vector<TaskID> results;
        std::unique_lock<std::mutex> guard(cnt_mtx_);
        // debug("wait_finish");
        while (!un_finish_queue_.empty() || !finish_queue_.empty()) {
            while (finish_queue_.empty()) {
                cnt_cv_.wait(guard);
            }
            for (const auto &finish_id: finish_queue_) {
                results.push_back(finish_id);
            }
            for (const auto &finish_id: results) {
                finish_queue_.erase(finish_id);
            }
        }
        // debug("finish wait_finish");
        return results;
    }

    void debug(const char *msg) {
        printf("[thread_pool::debug] %s -------------------------------\n", msg);
        printf("unfinish group num = %lu, finish group num = %lu\n", un_finish_queue_.size(), finish_queue_.size());
        for (const auto &id: un_finish_queue_) {
            auto finish_cnt = finish_cnt_[id];
            auto need_finish_cnt = need_finish_cnt_[id];
            printf("group_id = %d, finish_cnt = %d, need_finish_cnt_ = %d\n", id, finish_cnt, need_finish_cnt);
        }
    }

private:
    std::atomic<bool> is_start_ = {false};
    std::vector<std::thread> ths_ = {};
    std::queue<std::pair<int, std::function<void()>>> jobs_ = {};
    std::mutex mtx_ = {};
    std::condition_variable cv_ = {};

    std::map<int, int> finish_cnt_ = {};
    std::map<int, int> need_finish_cnt_ = {};

    std::set<int> un_finish_queue_ = {};
    std::set<int> finish_queue_ = {};
    std::mutex cnt_mtx_ = {};
    std::condition_variable cnt_cv_ = {};
};
