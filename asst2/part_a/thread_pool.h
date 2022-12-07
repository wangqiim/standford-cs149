#include <vector>
#include <atomic>
#include <functional>
#include <condition_variable>
#include <queue>
#include <thread>
#include <mutex>
#include <cstdio>

class SpinThreadPool {
public:
    SpinThreadPool(int thread_num) {
        is_start_ = true;
        for (int i = 0; i < thread_num; i++) {
            ths_.emplace_back(std::thread(std::bind(&SpinThreadPool::worker, this)));
        }
    }

    ~SpinThreadPool() {
        is_start_ = false;
        for (int i = 0; i < int(ths_.size()); i++) {
            ths_[i].join();
        }
    }

    void add_job(std::function<void()> job) {
        std::lock_guard<std::mutex> guard(mtx_);
        jobs_.push(job);
    }

    void worker() {
        while (is_start_) {
            mtx_.lock();
            if (jobs_.empty()) {
                mtx_.unlock();
                continue;
            }
            auto job = jobs_.front();
            jobs_.pop();
            mtx_.unlock();
            job();
            finish_cnt_++;
        }
    }
    
    void reset_cnt() { finish_cnt_ = 0; }
    int finish_cnt() { return finish_cnt_; }
private:
    std::atomic<bool> is_start_ = {false};
    std::vector<std::thread> ths_ = {};
    std::queue<std::function<void()>> jobs_ = {};
    std::mutex mtx_ = {};

    std::atomic<int> finish_cnt_ = {0};
};

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

    void add_job(std::function<void()> job) {
        std::unique_lock<std::mutex> guard(mtx_);
        jobs_.push(job);
        cv_.notify_one();
    }

    void worker() {
        // printf("[SleepThreadPool::worker] launch!\n");
        std::function<void ()> job = {};
        while (is_start_) {
            {
                std::unique_lock<std::mutex> guard(mtx_);
                while (jobs_.empty() && is_start_) {
                    cv_.wait(guard);
                }
                if (!is_start_) {
                    return;
                }
                job = jobs_.front();
                jobs_.pop();
            }
            job();
            // printf("[SleepThreadPool::worker] finish a job\n");

            std::unique_lock<std::mutex> guard(cnt_mtx_);
            if (++finish_cnt_ == need_finish_cnt_) {
                cnt_cv_.notify_one();
                // printf("[SleepThreadPool::worker] cnt_cv_.notify_one()\n");
            }

        }
    }
    
    void reset_cnt(int job_cnt) {
        std::lock_guard<std::mutex> guard(cnt_mtx_);
        finish_cnt_ = 0;
        need_finish_cnt_ = job_cnt;
    }

    void wait_finish( ) {
        std::unique_lock<std::mutex> guard(cnt_mtx_);
        while (finish_cnt_ != need_finish_cnt_) {
            // printf("[SleepThreadPool::wait_finish] fall asleep\n");
            cnt_cv_.wait(guard);
            // printf("[SleepThreadPool::wait_finish] awake finish_cnt_ = %d, "
            //     "need_finish_cnt_ = %d\n", finish_cnt_, need_finish_cnt_);
        }
    }

private:
    std::atomic<bool> is_start_ = {false};
    std::vector<std::thread> ths_ = {};
    std::queue<std::function<void()>> jobs_ = {};
    std::mutex mtx_ = {};
    std::condition_variable cv_ = {};

    int finish_cnt_ = {0};
    int need_finish_cnt_ = {0};
    std::mutex cnt_mtx_ = {};
    std::condition_variable cnt_cv_ = {};
};
