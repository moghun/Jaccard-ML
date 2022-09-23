#include "json.h"
#include "json.hpp"
#include <iostream>
#include <atomic>
#include <unordered_map>
#include <mutex>

class JSONStore{
private:
    static std::unordered_map<int, nlohmann::json> store_;
    static int ids_;
    static JSONStore * singleton_;
    static std::mutex mutex_;
    JSONStore(){
                store_.reserve(100);
                ids_ = 0;
        }
public:
    JSONStore* GetStore(){
        mutex_.lock();
        if (singleton_ == nullptr)
            singleton_ = new JSONStore;
        mutex_.unlock();
        return singleton_;
    }    
    nlohmann::json& AllocateJSONObject(){
        std::unique_lock<std::mutex> lock(mutex_);
        int new_id = ids_++;
        return store_[new_id];
    }
    nlohmann::json& GetJSONObject(int id){
        std::unique_lock<std::mutex> lock(mutex_);
        return store_[id];
    }
};
JSONStore* JSONStore::singleton_{nullptr};
