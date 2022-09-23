#include "json.h"
#include "json.hpp"
#include <iostream>
#include <atomic>
#include <unordered_map>
#include <mutex>

JSONWrapper JSONWrapper::GetJSON(std::string key){
    JSONWrapper newone;
    newone.object = object[key];
    return newone;
}
void JSONWrapper::SetJSON(std::string key, JSONWrapper value){
    object[key] = value.object;
}
void JSONWrapper::SetJSONNested(std::string key1, std::string key2, JSONWrapper value){
    object[key1][key2] = value.object;
}
void JSONWrapper::SetJSONNested(std::string key1, std::string key2, std::string key3, JSONWrapper value){
    object[key1][key2][key3] = value.object;
}
template <typename Value>
void JSONWrapper::Set(std::string key, Value value){
    object[key] = value;
}
template <typename Value>
void JSONWrapper::NestedSet(std::string key1, std::string key2,Value value){
    object[key1][key2] = value;
}
template <typename Value>
void JSONWrapper::NestedSet(std::string key1, std::string key2,std::string key3,Value value){
    object[key1][key2][key3] = value;
}
template <typename Value>
void JSONWrapper::NestedSet(std::string key1, std::string key2, std::string key3, std::string key4, Value value){
    object[key1][key2][key3][key4] = value;
}
template <typename Value>
Value JSONWrapper::Get(std::string key){
    return object[key];
}
template <typename Value>
Value JSONWrapper::NestedGet(std::string key, std::string key2){
    return object[key][key2];
}
template <typename Value>
Value JSONWrapper::NestedGet(std::string key, std::string key2, std::string key3){
    return object[key][key2][key3];
}
template <typename Value>
Value JSONWrapper::NestedGet(std::string key, std::string key2, std::string key3, std::string key4){
    return object[key][key2][key3][key4];
}
bool JSONWrapper::contains(std::string key){
    return object.contains(key);
}
#include "json.inc"
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
