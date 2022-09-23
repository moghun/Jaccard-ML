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
    JSONStore(){};
public:
    static JSONStore* GetStore(){
        mutex_.lock();
        if (singleton_ == nullptr){
            singleton_ = new JSONStore;
            store_.reserve(100);
            ids_ = 0;
        }
        mutex_.unlock();
        return singleton_;
    }    
    int AllocateJSONObject(){
        std::unique_lock<std::mutex> lock(mutex_);
        int new_id = ids_++;
        store_[new_id];
        return new_id;
    }
    nlohmann::json& GetJSONObject(int id){
        std::unique_lock<std::mutex> lock(mutex_);
        return store_[id];
    }
};
nlohmann::json& IDToJSON(int id){
    return JSONStore::GetStore()->GetJSONObject(id);
}
JSONStore* JSONStore::singleton_{nullptr};
std::mutex JSONStore::mutex_;
std::unordered_map<int, nlohmann::json> JSONStore::store_;
int JSONStore::ids_;

JSONWrapper::JSONWrapper(){
    JSONStore* store = JSONStore::GetStore();
    id = store->AllocateJSONObject();
}

int JSONWrapper::GetID(){
    return id;
}

JSONWrapper JSONWrapper::GetJSON(std::string key){
    JSONWrapper newone;
    IDToJSON(newone.id) = IDToJSON(id)[key];
    return newone;
}
void JSONWrapper::SetJSON(std::string key, JSONWrapper value){
    IDToJSON(id)[key] = IDToJSON(value.id);
}
void JSONWrapper::SetJSONNested(std::string key1, std::string key2, JSONWrapper value){
    IDToJSON(id)[key1][key2] = IDToJSON(value.id);
}
void JSONWrapper::SetJSONNested(std::string key1, std::string key2, std::string key3, JSONWrapper value){
    IDToJSON(id)[key1][key2][key3] = IDToJSON(value.id);
}
template <typename Value>
void JSONWrapper::Set(std::string key, Value value){
    IDToJSON(id)[key] = value;
}
template <typename Value>
void JSONWrapper::NestedSet(std::string key1, std::string key2,Value value){
    IDToJSON(id)[key1][key2] = value;
}
template <typename Value>
void JSONWrapper::NestedSet(std::string key1, std::string key2,std::string key3,Value value){
    IDToJSON(id)[key1][key2][key3] = value;
}
template <typename Value>
void JSONWrapper::NestedSet(std::string key1, std::string key2, std::string key3, std::string key4, Value value){
    IDToJSON(id)[key1][key2][key3][key4] = value;
}
template <typename Value>
Value JSONWrapper::Get(std::string key){
    return IDToJSON(id)[key];
}
template <typename Value>
Value JSONWrapper::NestedGet(std::string key, std::string key2){
    return IDToJSON(id)[key][key2];
}
template <typename Value>
Value JSONWrapper::NestedGet(std::string key, std::string key2, std::string key3){
    return IDToJSON(id)[key][key2][key3];
}
template <typename Value>
Value JSONWrapper::NestedGet(std::string key, std::string key2, std::string key3, std::string key4){
    return IDToJSON(id)[key][key2][key3][key4];
}
bool JSONWrapper::contains(std::string key){
    return IDToJSON(id).contains(key);
}

    std::istream& operator>>(std::istream& in, JSONWrapper& j){
        in >> IDToJSON(j.GetID());
        return in;
    }
    std::ostream& operator<<(std::ostream& out, JSONWrapper& j){
        out << IDToJSON(j.GetID()).dump(4);
        return out;
    }
#include "json.inc"