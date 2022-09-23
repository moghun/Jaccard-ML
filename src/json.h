#ifndef _JSON_WRAPPER
#define _JSON_WRAPPER
#include "json.hpp"
#include <iostream>
#include <string>

class JSONWrapper {
    nlohmann::json object;

public:
    JSONWrapper GetJSON(std::string key){
        JSONWrapper newone;
        newone.object = object[key];
        return newone;
    }
    void SetJSON(std::string key, JSONWrapper value){
        object[key] = value.object;
    }
    void SetJSONNested(std::string key1, std::string key2, JSONWrapper value){
        object[key1][key2] = value.object;
    }
    void SetJSONNested(std::string key1, std::string key2, std::string key3, JSONWrapper value){
        object[key1][key2][key3] = value.object;
    }
    template <typename Value>
    void Set(std::string key, Value value){
        object[key] = value;
    }
    template <typename Value>
    void NestedSet(std::string key1, std::string key2,Value value){
        object[key1][key2] = value;
    }
    template <typename Value>
    void NestedSet(std::string key1, std::string key2,std::string key3,Value value){
        object[key1][key2][key3] = value;
    }
    template <typename Value>
    void NestedSet(std::string key1, std::string key2, std::string key3, std::string key4, Value value){
        object[key1][key2][key3][key4] = value;
    }
    template <typename Value>
    Value Get(std::string key){
        return object[key];
    }
    template <typename Value>
    Value NestedGet(std::string key, std::string key2){
        return object[key][key2];
    }
    template <typename Value>
    Value NestedGet(std::string key, std::string key2, std::string key3){
        return object[key][key2][key3];
    }
    template <typename Value>
    Value NestedGet(std::string key, std::string key2, std::string key3, std::string key4){
        return object[key][key2][key3][key4];
    }
    bool contains(std::string key){
        return object.contains(key);
    }

    friend std::istream& operator>>(std::istream& in, JSONWrapper& j);

    friend std::ostream& operator<<(std::ostream& out, JSONWrapper& j);
    
};

    std::istream& operator>>(std::istream& in, JSONWrapper& j){
        in >> j.object;
        return in;
    }
    std::ostream& operator<<(std::ostream& out, JSONWrapper& j){
        out << j.object.dump(4);
        return out;
    }
#endif