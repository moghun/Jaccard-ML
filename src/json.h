#include "json.hpp"
#include <iostream>
#include <string>
#ifndef _JSON_WRAPPER
#define _JSON_WRAPPER

class JSONWrapper {

    nlohmann::json object;

public:
    JSONWrapper GetJSON(std::string key);
    void SetJSON(std::string key, JSONWrapper value);
    void SetJSONNested(std::string key1, std::string key2, JSONWrapper value);
    void SetJSONNested(std::string key1, std::string key2, std::string key3, JSONWrapper value);
    template <typename Value>
    void Set(std::string key, Value value);
    template <typename Value>
    void NestedSet(std::string key1, std::string key2,Value value);
    template <typename Value>
    void NestedSet(std::string key1, std::string key2,std::string key3,Value value);
    template <typename Value>
    void NestedSet(std::string key1, std::string key2, std::string key3, std::string key4, Value value);
    template <typename Value>
    Value Get(std::string key);
    template <typename Value>
    Value NestedGet(std::string key, std::string key2);
    template <typename Value>
    Value NestedGet(std::string key, std::string key2, std::string key3);
    template <typename Value>
    Value NestedGet(std::string key, std::string key2, std::string key3, std::string key4);
    bool contains(std::string key);

    friend std::istream& operator>>(std::istream& in, JSONWrapper& j);

    friend std::ostream& operator<<(std::ostream& out, JSONWrapper& j);
    
};

    inline std::istream& operator>>(std::istream& in, JSONWrapper& j){
        in >> j.object;
        return in;
    }
    inline std::ostream& operator<<(std::ostream& out, JSONWrapper& j){
        out << j.object.dump(4);
        return out;
    }
#endif