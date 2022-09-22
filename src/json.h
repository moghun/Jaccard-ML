#ifndef _JSON_WRAPPER
#define _JSON_WRAPPER
#include "json.hpp"
#include <iostream>

class JSONWrapper {
    nlohmann::json object;

public:
    template <typename Key, typename Value>
    Value Use(Key key, Value value, bool set_not_get=true){
        if (set_not_get)
            object[key] = value;
        return object[key];
    }
    template <typename Key>
    JSONWrapper GetJSON(Key key){
        JSONWrapper newone;
        newone.object = object[key];
        return newone;
    }
    template <typename Key>
    void SetJSON(Key key, JSONWrapper value){
        object[key] = value.object;
    }
    template <typename Key1, typename Key2>
    void SetJSONNested(Key1 key1, Key2 key2, JSONWrapper value){
        object[key1][key2] = value.object;
    }
    template <typename Key1, typename Key2, typename Key3>
    void SetJSONNested(Key1 key1, Key2 key2, Key3 key3, JSONWrapper value){
        object[key1][key2][key3] = value.object;
    }
    template <typename Key, typename Value>
    void Set(Key key, Value value){
        object[key] = value;
    }
    template <typename Key1, typename Key2, typename Key3, typename Value>
    void NestedSet(Key1 key1, Key2 key2,Key3 key3,Value value){
        object[key1][key2][key3] = value;
    }
    template <typename Key1, typename Key2, typename Value>
    Value NestedUse(Key1 key, Key2 key2,  Value value, bool set_not_get){
        if (set_not_get)
            object[key][key2] = value;
        return object[key][key2];
    }
    template <typename Key1, typename Key2, typename Key3, typename Value>
    Value NestedUse(Key1 key, Key2 key2, Key3 key3, Value value, bool set_not_get){
        if (set_not_get){
            object[key][key2][key3];
            object[key][key2][key3] = value;
        }
        return object[key][key2][key3];
    }
    template <typename Key1, typename Key2, typename Key3,typename Key4, typename Value>
    Value NestedUse(Key1 key, Key2 key2, Key3 key3, Key4 key4, Value value, bool set_not_get){
        if (set_not_get)
            object[key][key2][key3][key4] = value;
        return object[key][key2][key3][key4];
    }
    template <typename Key>
    bool contains(Key key){
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