//
// Created by Amro on 3/28/2022.
//

#ifndef JACCARD_ML_IO_H
#define JACCARD_ML_IO_H
#include "json.hpp"
#include "utility"
#include "time.h"
#include <iostream>
#include <fstream>

nlohmann::json& operator<<(nlohmann::json& j, std::pair<std::string, std::string> data){
   j[data.first] = data.second;
   return j;
}
nlohmann::json get_result_json(double time, unsigned long long errors){
    nlohmann::json output;
    output["time"] = time;
    output["errors"] = errors;
    time_t time_obj = time;
    output["timestamp"] = ctime(&time_obj);
    return output;
}

nlohmann::json initialize_output_json(std::string graph_name){
    nlohmann::json metadata;
    metadata["graph name"] = graph_name;
    nlohmann::json j;
    j["metadata"] = metadata;
    j["experiments"] = nlohmann::json();
    return j;
}

nlohmann::json read_json(std::string filename){
    std::ifstream fin(filename);
    nlohmann::json output_json;
    fin >> output_json;
    return output_json;
}

void write_json_to_file(std::string filename, nlohmann::json& output_json){
    std::ofstream out(filename);
    out << output_json;
}

#endif //JACCARD_ML_IO_H
