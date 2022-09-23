#include <string>
#include <vector>
#include <sstream>
#include <utility>
#include <fstream>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <unordered_set>
#include <tuple>
#include <string>
#include <fstream>
#include <omp.h>
#include "config.h"
#include "io.h"
#include "argparse/argparse.h"
using namespace std;

typedef unsigned long long ull;

#ifndef _UTILS
#define _UTILS
#include "json.h"

template<typename T, typename U>
bool sortedge(const pair<T,U> &a,
              const pair<T,U> &b) {
    if(a.first == b.first) {
        return (a.second < b.second);
    } else {
        return (a.first < b.first);
    }
}

bool parse_arguments(int argc, const char** argv, string& input_graph,
  #ifdef _BINNING
  string& binning_experiment_json_file_name,
  #endif
  string& output_json_file_name, int &num_average){
  argparse::ArgumentParser parser("JaccardML", "GPU calculation of jaccard weights");
  parser.add_argument().names({"-i"}).description("Path to the input graph edge list file").required(true);
  #ifdef _BINNING
  parser.add_argument().names({"-e"}).description("Path to the JSONWrapper file with binning experiment parameters").required(true);
  #endif
  parser.add_argument().names({"-a"}).description("Number of runs to average timings over").required(false);
  parser.add_argument().names({"-j"}).description("Path to the JSONWrapper file to print experiment outputs").required(false);
  parser.enable_help();
  auto error = parser.parse(argc, argv);
  if (error){
    cout << error << endl;
    parser.print_help();
    return false;
  }
  if (parser.exists("help")){
    parser.print_help();
    return 0;
  }
  input_graph = parser.get<string>("i");
  #ifdef _BINNING
  binning_experiment_json_file_name = parser.get<string>("e");
  #endif
  if (parser.exists("a")){
    num_average = parser.get<int>("a");
  } else {
    num_average = 1;
  }
  if (parser.exists("j")){
    output_json_file_name  = parser.get<string>("j");
  } else {
    unsigned long long milliseconds_since_epoch = chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now().time_since_epoch()).count();
    output_json_file_name = "output_"+to_string(milliseconds_since_epoch)+".json";
  }
  return true;
}

template<typename T, typename C>
T min(T a, C b){
  if (a<b) return a;
  return b;
}
template <typename EN, typename VID>
inline EN host_bst(const EN* xadj, const VID* adj, VID neighbor, VID target){
  EN match   = (EN)(-1); 
  EN left  = xadj[neighbor]+1; 
  EN right = xadj[neighbor + 1]; 
  VID curr;
  EN middle;
  while (left <= right) { 
    middle = ((unsigned long long)left + (unsigned long long)right) >> 1;
    curr  = adj[middle-1];
    if (curr > target) {
      right = middle - 1;
    } else if (curr < target) {
      left = middle + 1;
    } else {
      match = middle-1;
      break;
    }
  }
  return match;
}

template <typename IDType, typename NNZType>
struct graph{
    NNZType *xadj, *xadj_start;
    IDType *adj, *is, *tadj;
    IDType n;
    NNZType m;
};

template <typename IDType, typename ReadType>
graph<IDType, IDType> open_graph(string graph_name, bool directed){
    std::string binary_name = graph_name + ".met.bin";
    ifstream infile_bin(binary_name, ios::in | ios::binary);
    IDType *xadj, *adj, *is, *tadj, *xadj_start;
    IDType n, m;
    if(infile_bin.is_open()) {

        ReadType *xadj_r, *adj_r, *is_r, *tadj_r, n_r, m_r;
        cout << "Binary CSR is found. Reading ... " << endl;
        infile_bin.read((char*)(&n_r), sizeof(ReadType));
        infile_bin.read((char*)(&m_r), sizeof(ReadType));
        m = m_r;
        n = n_r;
        xadj_r = new ReadType[n + 1];
        infile_bin.read((char*)xadj_r, sizeof(ReadType) * (n + 1));

        adj_r = new ReadType[m];
        infile_bin.read((char*)adj_r, sizeof(ReadType) * m);

        tadj_r = new ReadType[m];
        infile_bin.read((char*)tadj_r, sizeof(ReadType) * m);

        is_r = new ReadType[m];
        infile_bin.read((char*)is_r, sizeof(ReadType) * m);

        if (is_same<IDType, ReadType>::value == true){
            xadj = (IDType*)xadj_r;
            tadj =(IDType*)tadj_r;
            adj = (IDType*)adj_r;
            is = (IDType*)is_r;
        } else {
            xadj = new IDType[n + 1];
            adj = new IDType[m];
            tadj = new IDType[m];
            is = new IDType[m];
            for (unsigned long long i =0; i<m; i++){
                tadj[i] = tadj_r[i];
                is[i] = is_r[i];
                adj[i] = adj_r[i];
            }
            for (unsigned long long i = 0; i<n+1; i++) xadj[i] = xadj_r[i];
            delete [] xadj_r;
            delete [] adj_r;
            delete [] tadj_r;
            delete [] is_r;
        }
        cout << "Done reading binary CSR. " << endl;
    } else {
        cout << "Reading edge list and creating CSR ..." <<endl;
        ifstream infile(graph_name);
        if(infile.is_open()) {
            IDType u, v, edges_read = 0;
            n = 0;

            vector< std::pair<IDType, IDType> > edges;
            //vertices are 0-based
            while (infile >> u >> v) {
                if(u != v) {
                    edges.push_back(std::pair<IDType, IDType>(u, v));
                    if (!directed)
                        edges.push_back(std::pair<IDType, IDType>(v, u));

                    n = max(n, u);
                    n = max(n, v);

                    edges_read++;
                }
            }
            n++;
            cout << "No vertices is " << n << endl;
            cout << "No read edges " << edges_read << endl;

            sort(edges.begin(), edges.end(), sortedge<IDType, IDType>);
            edges.erase( unique( edges.begin(), edges.end() ), edges.end() );

            //allocate the memory
            xadj = new IDType[n + 1];

            m = edges.size();
            adj = new IDType[m];
            tadj = new IDType[m];
            is = new IDType[m];
            cout << "No edges is " << m << endl;

            //populate adj and xadj
            memset(xadj, 0, sizeof(IDType) * (n + 1));
            for (IDType i = 0; i < n+1; i++) xadj[i] =0;
            IDType mt = 0;
            for(std::pair<IDType, IDType>& e : edges) {
                xadj[e.first + 1]++;
                is[mt] = e.first;
                adj[mt++] = e.second;
            }

            for(IDType i = 1; i <= n; i++) {
                xadj[i] += xadj[i-1];
            }

            if (!directed){
                for(IDType i = 0; i < m; i++) {
                    tadj[i] = xadj[adj[i]]++;
                }
                for(IDType i = n; i > 0; i--) {
                    xadj[i] = xadj[i-1];
                }
                xadj[0] = 0;
            } else {
                for (IDType i =0; i < m; i++){
                    tadj[i] = host_bst(xadj, adj, adj[i], is[i]);
                }
            }

            cout << "Done reading CSR." << endl;
            ofstream outfile_bin(binary_name, ios::out | ios::binary);
            if(outfile_bin.is_open()) {
                cout << "Writing the CSR as a binary file - ";
                if (is_same<IDType, ReadType>::value){
                    cout << "writing with the same type ..." << endl;
                    outfile_bin.write((char*)(&n), sizeof(IDType));
                    outfile_bin.write((char*)(&m), sizeof(IDType));
                    outfile_bin.write((char*)xadj, sizeof(IDType) * (n + 1));
                    outfile_bin.write((char*)adj, sizeof(IDType) * m);
                    outfile_bin.write((char*)tadj, sizeof(IDType) * m);
                    outfile_bin.write((char*)is, sizeof(IDType) * m);
                } else {
                    cout << "writing with different types ..." << endl;
                    ReadType *xadj_r, *adj_r, *tadj_r, *is_r, m_r, n_r;
                    n_r = n;
                    m_r = m;
                    xadj_r = new ReadType[n + 1];
                    adj_r = new ReadType[m];
                    tadj_r = new ReadType[m];
                    is_r = new ReadType[m];
                    for (unsigned long long i =0; i<m; i++){
                        adj_r[i] = adj[i];
                        tadj_r[i] = tadj[i];
                        is_r[i] = is[i];
                    }
                    for (unsigned long long i = 0; i<n+1; i++) xadj_r[i] = xadj[i];
                    outfile_bin.write((char*)(&n_r), sizeof(ReadType));
                    outfile_bin.write((char*)(&m_r), sizeof(ReadType));
                    outfile_bin.write((char*)xadj_r, sizeof(ReadType) * (n + 1));
                    outfile_bin.write((char*)adj_r, sizeof(ReadType) * m);
                    outfile_bin.write((char*)tadj_r, sizeof(ReadType) * m);
                    outfile_bin.write((char*)is_r, sizeof(ReadType) * m);
                    delete [] xadj_r;
                    delete [] adj_r;
                    delete [] tadj_r;
                    delete [] is_r;
                }
                cout << "Done writing binary CSR file."<< endl;
            }
        } else {
            cout << "The file does not exist " << endl;
            throw -1;
        }

    }
#ifdef SORT_ASC
    double sorting_time = sort_graph<SORT_ASC, IDType, IDType>(xadj, adj, n, m);
  print_res(cout, "Sorting" , to_string(sorting_time), "-");
  output_file << argv[1]  <<'\t'<< "Sorting" << '\t' << sorting_time << '\t' << 0 << endl;
  for (IDType i = 0; i < n; i++){
    for(IDType j = xadj[i]; j < xadj[i+1]; j++) {
      is[j] = i;
    }
  }
#endif
    if (!directed){
        xadj_start = new IDType[n];
        for (IDType i = 0; i < n; i++) {
            IDType start_index = 0;
            while (start_index < xadj[i+1] - xadj[i] && adj[start_index + xadj[i]] < i) start_index++;
            xadj_start[i] = start_index;
        }
        for (IDType i =0; i< n; i++){
            if (xadj_start[i]+xadj[i]<xadj[i] || xadj_start[i]+xadj[i] > xadj[i+1]) cout << "bad xadj start at " << i << "xadjstart = " <<  xadj_start[i] << "xadj (" << xadj[i] << ", " << xadj[i+1] << endl;
            int bad = 0;
            for (IDType j = xadj_start[i]+xadj[i] ;j < xadj[i+1]; j++){
                if (adj[j] < i) bad++;
            }
            if (bad >0 )cout << "bad " << i <<" = " << bad << endl;
            bad =0;
            for (IDType j = xadj[i]; j < xadj_start[i]+xadj[i]; j++){
                if (adj[j] > i ) bad++;
            }
            if (bad >0 )cout << "bad " << i <<" = " << bad << endl;
        }
#ifndef SORT_ASC
        for(IDType i = 0; i < n; i++) {
            for(IDType j = xadj[i]; j < xadj[i + 1]; j++) {
                if(i != adj[tadj[j]]) {
                    cout << "problem: " << i << " " << j << " " << adj[j] << " " << tadj[j] <<  endl;
                }
            }
        }
#endif
    } else {
        xadj_start = nullptr;
    }
    graph<IDType, IDType> ret;
    ret.xadj = xadj;
    ret.adj = adj;
    ret.tadj = tadj;
    ret.is = is;
    ret.m = m;
    ret.n = n;
    ret.xadj_start = xadj_start;
    return ret;
}


template <typename IDType, typename ReadType>
void print_graph_statistics(graph<IDType, IDType> g, JSONWrapper& output_json){
    IDType *xadj = g.xadj, m = g.m, n = g.n;
    JSONWrapper metadata = output_json.GetJSON("metadata");
    metadata.Set("m", m);
    metadata.Set("n", n);

    IDType max_deg = 0, min_deg = n, deg, deg1 = 0, deg2 = 0, deg3 = 0, degg32 = 0, degg64 = 0, degg128 = 0, degg256 = 0, degg512 = 0, degg1024 = 0;
    for(IDType u = 0; u < n; u++) {
        deg = (xadj[u + 1] - xadj[u]);
        deg1 += (deg == 1);
        deg2 += (deg == 2);
        deg3 += (deg == 3);
        degg32 += (deg >= 32);
        degg64 += (deg >= 64);
        degg128 += (deg >= 128);
        degg256 += (deg >= 256);
        degg512 += (deg >= 512);
        degg1024 += (deg >= 1024);
        if(deg < min_deg) {min_deg = deg;}
        if(deg > max_deg) {max_deg = deg;}
    }

    metadata.Set("min_deg", min_deg);
    metadata.Set("max_deg", max_deg);
    metadata.Set("avg_deg", ((float)m)/n);

    cout << "Min deg: " << min_deg << endl;
    cout << "Max deg: " << max_deg << endl;
    cout << "Avg deg: " << ((float)m)/n << endl;
    cout << "---------------------------" << endl;
    cout << "# deg 1: " << deg1 << endl;
    cout << "# deg 2: " << deg2 << endl;
    cout << "# deg 3: " << deg3 << endl;
    cout << "---------------------------" << endl;
    cout << "# deg>32: " << degg32 << endl;
    cout << "# deg>64: " << degg64 << endl;
    cout << "# deg>128: " << degg128 << endl;
    cout << "# deg>256: " << degg256 << endl;
    cout << "# deg>512: " << degg512 << endl;
    cout << "# deg>1024: " << degg1024 << endl;
    cout << "---------------------------" << endl << endl;
    output_json.SetJSON("metadata", metadata);
    //int in_range = 0;
    //int edges = 0;
    //for (IDType u = 0; u < n; u++){
    //    IDType degu = xadj[u+1]-xadj[u];
    //    if (degu >= MINDEG && degu <= MAXDEG) {
    //        in_range++;
    //        for (IDType ptr = xadj[u]; ptr < xadj[u+1]; ptr++){
    //            if (u < adj[ptr]) edges++;
    //        }
    //    }
    //}
    //cout << "In range " << in_range << " will be processed " << edges << endl;
    cout << "###### GRAPH STATISTICS ######" << endl;
    cout << "---------------------------" << endl;
    cout << "No vertices is " << n << endl;
    cout << "No edges is " << m << endl;
    cout << "---------------------------" << endl;
    //Compute edge-based metrics

}
// Writes the correct Jaccard values to disk if not yet written
// If `emetrics_correct` doesn't contain jaccard values, copies them from `emetrics` to it
template <class JAC, class C>
void write_correct(bool& have_correct, JAC* emetrics, JAC* emetrics_correct, C m, string out_name){
  if (!have_correct){
      ofstream outfile_bin(out_name, ios::out | ios::binary);
      cout << "Writing correct jaccard values to disk ...\n";
      outfile_bin.write((char*)(emetrics), sizeof(JAC)*m);
      if (emetrics != emetrics_correct)
        memcpy(emetrics_correct, emetrics, sizeof(JAC)*(long long)m);
      cout << "Finished writing correct jaccard values\n";
      have_correct = true;
  } else {
    cout << "Already have correct jaccard values loaded\n";
  }
}

//#define _CHECK_CORRECTNESS
template <bool ASC, typename EN, typename VID>
double sort_graph(EN *& xadj, VID *& adj, VID n, EN m){
  double start = omp_get_wtime();
  EN *edge_count = new EN[n + 2](); // edge_count[i+1] = k -> there are k vertices with i edges (ASC)
                                    //                      there are k vertices with n-i edges (!ASC)
  EN *mrkr = new EN[n+1]();         // mrkr[i] = j -> so far, j nodes with i edges have been given new ids
  for (VID i = 0; i < n; i++)
  {
    if (ASC)
      edge_count[(xadj[i + 1] - xadj[i])+1]++;
    else
      edge_count[n - (xadj[i + 1] - xadj[i])+1]++;
  }
  for (VID i = 1; i <= n; i++)
  {
    edge_count[i] += edge_count[i - 1];
  }
  VID * new_to_old = new VID[n]; // new_to_old[i] = j -> the vertex i in the new CSR was j in the old on
  VID * old_to_new = new VID[n]; // new_to_old[j] = i -> the vertex i in the new CSR was j in the old on
  for (VID i = 0; i < n; i++)
  {
    EN edges = xadj[i + 1] - xadj[i];
    EN ec;
    if (ASC)
      ec = edge_count[(edges)]; // number of vertices who have more edges than (V[i+1] - V[i])
    else
      ec = edge_count[n-(edges)]; // number of vertices who have more edges than (V[i+1] - V[i])
    new_to_old[ec + mrkr[ec]] = i;
    old_to_new[i] = ec+mrkr[ec];
    mrkr[ec]++;
  } 
  delete[] mrkr;
  delete[] edge_count;
  EN * new_xadj = new EN[n+1];
  VID * new_adj = new VID[m];
  EN curr = new_xadj[0] = 0;
  for (VID i =0; i< n; i++){
    for (VID j = xadj[new_to_old[i]] ;j<xadj[new_to_old[i]+1] ;j++){
      new_adj[curr++] = old_to_new[adj[j]];
    }
    std::sort(&new_adj[new_xadj[i]], &new_adj[curr]);
    new_xadj[i+1] = curr;
  }
  double end = omp_get_wtime();
  #ifdef _CHECK_CORRECTNESS
  std::unordered_set<VID> vids;
  for (VID old_vertex =0; old_vertex<n ; old_vertex++){
    VID new_vertex = old_to_new[old_vertex];
    if (xadj[old_vertex+1] - xadj[old_vertex] != new_xadj[new_vertex+1] - new_xadj[new_vertex]){
      printf("SORT ERROR: new |E| %d old |E| %d\n", adj[old_vertex+1] - adj[old_vertex], new_adj[new_vertex+1] - new_adj[new_vertex]);
      exit(1);
    }
    for (EN edge = 0; edge < xadj[old_vertex+1] - xadj[old_vertex]; edge++){
      EN old_edge = adj[xadj[old_vertex]+edge];
      vids.insert(old_edge);
    }
    for (EN edge = 0; edge < xadj[old_vertex+1] - xadj[old_vertex]; edge++){
      EN new_edge = new_adj[new_xadj[new_vertex]+edge];
      if (vids.find(new_to_old[new_edge]) == vids.end()){
        printf("SORT ERROR: edges don't match for  new %d\n", new_edge);
        exit(1);
      }
    }
    vids.clear();
  }
  for (EN i = 0; i<n-1; i++){
    //if (new_xadj[i+1] - new_xadj[i] < new_xadj[i+2]-new_xadj[i+1]){
    if (ASC){
      if (new_xadj[i+1] - new_xadj[i] > new_xadj[i+2]-new_xadj[i+1]){
        printf("SORT ERROR: bad sort at i %d -  %d %d %d \n", i, new_xadj[i] , new_xadj[i+1] , new_xadj[i+2]);
        exit(1);
      }
    } else {
      if (new_xadj[i+1] - new_xadj[i] < new_xadj[i+2]-new_xadj[i+1]){
        printf("SORT ERROR: bad sort at i %d -  %d %d %d \n", i, new_xadj[i] , new_xadj[i+1] , new_xadj[i+2]);
        exit(1);
      }
    }
  }
  printf("No errors during sort!\n");
  #endif
  delete [] adj;
  delete [] xadj;
  adj = new_adj;
  xadj = new_xadj;
  return end - start;
}

std::vector<std::string > tokenize_string(std::string input, char delim){
  std::stringstream ss(input);
  std::vector<std::string > tokens;
  std::string token;
  while (getline(ss, token, delim)) tokens.push_back(token);
  return tokens;
}

bool check_file_exists(std::string name){
  std::ifstream f(name.c_str());
  return f.good();
} 


ostream & operator<<(ostream & out, vector<tuple<string, unsigned long long, double>> rhs){
  for (auto tup : rhs){
    out << get<0>(tup) <<" " << get<1>(tup) << " " << get<2>(tup) << std::endl;
  }
  return out;
}

vector<tuple<pair<unsigned long long, unsigned long long>, double, JSONWrapper>> average_kernel_times(vector<vector<tuple<pair<unsigned long long, unsigned long long>, double, JSONWrapper>>> kernel_times){
  vector<tuple<pair<unsigned long long, unsigned long long>, double, JSONWrapper>> ret;
  for (int j = 0; j<kernel_times[0].size(); j++){
    double time = 0;
    for (int i = 0; i<kernel_times.size(); i++){
      time+=get<1>(kernel_times[i][j]);
    }
    ret.push_back(make_tuple(get<0>(kernel_times[0][j]), time/kernel_times.size(),get<2>(kernel_times[0][j])));
  }
  return ret; 
}

void print_res_bin(ostream& out, std::string name, std::string bin_size, std::string time){
 out << std::setw(30) << std::left << name << std::setw(20) << std::right << bin_size << setw(15) << time << endl;
}

void pretty_print_results(ostream& out, std::string name, std::string time, std::string errors){
 out << std::setw(30) << std::left << name << std::setw(20) << std::right << time << setw(10) << errors << endl;
}
std::vector<int> get_slurm_available_gpus(){
  std::cout << "Checking GPU availability" << std::endl;
  std::vector<int > gpus_ints;
  const char * slurm_gpus = std::getenv("CUDA_VISIBLE_DEVICES");
  if (slurm_gpus != NULL){
    std::cout << "Found SLURM allocated GPUs: " << slurm_gpus << std::endl;
    std::vector<std::string > gpus_strings = tokenize_string(std::string(slurm_gpus), ',');
    for (auto gpu_string : gpus_strings)
      gpus_ints.push_back(stoi(gpu_string));

  } 
  std::cout << "No SLURM allocated GPUs" << std::endl;
  return gpus_ints;  
}

int get_device_id(int user_choice){
    vector<int> available_devices =  get_slurm_available_gpus();
    int device_id = 0;
    if (available_devices.size()>0){
        cout << "Detected that this a SLURM job - using available GPUs" << endl;
        bool device_is_available = false;
        for (auto device : available_devices) if (user_choice == device_id) device_is_available = true;
        if (!device_is_available) device_id = available_devices[0];
        int count;
        cudaGetDeviceCount(&count);
        cout << "Only " << count << " devices are available\n";
        if (device_id>= count){
            cout << "Using device 0 " << endl;
            device_id = 0;
        }
    }
    return device_id;
}

std::string to_string(std::pair<unsigned long long, unsigned long long> p){
  return "\""+to_string(p.first)+"-"+to_string(p.second)+"\"";
}

void print_binning_stuff(ostream& ccout, ofstream& binning_output_file, vector<tuple<string, std::pair<unsigned long long, unsigned long long>, double>> kernel_time, unsigned long long errors, double start, double end, string graph){

  std::string details = "[";
  print_res_bin(ccout, "Algorithm", "Bin ver/edge", "Time");
  for (auto & kernel : kernel_time){
    print_res_bin(ccout, get<0>(kernel), to_string(get<1>(kernel)), to_string(get<2>(kernel)));
    details+="[\"" +get<0>(kernel)+"\","+to_string(get<1>(kernel))+","+to_string(get<2>(kernel))+"],";
  }
  details = details.substr(0, details.length()-1);
  details+="]";
  print_res_bin(ccout, "", "Errors", "Total time");
  print_res_bin(ccout, "",  to_string(errors), to_string(end-start));
  binning_output_file << graph <<"\t"<<"_BINNING"<< '\t' <<  end-start << '\t' << errors << '\t' << details << endl;

}

template <typename T>
T max1(T a){
  return max(a,(T) 1);
}

template<typename VID>
std::string generate_name(std::string name, std::string range, dim3 grid, dim3 block, VID SM_FAC){
  name+="_max-"+range;
  name+="_grid.x-"+to_string(grid.x); 
  name+="_grid.y-"+to_string(grid.y); 
  name+="_grid.z-"+to_string(grid.z); 
  name+="_block.x-"+to_string(block.x); 
  name+="_block.y-"+to_string(block.y); 
  name+="_block.z-"+to_string(block.z); 
  name+="_SM-"+to_string(SM_FAC);
  return name;
}

template<typename VID>
JSONWrapper generate_json(std::string name, int g, int a, int range, dim3 grid, dim3 block, VID SM_FAC){
  JSONWrapper information;
  information.Set("name", name);
  information.Set("g",  g);
  information.Set("a",  a);
  information.Set("b_max",  range);
  information.SetJSON("grid",  JSONWrapper());
  information.NestedSet("grid", "x", grid.x);
  information.NestedSet("grid", "y", grid.y);
  information.NestedSet("grid", "z", grid.z);
  information.SetJSON("block", JSONWrapper());
  information.NestedSet("block", "x", block.x);
  information.NestedSet("block", "y", block.y);
  information.NestedSet("block", "z", block.z);
  information.Set("sm_fac", SM_FAC);
  return information;
}

// compares two vectors of numbers element wise and returns (mean, std, num_different)
template<typename T, typename C>
std::tuple<double, double, ull> compare_vectors(T* v1, T* v2, C count, T threshold, vector<C>& indexes){
  ull  total_different = 0;
  double sum = 0;
  for (C i = 0; i<count ; i++){
    double diff =abs(v1[i]-v2[i]); 
    if (diff>threshold){
      indexes.push_back(i);
      total_different++;
      sum+=diff;
    }
  }
  double mean = sum/ double(count);
  double sum_sqrd=0;
  for (C i = 0; i<count ; i++){
    double diff =abs(v1[i]-v2[i]); 
    if (diff>threshold){
      sum_sqrd+=pow(diff-mean,2);
    }
  }
  double std_div = sqrt(sum_sqrd/double(count));
  return std::make_tuple(mean, std_div, total_different);
}

template <typename T>
ull compare_metrics(T* emetrics, T* emetrics_second, ull m) {
  ull error_count = 0;
  for(ull e = 0; e < m; e++) {
    for(ull i = 0; i < 1; i++) {
      if(fabs(emetrics[e * 1 + i] - emetrics_second[e * 1 + i]) > 0.001) {
        error_count++;
      }	
    }
  }
  //cout << "Error in " << error_count << " edges out of " << no_emetrics * m << endl;
  return error_count;
}

template<typename T, typename C, typename EN>
std::tuple<double, double, ull> compare_jaccards(string j1_name, string j2_name, T* j1, T* j2, C count, T threshold,EN * xadj, EN* adj, EN* is, bool verbose = false, bool print_samples = false){
  vector<C> indexes;
  std::tuple<double, double, ull> comparison_tuple= compare_vectors(j1, j2, count,  threshold, indexes);
  if (indexes.size() > 0){
    cout << "Errors:\n";
    for (int i =0; i<indexes.size() && i < 10; i++){
      cout << indexes[i] << " " << j1[indexes[i]] << " " << j2[indexes[i]] << endl;
    }
  }
  if (verbose){
    cout << "Comparing "<< j1_name << " Vs. "<< j2_name << endl;
    cout << "Mean = " << get<0>(comparison_tuple) << " STD = " << get<1>(comparison_tuple)  << "Tot. diff. = " << get<2>(comparison_tuple) << endl;
    if (print_samples){
      cout << "Samples:\n";
      cout << j1[0] << " " << j2[0] << endl; 
      cout << j1[10] << " " << j2[10] << endl; 
      cout << j1[100] << " " << j2[100] << endl; 
      cout << j1[1000] << " " << j2[1000] << endl; 
      cout << j1[10000] << " " << j2[10000] << endl; 
    }
  }
  return comparison_tuple;
}

template <typename IDType, typename JaccardType>
void validate_and_write(graph<IDType, IDType> g, string method_name, JaccardType * emetrics_truth, JaccardType * emetrics_calculated,  double total_time, int num_average, std::string output_file_name, JSONWrapper& output_json, string jaccards_output_path, bool& have_correct){
  write_correct(have_correct, emetrics_calculated, emetrics_truth, g.m, jaccards_output_path);
  std::tuple<double, double, unsigned long long> res;
  res = compare_jaccards(string("Ground truth"), method_name, emetrics_truth, emetrics_calculated, g.m, JaccardType(0), g.xadj, g.adj, g.is); 
  unsigned long long errors = get<2>(res);
  pretty_print_results(cout, method_name, to_string(total_time/num_average), to_string(errors));
  output_json.SetJSONNested("experiments", method_name, get_result_json(total_time/num_average, errors));
  write_json_to_file(output_file_name, output_json);
}

template <typename IDType, typename JaccardType>
void validate_and_write_binning(graph<IDType, IDType> g, vector<tuple<pair<unsigned long long, unsigned long long>, double, JSONWrapper>> timings, string method_name, JaccardType * emetrics_truth, JaccardType * emetrics_calculated,  double total_time, int num_average, std::string output_file_name, JSONWrapper& output_json, string jaccards_output_path, bool& have_correct){
  write_correct(have_correct, emetrics_calculated, emetrics_truth, g.m, jaccards_output_path);
  std::tuple<double, double, unsigned long long> res;
  res = compare_jaccards(string("Ground truth"), method_name, emetrics_truth, emetrics_calculated, g.m, JaccardType(0), g.xadj, g.adj, g.is); 
  unsigned long long errors = get<2>(res);
  pretty_print_results(cout, method_name, to_string(total_time/num_average), to_string(errors));
  JSONWrapper experiment_json;
  experiment_json.Set("time", total_time/num_average);
  experiment_json.SetJSON("bins", JSONWrapper());
  for (int i = 0; i < timings.size(); i++){
    auto timing = timings[i];
    experiment_json.SetJSONNested("bins", std::to_string(i), JSONWrapper());
    experiment_json.SetJSONNested("bins", std::to_string(i), "size", JSONWrapper());
    experiment_json.NestedSet("bins", std::to_string(i), "size", "first", get<0>(timing).first);
    experiment_json.NestedSet("bins", std::to_string(i), "size", "second", get<0>(timing).second);
    experiment_json.NestedSet("bins", std::to_string(i), "time", get<1>(timing));
    experiment_json.SetJSONNested("bins", std::to_string(i), "information", get<2>(timing));
  }
  output_json.SetJSONNested("experiments", "binning", method_name, experiment_json);
  write_json_to_file(output_file_name, output_json);
}
#endif
