#include <algorithm>
#include <cstring>
#include <iostream>
#include "infini_gram.h"

using namespace std;

void rank_processor(const size_t rank, const U64 MAX_BATCH_SIZE, const U64 MAX_SEQ_LEN, const U64 MAX_SUPPORT, const string MODE, const shared_ptr<NGramLanguageModeling> lm) {
    ifstream fifo_query("/tmp/infini_gram_query_" + to_string(rank), ios::binary);
    ofstream fifo_response("/tmp/infini_gram_response_" + to_string(rank), ios::binary);

    U64 batch_size, seq_len, support, min_cnt;
    vector<U16> input_ids(MAX_BATCH_SIZE * MAX_SEQ_LEN);
    vector<U16> output_ids(MAX_BATCH_SIZE * MAX_SEQ_LEN * MAX_SUPPORT);

    while (true) {
        auto start_time = chrono::high_resolution_clock::now();

        auto start_time_read = chrono::high_resolution_clock::now();
        fifo_query.read(reinterpret_cast<char*>(&batch_size), sizeof(U64));
        fifo_query.read(reinterpret_cast<char*>(&seq_len), sizeof(U64));
        fifo_query.read(reinterpret_cast<char*>(&support), sizeof(U64));
        fifo_query.read(reinterpret_cast<char*>(&min_cnt), sizeof(U64));
        if (batch_size > MAX_BATCH_SIZE) {
            cerr << "Error: batch_size = " << batch_size << " > " << MAX_BATCH_SIZE << endl;
            exit(1);
        }
        if (seq_len > MAX_SEQ_LEN) {
            cerr << "Error: seq_len = " << seq_len << " > " << MAX_SEQ_LEN << endl;
            exit(1);
        }
        if (support > MAX_SUPPORT) {
            cerr << "Error: support = " << support << " > " << MAX_SUPPORT << endl;
            exit(1);
        }
        fifo_query.read(reinterpret_cast<char*>(input_ids.data()), batch_size * seq_len * sizeof(U16));
        if (MODE == "debug") {
            string print_str = "[rank " + to_string(rank) + "] Request: B = " + to_string(batch_size) + ", L = " + to_string(seq_len) + ", support = " + to_string(support) + ", min_cnt = " + to_string(min_cnt) + ", input_ids = ";
            for (int i = 0; i < 10; i++) print_str += to_string(input_ids[i]) + " "; print_str += "...";
            for (int i = 0; i < 10; i++) print_str += " " + to_string(input_ids[batch_size * seq_len - 10 + i]);
            cout << print_str << endl;
        }
        auto end_time_read = chrono::high_resolution_clock::now();
        auto latency_us_read = chrono::duration_cast<chrono::microseconds>(end_time_read - start_time_read).count();

        auto start_time_decode = chrono::high_resolution_clock::now();
        vector<vector<U16>> input_idss(batch_size, vector<U16>(seq_len));
        for (auto b = 0; b < batch_size; b++) {
            copy(input_ids.begin() + b * seq_len, input_ids.begin() + (b+1) * seq_len, input_idss[b].begin());
        }
        auto end_time_decode = chrono::high_resolution_clock::now();
        auto latency_us_decode = chrono::duration_cast<chrono::microseconds>(end_time_decode - start_time_decode).count();

        auto start_time_compute = chrono::high_resolution_clock::now();
        vector<vector<DistResult>> rrr = lm->ntd_dense_batch(input_idss, min_cnt, support);
        auto end_time_compute = chrono::high_resolution_clock::now();
        auto latency_us_compute = chrono::duration_cast<chrono::microseconds>(end_time_compute - start_time_compute).count();

        auto start_time_encode = chrono::high_resolution_clock::now();
        for (auto b = 0; b < batch_size; b++) {
            for (auto l = 0; l < seq_len; l++) {
                const auto& tokens = rrr[b][l].tokens;
                assert (tokens.size() == support);
                copy(tokens.begin(), tokens.end(), output_ids.begin() + (b * seq_len + l) * support);
            }
        }
        auto end_time_encode = chrono::high_resolution_clock::now();
        auto latency_us_encode = chrono::duration_cast<chrono::microseconds>(end_time_encode - start_time_encode).count();

        auto start_time_write = chrono::high_resolution_clock::now();
        if (MODE == "debug") {
            string print_str = "[rank " + to_string(rank) + "] Response: output_ids = ";
            for (int i = 0; i < 10; i++) print_str += to_string(output_ids[i]) + " "; print_str += "...";
            for (int i = 0; i < 10; i++) print_str += " " + to_string(output_ids[batch_size * seq_len * support - 10 + i]);
            cout << print_str << endl;
        }
        fifo_response.write(reinterpret_cast<const char*>(output_ids.data()), batch_size * seq_len * support * sizeof(U16));
        fifo_response.flush();
        auto end_time_write = chrono::high_resolution_clock::now();
        auto latency_us_write = chrono::duration_cast<chrono::microseconds>(end_time_write - start_time_write).count();

        auto end_time = chrono::high_resolution_clock::now();
        auto latency_us = chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();

        if (MODE == "debug") {
            string print_str = "[rank " + to_string(rank) + "] Latency: all = " + to_string(latency_us) + "us, read = " + to_string(latency_us_read) + "us, decode = " + to_string(latency_us_decode) + "us, compute = " + to_string(latency_us_compute) + "us, encode = " + to_string(latency_us_encode) + "us, write = " + to_string(latency_us_write) + "us";
            cout << print_str << endl;
        }
    }
}

int main(int argc, char const *argv[]) {
    cout << endl;
    cout << "C++ engine rebooting" << endl;

    if (argc != 7) {
        cerr << "Usage: " << argv[0] << " <index_dir> <local_world_size> <MAX_BATCH_SIZE> <MAX_SEQ_LEN> <MAX_SUPPORT> <MODE>" << endl;
        exit(1);
    }
    const string index_dir = argv[1];
    const size_t local_world_size = std::stoi(argv[2]);
    const U64 MAX_BATCH_SIZE = std::stoi(argv[3]);
    const U64 MAX_SEQ_LEN = std::stoi(argv[4]);
    const U64 MAX_SUPPORT = std::stoi(argv[5]);
    const string MODE = argv[6];

    const U16 eos_token_id = 50279;
    shared_ptr<NGramLanguageModeling> lm = make_shared<NGramLanguageModeling>(index_dir, eos_token_id);

    cout << "C++ engine rebooted" << endl;

    vector<thread> threads;
    for (size_t rank = 0; rank < local_world_size; rank++) {
        threads.emplace_back(rank_processor, rank, MAX_BATCH_SIZE, MAX_SEQ_LEN, MAX_SUPPORT, MODE, lm);
    }
    for (auto& thread : threads) {
        thread.join();
    }

    return 0;
}
