#include <cassert>
#include <cstdint> // for uint64_t
#include <cstring> // for memcpy
#include <deque>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <filesystem>
#include <sys/mman.h> // for mmap, munmap
#include <sys/stat.h> // for struct stat
#include <fcntl.h> // for O_RDONLY
#include <unistd.h> // for close
#include <algorithm> // for sort
#include <random>
#include <thread>
#include <fstream>
#include <sstream>
#include <chrono>
#include <numeric>
#include <cmath>

#define U64 uint64_t
#define U32 uint32_t
#define U16 uint16_t
#define U8 uint8_t

using namespace std;
namespace fs = std::filesystem;

void assert_little_endian() {
    unsigned int i = 1;
    char *c = (char*)&i;
    assert (*c);
}
const auto PAGESIZE = sysconf(_SC_PAGESIZE);

struct DatastoreShard {
    U8* ds;
    U8* sa;
    U64 tok_cnt;
    U64 ds_size;
    U8 ptr_size;
};
struct FindResult {
    U64 cnt;
    vector<pair<U64, U64>> segment_by_shard;
};
struct DistResult {
    vector<U16> tokens;
};

pair<U8*, U64> mmap_file(const string &path) {
    int f = open(path.c_str(), O_RDONLY);
    assert (f != -1);
    struct stat s;
    auto fstat_ret = fstat(f, &s);
    assert (fstat_ret != -1);
    U8* p = static_cast<U8*>(mmap(NULL, s.st_size, PROT_READ, MAP_PRIVATE, f, 0));
    assert (p != MAP_FAILED);
    madvise(p, s.st_size, MADV_RANDOM);
    return {p, s.st_size};
}

pair<U8*, U64> load_file(const string &path) {
    ifstream f(path, ifstream::binary);
    assert (f);
    f.seekg(0, f.end);
    streamsize size = f.tellg();
    f.seekg(0, f.beg);
    U8* p = new U8[size];
    f.read(reinterpret_cast<char*>(p), size);
    assert (f);
    madvise(p, size, MADV_RANDOM);
    return {p, size};
}

class NGramLanguageModeling {
public:

    NGramLanguageModeling(): _index_dir(""), _eos_token_id(0) {}

    NGramLanguageModeling(const string index_dir, const U16 eos_token_id)
        : _index_dir(index_dir), _eos_token_id(eos_token_id) {

        assert_little_endian();
        assert (fs::exists(index_dir));

        _version = (index_dir.find("v5") == string::npos) ? 4 : 5;
        vector<string> ds_paths, sa_paths;
        for (const auto & entry : fs::directory_iterator(index_dir)) {
            if (entry.path().string().find("tokenized") != string::npos) {
                ds_paths.push_back(entry.path());
            } else if (entry.path().string().find("table") != string::npos) {
                sa_paths.push_back(entry.path());
            }
        }
        sort(ds_paths.begin(), ds_paths.end());
        sort(sa_paths.begin(), sa_paths.end());
        assert (ds_paths.size() == sa_paths.size());
        _num_shards = ds_paths.size();
        assert (_num_shards > 0);

        for (auto s = 0; s < _num_shards; s++) {
            auto [ds, ds_size] = load_file(ds_paths[s]);
            auto [sa, sa_size] = load_file(sa_paths[s]);

            assert (ds_size % 2 == 0);
            U64 tok_cnt = ds_size / sizeof(U16);
            assert (sa_size % tok_cnt == 0);
            U8 ptr_size = (U8)(sa_size / tok_cnt);

            auto shard = DatastoreShard{ds, sa, tok_cnt, ds_size, ptr_size};
            _shards.push_back(shard);
        }

        cout << "Loaded index " << index_dir << " with " << _num_shards << " shards" << endl;
    }

    ~NGramLanguageModeling() {
        for (auto shard : _shards) {
            munmap(shard.ds, shard.ds_size);
            munmap(shard.sa, shard.tok_cnt * shard.ptr_size);
        }
    }

    virtual FindResult find(const vector<U16> &input_ids, const size_t start, const size_t end, vector<pair<U64, U64>> hint_segment_by_shard = {}) const {

        assert (start <= end);
        assert (end <= input_ids.size());
        vector<pair<U64, U64>> segment_by_shard(_num_shards);
        if (start == end) {
            assert (hint_segment_by_shard.empty());
            for (auto s = 0; s < _num_shards; s++) {
                segment_by_shard[s] = {0, _shards[s].tok_cnt};
            }
        } else {
            if (hint_segment_by_shard.empty()) {
                hint_segment_by_shard.resize(_num_shards);
                for (auto s = 0; s < _num_shards; s++) {
                    hint_segment_by_shard[s] = {0, _shards[s].tok_cnt};
                }
            }
            assert (hint_segment_by_shard.size() == _num_shards);
            vector<U16> reversed_input_ids;
            const U8* input_buf;
            if (_version == 4) {
                input_buf = reinterpret_cast<const U8*>(input_ids.data() + start);
            } else if (_version == 5) {
                reversed_input_ids = vector<U16>(input_ids.begin() + start, input_ids.begin() + end);
                reverse(reversed_input_ids.begin(), reversed_input_ids.end());
                input_buf = reinterpret_cast<const U8*>(reversed_input_ids.data());
            }
            U64 num_bytes = (end - start) * sizeof(U16);

            for (auto s = 0; s < _num_shards; s++) {
                const auto &shard = _shards[s];

                U64 lo = hint_segment_by_shard[s].first, hi = hint_segment_by_shard[s].second;
                U64 mi;
                while (lo < hi) {
                    mi = (lo + hi - 1) >> 1;
                    U64 ptr = _convert_rank_to_ptr(shard, mi);
                    auto o = lexicographical_compare_three_way(
                        shard.ds + ptr, shard.ds + min(ptr + num_bytes, shard.ds_size),
                        input_buf, input_buf + num_bytes);
                    if (o == strong_ordering::less) {
                        lo = mi + 1;
                    } else if (o == strong_ordering::greater) {
                        hi = mi;
                    } else { // o == strong_ordering::equal
                        break;
                    }
                }
                if (lo == hi) {
                    segment_by_shard[s].first = lo;
                    segment_by_shard[s].second = hi;
                    continue;
                }

                // search left boundary in (lo-1, mi], which should be >= query
                U64 l = lo - 1, r = mi; // l is always < query, r is always >= query
                while (r - l > 1) {
                    U64 m = (l + r) >> 1;
                    U64 ptr = _convert_rank_to_ptr(shard, m);
                    bool less = lexicographical_compare(
                        shard.ds + ptr, shard.ds + min(ptr + num_bytes, shard.ds_size),
                        input_buf, input_buf + num_bytes);
                    if (less) {
                        l = m;
                    } else {
                        r = m;
                    }
                }
                segment_by_shard[s].first = r;

                // search right boundary in (mi, hi], which should be > query
                l = mi, r = hi; // l is always <= query, r is always > query
                while (r - l > 1) {
                    U64 m = (l + r) >> 1;
                    U64 ptr = _convert_rank_to_ptr(shard, m);
                    bool less = lexicographical_compare(
                        input_buf, input_buf + num_bytes,
                        shard.ds + ptr, shard.ds + min(ptr + num_bytes, shard.ds_size));
                    if (less) {
                        r = m;
                    } else {
                        l = m;
                    }
                }
                segment_by_shard[s].second = r;
            }
        }
        U64 cnt = 0;
        for (const auto &segment : segment_by_shard) {
            cnt += segment.second - segment.first;
        }
        return FindResult{cnt, segment_by_shard};
    }

    virtual DistResult ntd(const vector<U16> &input_ids, const size_t start, const size_t end, const size_t support, const FindResult &prompt_find_result) const {

        assert (start <= end);
        assert (end <= input_ids.size());

        if (prompt_find_result.cnt == 0) {
            return DistResult{{}};
        }

        U64 num_bytes = (end - start) * sizeof(U16);
        vector<U16> tokens(support);
        for (U64 i = 0; i < support; i++) {
            U64 ix = (prompt_find_result.cnt * i + prompt_find_result.cnt / 2) / support;
            size_t s = 0;
            while (true) {
                const auto& shard = _shards[s];
                auto [l, r] = prompt_find_result.segment_by_shard[s];
                if (ix < r - l) break;
                ix -= r - l;
                s++;
            }
            const auto& shard = _shards[s];
            auto [l, r] = prompt_find_result.segment_by_shard[s];
            U64 rank = l + ix;
            U64 ptr = _convert_rank_to_ptr(shard, rank);
            U64 offset = ptr + num_bytes;
            U16 token_id = _convert_offset_to_token_id(shard, offset);
            tokens[i] = token_id;
        }

        return DistResult{tokens};
    }

    virtual DistResult ntd_v5(const size_t support, const FindResult &find_result, const FindResult &find_result_exclude) const {

        if (find_result.cnt == 0) {
            return DistResult{{}};
        }

        U64 cnt = find_result.cnt - find_result_exclude.cnt;
        vector<U16> tokens(support);
        for (U64 i = 0; i < support; i++) {
            U64 ix = (cnt * i + cnt / 2) / support;
            size_t s = 0;
            while (true) {
                const auto& shard = _shards[s];
                auto [l, r] = find_result.segment_by_shard[s];
                auto [l_exclude, r_exclude] = find_result_exclude.segment_by_shard[s];
                assert (l <= l_exclude && r_exclude <= r);
                U64 cnt_segment = (r - l) - (r_exclude - l_exclude);
                if (ix < cnt_segment) break;
                ix -= cnt_segment;
                s++;
            }
            const auto& shard = _shards[s];
            auto [l, r] = find_result.segment_by_shard[s];
            auto [l_exclude, r_exclude] = find_result_exclude.segment_by_shard[s];
            U64 rank = (ix < l_exclude - l) ? (l + ix) : (r_exclude + (ix - (l_exclude - l)));
            U64 ptr = _convert_rank_to_ptr(shard, rank);
            assert (ptr > 0); // because the first token is always \xff\xff and ptr cannot land there
            U64 offset = ptr - sizeof(U16);
            U16 token_id = _convert_offset_to_token_id(shard, offset);
            tokens[i] = token_id;
        }

        return DistResult{tokens};
    }

    virtual void find_5gram_dense(const vector<U16> input_ids, vector<FindResult>* results) const {
        results->resize(input_ids.size());
        auto tot_duration_us = 0;
        auto thread_start_time = chrono::high_resolution_clock::now();
        for (size_t i = 5; i <= input_ids.size(); i++) {
            auto start_time = chrono::high_resolution_clock::now();
            (*results)[i-1] = find(input_ids, i-5, i);
            auto end_time = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();
            tot_duration_us += duration;
        }
        auto avg_duration_us = tot_duration_us / (input_ids.size() - 5 + 1);
        auto tot_duration_ms = tot_duration_us / 1000;
        auto thread_end_time = chrono::high_resolution_clock::now();
        auto thread_duration_ms = chrono::duration_cast<chrono::milliseconds>(thread_end_time - thread_start_time).count();
        cerr << "find_5gram: total = " << tot_duration_ms << " ms, avg = " << avg_duration_us << " us" << endl;
        cerr << "thread duration = " << thread_duration_ms << " ms" << endl;
    }

    virtual void ntd_dense(const vector<U16> input_ids, const U64 min_cnt, const size_t support, const bool debug, vector<DistResult>* results, vector<U16>* lfns) const {
        auto thread_start_time = chrono::high_resolution_clock::now();
        auto tot_duration_us = 0;
        results->resize(input_ids.size());
        if (lfns) lfns->resize(input_ids.size());

        if (_version == 4) {
            size_t j = 0;
            FindResult find_result;
            for (size_t i = 1; i <= input_ids.size(); i++) {
                auto start_time = chrono::high_resolution_clock::now();
                if (i == 1) {
                    find_result = find(input_ids, j, i);
                } else {
                    find_result = find(input_ids, j, i, find_result.segment_by_shard);
                }
                while (find_result.cnt < min_cnt && j < i) {
                    j++;
                    find_result = find(input_ids, j, i);
                }
                (*results)[i-1] = ntd(input_ids, j, i, support, find_result);
                if (lfns) (*lfns)[i-1] = i - j;
                auto end_time = chrono::high_resolution_clock::now();
                auto duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();
                tot_duration_us += duration;
            }
        } else if (_version == 5) {
            size_t j = 0;
            for (size_t i = 1; i <= input_ids.size(); i++) {
                auto start_time = chrono::high_resolution_clock::now();
                FindResult find_result = find(input_ids, j, i);
                while (find_result.cnt < min_cnt && j < i) {
                    j++;
                    find_result = find(input_ids, j, i);
                }
                FindResult find_result_exclude;
                if (j > 0) {
                    find_result_exclude = find(input_ids, j-1, i, find_result.segment_by_shard);
                } else {
                    find_result_exclude.cnt = 0;
                    auto segment_by_shard = find_result.segment_by_shard;
                    for (auto &segment : segment_by_shard) {
                        segment.first = segment.second;
                    }
                    find_result_exclude.segment_by_shard = segment_by_shard;
                }
                (*results)[i-1] = ntd_v5(support, find_result, find_result_exclude);
                if (lfns) (*lfns)[i-1] = i - j;
                auto end_time = chrono::high_resolution_clock::now();
                auto duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();
                tot_duration_us += duration;
            }
        }

        auto avg_duration_us = tot_duration_us / input_ids.size();
        auto tot_duration_ms = tot_duration_us / 1000;
        auto thread_end_time = chrono::high_resolution_clock::now();
        auto thread_duration_ms = chrono::duration_cast<chrono::milliseconds>(thread_end_time - thread_start_time).count();
        if (debug) {
            cerr << "ntd_dense: total = " << tot_duration_ms << " ms, avg = " << avg_duration_us << " us" << endl;
            cerr << "thread duration = " << thread_duration_ms << " ms" << endl;
        }
    }

    virtual vector<vector<DistResult>> ntd_dense_batch(const vector<vector<U16>> &input_idss, const U64 min_cnt, const size_t support, const bool debug = false, vector<vector<U16>>* lfnss = nullptr) const {
        size_t B = input_idss.size();
        vector<vector<DistResult>> resultss(B);
        vector<thread> threads;
        for (size_t b = 0; b < B; b++) {
            threads.emplace_back(&NGramLanguageModeling::ntd_dense, this, input_idss[b], min_cnt, support, debug, &resultss[b], lfnss ? &(*lfnss)[b] : nullptr);
        }
        for (auto &thread : threads) {
            thread.join();
        }
        return resultss;
    }

public:

    inline U16 _convert_offset_to_token_id(const DatastoreShard &shard, const U64 offset) const {
        assert (offset % 2 == 0);
        assert (offset <= shard.ds_size);
        if (offset == shard.ds_size) {
            // This happens when we matched the very end of the ds.
            return _eos_token_id;
        }
        U16 token_id; // no need to initialize
        memcpy(&token_id, shard.ds + offset, sizeof(U16));
        // If you see \xff\xff, this actually means we're at the very end of a document.
        if (token_id == 65535) token_id = _eos_token_id;
        return token_id;
    }

    inline U64 _convert_rank_to_ptr(const DatastoreShard &shard, const U64 rank) const {
        assert (rank < shard.tok_cnt);
        U64 ptr = 0; // need to zero-initialize such that all 8 bytes are filled
        memcpy(&ptr, shard.sa + rank * shard.ptr_size, shard.ptr_size);
        return ptr;
    }

    const string _index_dir;
    const U16 _eos_token_id;
    size_t _version;
    size_t _num_shards;
    vector<DatastoreShard> _shards;
};
