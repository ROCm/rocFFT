// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef FFT_HASH_H
#define FFT_HASH_H

#include "../../../library/include/rocfft/rocfft.h"
#include "../../../shared/arithmetic.h"
#include "../../../shared/fft_params.h"
#include "../../../shared/hostbuf.h"
#include "sqlite3.h"

#include <algorithm>
#include <fstream>
#include <hip/hip_runtime_api.h>
#include <iostream>
#include <memory>
#include <string>

#if __has_include(<filesystem>)
#include <filesystem>
#else
#include <experimental/filesystem>
namespace std
{
    namespace filesystem = experimental::filesystem;
}
#endif

typedef size_t default_hash_type;

struct hash_input
{
    hash_input(const fft_params& params, const bool use_ibuffer)
    {
        precision = params.precision;

        if(use_ibuffer)
        {
            buf_length   = params.ilength();
            buffer_sizes = params.ibuffer_sizes();
            buf_stride   = params.istride;
            buf_dist     = params.idist;
            buf_size     = params.isize;
            buf_type     = params.itype;
        }
        else
        {
            buf_length   = params.olength();
            buffer_sizes = params.obuffer_sizes();
            buf_stride   = params.ostride;
            buf_dist     = params.odist;
            buf_size     = params.osize;
            buf_type     = params.otype;
        }

        length = params.length;

        nbatch = params.nbatch;

        token = params.token();
    }

    ~hash_input() {}

    fft_precision precision;

    std::vector<size_t> buf_length;
    std::vector<size_t> buffer_sizes;
    std::vector<size_t> buf_stride;
    size_t              buf_dist;
    std::vector<size_t> buf_size;
    fft_array_type      buf_type;

    std::vector<size_t> length;

    size_t nbatch;

    std::string token;
};

template <typename Tint>
struct hash_output
{
    hash_output()
        : buffer_real(static_cast<Tint>(0))
        , buffer_imag(static_cast<Tint>(0))
    {
    }

    ~hash_output() {}

    bool operator==(const hash_output& rhs) const
    {
        return (buffer_real == rhs.buffer_real && buffer_imag == rhs.buffer_imag);
    }

    Tint buffer_real;
    Tint buffer_imag;
};

template <typename Tint = default_hash_type>
struct rocfft_test_run
{
    rocfft_test_run(Tint        ibuffer_hash_real_,
                    Tint        ibuffer_hash_imag_,
                    Tint        obuffer_hash_real_,
                    Tint        obuffer_hash_imag_,
                    std::string token_,
                    std::string runtime_version_,
                    std::string lib_version_,
                    std::string gpu_architecure_)
        : ibuffer_hash_real(ibuffer_hash_real_)
        , ibuffer_hash_imag(ibuffer_hash_imag_)
        , obuffer_hash_real(obuffer_hash_real_)
        , obuffer_hash_imag(obuffer_hash_imag_)
        , token(token_)
        , runtime_version(runtime_version_)
        , lib_version(lib_version_)
        , gpu_architecture(gpu_architecure_)
    {
    }

    static std::string get_create_rocfft_test_run_sql()
    {
        return "CREATE TABLE IF NOT EXISTS rocfft_test_run(ibuffer_hash_real TEXT NOT NULL, "
               "ibuffer_hash_imag TEXT NOT NULL, obuffer_hash_real TEXT NOT NULL, "
               "obuffer_hash_imag TEXT NOT NULL, token TEXT NOT NULL, runtime_version TEXT NOT "
               "NULL, lib_version TEXT NOT NULL, gpu_architecture TEXT NOT NULL); CREATE UNIQUE "
               "INDEX IF NOT EXISTS idx_unique_run ON rocfft_test_run(token, runtime_version, "
               "lib_version, gpu_architecture);";
    }

    static std::string get_match_sql()
    {
        return "SELECT ibuffer_hash_real, ibuffer_hash_imag, obuffer_hash_real, obuffer_hash_imag, "
               "token, runtime_version, lib_version, gpu_architecture FROM rocfft_test_run WHERE "
               "token = ? AND runtime_version = ? AND lib_version = ? AND gpu_architecture = ? ";
    }

    static std::string get_insert_sql()
    {
        return "INSERT INTO rocfft_test_run(ibuffer_hash_real, ibuffer_hash_imag, "
               "obuffer_hash_real, obuffer_hash_imag, token, runtime_version, lib_version, "
               "gpu_architecture) VALUES (?,?,?,?,?,?,?,?)";
    }

    void bind_insert_statement(sqlite3_stmt* stmt)
    {
        bind_ibuffer_hash_real(stmt, 1);
        bind_ibuffer_hash_imag(stmt, 2);
        bind_obuffer_hash_real(stmt, 3);
        bind_obuffer_hash_imag(stmt, 4);
        bind_token(stmt, 5);
        bind_runtime_version(stmt, 6);
        bind_lib_version(stmt, 7);
        bind_gpu_architecture(stmt, 8);
    }

    void bind_match_statement(sqlite3_stmt* stmt)
    {
        bind_token(stmt, 1);
        bind_runtime_version(stmt, 2);
        bind_lib_version(stmt, 3);
        bind_gpu_architecture(stmt, 4);
    }

    void update(sqlite3_stmt* stmt)
    {
        for(int col = 0; col < sqlite3_column_count(stmt); ++col)
        {
            auto col_name = sqlite3_column_name(stmt, col);
            auto col_value
                = std::string(reinterpret_cast<const char*>(sqlite3_column_text(stmt, col)));

            if(strcmp(col_name, "ibuffer_hash_real") == 0)
                update_ibuffer_hash_real(col_value);

            if(strcmp(col_name, "ibuffer_hash_imag") == 0)
                update_ibuffer_hash_imag(col_value);

            if(strcmp(col_name, "obuffer_hash_real") == 0)
                update_obuffer_hash_real(col_value);

            if(strcmp(col_name, "obuffer_hash_imag") == 0)
                update_obuffer_hash_imag(col_value);

            if(strcmp(col_name, "token") == 0)
                update_token(col_value);

            if(strcmp(col_name, "runtime_version") == 0)
                update_runtime_version(col_value);

            if(strcmp(col_name, "lib_version") == 0)
                update_lib_version(col_value);

            if(strcmp(col_name, "gpu_architecture") == 0)
                update_gpu_architecture(col_value);
        }
    }

    Tint ibuffer_hash_real;
    Tint ibuffer_hash_imag;
    Tint obuffer_hash_real;
    Tint obuffer_hash_imag;

private:
    std::string token;
    std::string runtime_version;
    std::string lib_version;
    std::string gpu_architecture;

    std::string get_ibuffer_hash_real() const
    {
        return std::to_string(ibuffer_hash_real);
    }

    std::string get_ibuffer_hash_imag() const
    {
        return std::to_string(ibuffer_hash_imag);
    }

    std::string get_obuffer_hash_real() const
    {
        return std::to_string(obuffer_hash_real);
    }

    std::string get_obuffer_hash_imag() const
    {
        return std::to_string(obuffer_hash_imag);
    }

    std::string get_token() const
    {
        return token;
    }

    std::string get_runtime_version() const
    {
        return runtime_version;
    }

    std::string get_lib_version() const
    {
        return lib_version;
    }

    std::string get_gpu_architecture() const
    {
        return gpu_architecture;
    }

    void bind_ibuffer_hash_real(sqlite3_stmt* stmt, int index)
    {
        auto ret
            = sqlite3_bind_text(stmt, index, get_ibuffer_hash_real().c_str(), -1, SQLITE_TRANSIENT);
        if(ret != SQLITE_OK)
            throw std::runtime_error(
                std::string("Error binding ibuffer_hash_real field in insert statement"));
    }

    void bind_ibuffer_hash_imag(sqlite3_stmt* stmt, int index)
    {
        auto ret
            = sqlite3_bind_text(stmt, index, get_ibuffer_hash_imag().c_str(), -1, SQLITE_TRANSIENT);
        if(ret != SQLITE_OK)
            throw std::runtime_error(
                std::string("Error binding ibuffer_hash_imag field in insert statement"));
    }

    void bind_obuffer_hash_real(sqlite3_stmt* stmt, int index)
    {
        auto ret
            = sqlite3_bind_text(stmt, index, get_obuffer_hash_real().c_str(), -1, SQLITE_TRANSIENT);
        if(ret != SQLITE_OK)
            throw std::runtime_error(
                std::string("Error binding obuffer_hash_real field in insert statement"));
    }

    void bind_obuffer_hash_imag(sqlite3_stmt* stmt, int index)
    {
        auto ret
            = sqlite3_bind_text(stmt, index, get_obuffer_hash_imag().c_str(), -1, SQLITE_TRANSIENT);
        if(ret != SQLITE_OK)
            throw std::runtime_error(
                std::string("Error binding obuffer_hash_imag field in insert statement"));
    }

    void bind_token(sqlite3_stmt* stmt, int index)
    {
        auto ret = sqlite3_bind_text(stmt, index, get_token().c_str(), -1, SQLITE_TRANSIENT);
        if(ret != SQLITE_OK)
            throw std::runtime_error(std::string("Error binding token field in insert statement"));
    }

    void bind_runtime_version(sqlite3_stmt* stmt, int index)
    {
        auto ret
            = sqlite3_bind_text(stmt, index, get_runtime_version().c_str(), -1, SQLITE_TRANSIENT);
        if(ret != SQLITE_OK)
            throw std::runtime_error(
                std::string("Error binding runtime_version field in insert statement"));
    }

    void bind_lib_version(sqlite3_stmt* stmt, int index)
    {
        auto ret = sqlite3_bind_text(stmt, index, get_lib_version().c_str(), -1, SQLITE_TRANSIENT);
        if(ret != SQLITE_OK)
            throw std::runtime_error(
                std::string("Error binding lib_version field in insert statement"));
    }

    void bind_gpu_architecture(sqlite3_stmt* stmt, int index)
    {
        auto ret
            = sqlite3_bind_text(stmt, index, get_gpu_architecture().c_str(), -1, SQLITE_TRANSIENT);
        if(ret != SQLITE_OK)
            throw std::runtime_error(
                std::string("Error binding gpu_architecture field in insert statement"));
    }

    void update_ibuffer_hash_real(const std::string& value)
    {
        std::stringstream stream(value);
        stream >> ibuffer_hash_real;
    }

    void update_ibuffer_hash_imag(const std::string& value)
    {
        std::stringstream stream(value);
        stream >> ibuffer_hash_imag;
    }

    void update_obuffer_hash_real(const std::string& value)
    {
        std::stringstream stream(value);
        stream >> obuffer_hash_real;
    }

    void update_obuffer_hash_imag(const std::string& value)
    {
        std::stringstream stream(value);
        stream >> obuffer_hash_imag;
    }

    void update_token(const std::string& value)
    {
        token = value;
    }

    void update_runtime_version(const std::string& value)
    {
        runtime_version = value;
    }

    void update_lib_version(const std::string& value)
    {
        lib_version = value;
    }

    void update_gpu_architecture(const std::string& value)
    {
        gpu_architecture = value;
    }
};

template <typename Tint>
inline rocfft_test_run<Tint> get_rocfft_test_run(const hash_output<Tint>& ibuffer_hash,
                                                 const hash_output<Tint>& obuffer_hash,
                                                 const std::string&       token)
{
    hipDeviceProp_t device_prop;

    if(hipGetDeviceProperties(&device_prop, 0) != hipSuccess)
        throw std::runtime_error("hipGetDeviceProperties failure");

    auto gpu_arch = std::string(device_prop.gcnArchName);

    auto ver_sep = std::string(".");
    auto runtime_ver
        = std::to_string(HIP_VERSION_MAJOR) + ver_sep + std::to_string(HIP_VERSION_MINOR);

    const size_t ver_size = 256;
    char         lib_version[ver_size];
    rocfft_get_version_string(lib_version, ver_size);
    auto lib_ver_full = std::string(lib_version);

    auto idx_maj = lib_ver_full.find(ver_sep);
    auto idx_min = lib_ver_full.find(ver_sep, idx_maj + 1);
    auto idx_rev = lib_ver_full.find(ver_sep, idx_min + 1);
    auto ver_maj = lib_ver_full.substr(0, idx_maj);
    auto ver_min = lib_ver_full.substr(idx_maj + 1, idx_min - idx_maj - 1);
    auto ver_rev = lib_ver_full.substr(idx_min + 1, idx_rev - idx_min - 1);

    auto lib_ver = ver_maj + ver_sep + ver_min + ver_sep + ver_rev;

    return rocfft_test_run<Tint>(ibuffer_hash.buffer_real,
                                 ibuffer_hash.buffer_imag,
                                 obuffer_hash.buffer_real,
                                 obuffer_hash.buffer_imag,
                                 token,
                                 runtime_ver,
                                 lib_ver,
                                 gpu_arch);
}

class fft_hash_db
{
public:
    fft_hash_db(std::string db_path)
        : ret(SQLITE_OK)
        , db_connection(nullptr)
        , begin_stmt(nullptr)
        , end_stmt(nullptr)
        , insert_stmt(nullptr)
        , match_stmt(nullptr)
    {
        ret = sqlite3_open(db_path.c_str(), &db_connection);
        if(ret != SQLITE_OK)
            throw std::runtime_error(std::string("Cannot open repro-db: ") + db_path);

        // Access to a database file may occur in parallel.
        // Increase default sqlite timeout, so diferent process
        // can wait for one another.
        sqlite3_busy_timeout(db_connection, 30000);

        // Set sqlite3 engine to WAL mode to avoid potential deadlocks with multiple
        // concurrent processes (if a deadlock occurs, the busy timeout is not honored).
        ret = sqlite3_exec(db_connection, "PRAGMA journal_mode = WAL", nullptr, nullptr, nullptr);
        if(ret != SQLITE_OK)
            throw std::runtime_error("Error setting WAL mode: "
                                     + std::string(sqlite3_errmsg(db_connection)));

        ret = sqlite3_exec(db_connection,
                           rocfft_test_run<>::get_create_rocfft_test_run_sql().c_str(),
                           nullptr,
                           nullptr,
                           nullptr);
        if(ret != SQLITE_OK)
            throw std::runtime_error("Error creating table: "
                                     + std::string(sqlite3_errmsg(db_connection)));

        prepare_begin_end_stmts();
        prepare_match_stmt();
        prepare_insert_stmt();
    }

    ~fft_hash_db()
    {
        sqlite3_finalize(begin_stmt);
        sqlite3_finalize(end_stmt);
        sqlite3_finalize(match_stmt);
        sqlite3_finalize(insert_stmt);
        sqlite3_close(db_connection);
    }

    template <typename Tint>
    void check_hash_valid(const hash_output<Tint>& ibuffer_hash,
                          const hash_output<Tint>& obuffer_hash,
                          const std::string&       token,
                          bool&                    hash_entry_found,
                          bool&                    hash_valid)
    {
        hash_valid = true;

        auto test_run = get_rocfft_test_run<Tint>(ibuffer_hash, obuffer_hash, token);

        begin_transaction();

        hash_entry_found = check_match(&test_run);

        if(hash_entry_found)
            hash_valid = (test_run.ibuffer_hash_real == ibuffer_hash.buffer_real
                          && test_run.ibuffer_hash_imag == ibuffer_hash.buffer_imag
                          && test_run.obuffer_hash_real == obuffer_hash.buffer_real
                          && test_run.obuffer_hash_imag == obuffer_hash.buffer_imag)
                             ? true
                             : false;
        else
            insert(&test_run);

        end_transaction();
    }

private:
    void prepare_begin_end_stmts()
    {
        auto begin_sql = std::string("BEGIN TRANSACTION;");

        ret = sqlite3_prepare_v2(db_connection, begin_sql.c_str(), -1, &begin_stmt, nullptr);
        if(ret != SQLITE_OK)
            throw std::runtime_error("Cannot prepare begin statement: "
                                     + std::string(sqlite3_errmsg(db_connection)));

        auto end_sql = std::string("END TRANSACTION;");

        ret = sqlite3_prepare_v2(db_connection, end_sql.c_str(), -1, &end_stmt, nullptr);
        if(ret != SQLITE_OK)
            throw std::runtime_error("Cannot prepare end statement: "
                                     + std::string(sqlite3_errmsg(db_connection)));
    }

    void prepare_match_stmt()
    {
        auto match_sql = rocfft_test_run<>::get_match_sql();

        ret = sqlite3_prepare_v2(db_connection, match_sql.c_str(), -1, &match_stmt, nullptr);
        if(ret != SQLITE_OK)
            throw std::runtime_error("Cannot prepare match statement: "
                                     + std::string(sqlite3_errmsg(db_connection)));
    }

    void prepare_insert_stmt()
    {
        auto insert_sql = rocfft_test_run<>::get_insert_sql();

        ret = sqlite3_prepare_v2(db_connection, insert_sql.c_str(), -1, &insert_stmt, nullptr);
        if(ret != SQLITE_OK)
            throw std::runtime_error("Cannot prepare insert statement: "
                                     + std::string(sqlite3_errmsg(db_connection)));
    }

    void begin_transaction()
    {
        ret = sqlite3_step(begin_stmt);
        if(ret != SQLITE_DONE)
            throw std::runtime_error(std::string("Error executing begin statement: ")
                                     + std::string(sqlite3_errmsg(db_connection)));
    }

    void end_transaction()
    {
        ret = sqlite3_step(end_stmt);
        if(ret != SQLITE_DONE)
            throw std::runtime_error(std::string("Error executing end statement: ")
                                     + std::string(sqlite3_errmsg(db_connection)));
    }

    template <typename Tint>
    bool check_match(rocfft_test_run<Tint>* entry)
    {
        sqlite3_reset(match_stmt);

        entry->bind_match_statement(match_stmt);

        size_t match_count = 0;
        while((ret = sqlite3_step(match_stmt)) == SQLITE_ROW)
        {
            entry->update(match_stmt);
            match_count++;
        }

        // There can only be one result in this query
        if(match_count > 1)
            throw std::runtime_error("Corrupted database");

        if(ret != SQLITE_DONE)
            throw std::runtime_error(std::string("Error executing select statement: ")
                                     + std::string(sqlite3_errmsg(db_connection)));

        return match_count;
    }

    template <typename Tint>
    void insert(rocfft_test_run<Tint>* entry)
    {
        sqlite3_reset(insert_stmt);

        entry->bind_insert_statement(insert_stmt);

        ret = sqlite3_step(insert_stmt);
        if(ret != SQLITE_DONE)
            throw std::runtime_error(std::string("Error executing insert statement: ")
                                     + std::string(sqlite3_errmsg(db_connection)));
    }

    int           ret;
    sqlite3*      db_connection;
    sqlite3_stmt* begin_stmt;
    sqlite3_stmt* end_stmt;
    sqlite3_stmt* insert_stmt;
    sqlite3_stmt* match_stmt;
};

static inline double get_weight(const size_t counter, const size_t max_counter)
{
    return (static_cast<double>(counter) / static_cast<double>(max_counter));
}

template <typename Tint>
static inline void hash_value(Tint&           hash_value,
                              const size_t    counter,
                              const size_t    max_counter,
                              const _Float16& input_value)
{
    auto weight = get_weight(counter, max_counter);
    hash_value += std::hash<float>{}(weight * input_value);
}
template <typename Tint>
static inline void hash_value(Tint&        hash_value,
                              const size_t counter,
                              const size_t max_counter,
                              const float& input_value)
{
    auto weight = get_weight(counter, max_counter);
    hash_value += std::hash<float>{}(weight * input_value);
}
template <typename Tint>
static inline void hash_value(Tint&         hash_value,
                              const size_t  counter,
                              const size_t  max_counter,
                              const double& input_value)
{
    auto weight = get_weight(counter, max_counter);
    hash_value  = std::hash<double>{}(weight * input_value) + hash_value;
}

template <typename Tint>
static inline size_t get_max_counter(const Tint& whole_length, const size_t nbatch)
{
    return static_cast<size_t>(count_iters(whole_length) * nbatch);
}

template <typename T1>
static inline T1 get_unit_value(const T1& val)
{
    return static_cast<T1>(1);
}
template <typename T1>
static inline std::tuple<T1, T1> get_unit_value(const std::tuple<T1, T1>& val)
{
    return std::make_tuple(static_cast<T1>(1), static_cast<T1>(1));
}
template <typename T1>
static inline std::tuple<T1, T1, T1> get_unit_value(const std::tuple<T1, T1, T1>& val)
{
    return std::make_tuple(static_cast<T1>(1), static_cast<T1>(1), static_cast<T1>(1));
}

template <typename T1>
static inline void sum_hash_from_partitions(const std::vector<T1>& partition_hash, T1& hash_value)
{
    hash_value = 0;
    hash_value = std::accumulate(partition_hash.begin(), partition_hash.end(), hash_value);
}

template <typename Tfloat, typename Tint1, typename Tint2>
static inline void compute_real_buffer_hash(const std::vector<hostbuf>& ibuffer,
                                            const Tint1&                whole_length,
                                            const Tint1&                whole_stride,
                                            const size_t                idist,
                                            const size_t                nbatch,
                                            Tint2&                      hash_real,
                                            Tint2&                      hash_imag)
{
    auto unit_stride = get_unit_value(whole_stride);

    size_t max_counter = get_max_counter<Tint1>(whole_length, nbatch);

    auto   idata      = (Tfloat*)ibuffer[0].data();
    size_t i_base     = 0;
    auto   partitions = partition_rowmajor(whole_length);

    std::vector<Tint2> partition_hash_real(partitions.size(), 0);

    for(unsigned int b = 0; b < nbatch; b++, i_base += idist)
    {
#pragma omp parallel for num_threads(partitions.size())
        for(size_t part = 0; part < partitions.size(); ++part)
        {
            auto       index  = partitions[part].first;
            const auto length = partitions[part].second;
            do
            {
                const auto i       = compute_index(index, whole_stride, i_base);
                const auto counter = compute_index(index, unit_stride, i_base) + 1;

                hash_value<Tint2>(partition_hash_real[part], counter, max_counter, idata[i]);
            } while(increment_rowmajor(index, length));
        }
    }

    sum_hash_from_partitions(partition_hash_real, hash_real);
    hash_imag = 0;
}

template <typename Tfloat, typename Tint1, typename Tint2>
static inline void compute_planar_buffer_hash(const std::vector<hostbuf>& ibuffer,
                                              const Tint1&                whole_length,
                                              const Tint1&                whole_stride,
                                              const size_t                idist,
                                              const size_t                nbatch,
                                              Tint2&                      hash_real,
                                              Tint2&                      hash_imag)
{
    auto unit_stride = get_unit_value(whole_stride);

    size_t max_counter = get_max_counter<Tint1>(whole_length, nbatch);

    auto   ireal      = (Tfloat*)ibuffer[0].data();
    auto   iimag      = (Tfloat*)ibuffer[1].data();
    size_t i_base     = 0;
    auto   partitions = partition_rowmajor(whole_length);

    std::vector<Tint2> partition_hash_real(partitions.size(), 0);
    std::vector<Tint2> partition_hash_imag(partitions.size(), 0);

    for(unsigned int b = 0; b < nbatch; b++, i_base += idist)
    {
#pragma omp parallel for num_threads(partitions.size())
        for(size_t part = 0; part < partitions.size(); ++part)
        {
            auto       index  = partitions[part].first;
            const auto length = partitions[part].second;
            do
            {
                const auto i       = compute_index(index, whole_stride, i_base);
                const auto counter = compute_index(index, unit_stride, i_base) + 1;

                hash_value<Tint2>(partition_hash_real[part], counter, max_counter, ireal[i]);
                hash_value<Tint2>(partition_hash_imag[part], counter, max_counter, iimag[i]);
            } while(increment_rowmajor(index, length));
        }
    }

    sum_hash_from_partitions(partition_hash_real, hash_real);
    sum_hash_from_partitions(partition_hash_imag, hash_imag);
}

template <typename Tfloat, typename Tint1, typename Tint2>
static inline void compute_interleaved_buffer_hash(const std::vector<hostbuf>& ibuffer,
                                                   const Tint1&                whole_length,
                                                   const Tint1&                whole_stride,
                                                   const size_t                idist,
                                                   const size_t                nbatch,
                                                   Tint2&                      hash_real,
                                                   Tint2&                      hash_imag)
{
    auto unit_stride = get_unit_value(whole_stride);

    size_t max_counter = get_max_counter<Tint1>(whole_length, nbatch);

    auto   idata      = (std::complex<Tfloat>*)ibuffer[0].data();
    size_t i_base     = 0;
    auto   partitions = partition_rowmajor(whole_length);

    std::vector<Tint2> partition_hash_real(partitions.size(), 0);
    std::vector<Tint2> partition_hash_imag(partitions.size(), 0);

    for(unsigned int b = 0; b < nbatch; b++, i_base += idist)
    {
#pragma omp parallel for num_threads(partitions.size())
        for(size_t part = 0; part < partitions.size(); ++part)
        {
            auto       index  = partitions[part].first;
            const auto length = partitions[part].second;
            do
            {
                const auto i       = compute_index(index, whole_stride, i_base);
                const auto counter = compute_index(index, unit_stride, i_base) + 1;

                hash_value<Tint2>(partition_hash_real[part], counter, max_counter, idata[i].real());
                hash_value<Tint2>(partition_hash_imag[part], counter, max_counter, idata[i].imag());
            } while(increment_rowmajor(index, length));
        }
    }

    sum_hash_from_partitions(partition_hash_real, hash_real);
    sum_hash_from_partitions(partition_hash_imag, hash_imag);
}

template <typename Tfloat, typename Tint1, typename Tint2>
static inline void compute_buffer_hash(const std::vector<hostbuf>& ibuffer,
                                       const fft_array_type        itype,
                                       const Tint1&                whole_length,
                                       const Tint1&                whole_stride,
                                       const size_t                idist,
                                       const size_t                nbatch,
                                       Tint2&                      hash_real,
                                       Tint2&                      hash_imag)
{
    switch(itype)
    {
    case fft_array_type_complex_interleaved:
    case fft_array_type_hermitian_interleaved:
        compute_interleaved_buffer_hash<Tfloat, Tint1, Tint2>(
            ibuffer, whole_length, whole_stride, idist, nbatch, hash_real, hash_imag);
        break;
    case fft_array_type_complex_planar:
    case fft_array_type_hermitian_planar:
        compute_planar_buffer_hash<Tfloat, Tint1, Tint2>(
            ibuffer, whole_length, whole_stride, idist, nbatch, hash_real, hash_imag);
        break;
    case fft_array_type_real:
        compute_real_buffer_hash<Tfloat, Tint1, Tint2>(
            ibuffer, whole_length, whole_stride, idist, nbatch, hash_real, hash_imag);
        break;
    default:
        throw std::runtime_error("Input layout format not yet supported");
    }
}

template <typename Tfloat, typename Tint>
static inline void compute_buffer_hash(const std::vector<hostbuf>& ibuffer,
                                       const fft_array_type        itype,
                                       const std::vector<size_t>&  length,
                                       const std::vector<size_t>&  ilength,
                                       const std::vector<size_t>&  istride,
                                       const size_t                idist,
                                       const size_t                nbatch,
                                       Tint&                       hash_real,
                                       Tint&                       hash_imag)
{
    switch(length.size())
    {
    case 1:
        compute_buffer_hash<Tfloat, size_t, Tint>(
            ibuffer, itype, ilength[0], istride[0], idist, nbatch, hash_real, hash_imag);
        break;
    case 2:
        compute_buffer_hash<Tfloat, std::tuple<size_t, size_t>, Tint>(
            ibuffer,
            itype,
            std::make_tuple(ilength[0], ilength[1]),
            std::make_tuple(istride[0], istride[1]),
            idist,
            nbatch,
            hash_real,
            hash_imag);
        break;
    case 3:
        compute_buffer_hash<Tfloat, std::tuple<size_t, size_t, size_t>, Tint>(
            ibuffer,
            itype,
            std::make_tuple(ilength[0], ilength[1], ilength[2]),
            std::make_tuple(istride[0], istride[1], istride[2]),
            idist,
            nbatch,
            hash_real,
            hash_imag);
        break;
    default:
        abort();
    }
}

template <typename Tint>
static inline void compute_hash(const std::vector<hostbuf>& buffer,
                                const hash_input&           hash_in,
                                hash_output<Tint>&          hash_out)
{

    auto bsizes  = hash_in.buffer_sizes;
    auto blength = hash_in.buf_length;
    auto bstride = hash_in.buf_stride;
    auto bdist   = hash_in.buf_dist;
    auto btype   = hash_in.buf_type;
    auto bsize   = hash_in.buf_size;

    switch(hash_in.precision)
    {
    case fft_precision_half:
        compute_buffer_hash<_Float16>(buffer,
                                      btype,
                                      hash_in.length,
                                      blength,
                                      bstride,
                                      bdist,
                                      hash_in.nbatch,
                                      hash_out.buffer_real,
                                      hash_out.buffer_imag);
        break;
    case fft_precision_double:
        compute_buffer_hash<double>(buffer,
                                    btype,
                                    hash_in.length,
                                    blength,
                                    bstride,
                                    bdist,
                                    hash_in.nbatch,
                                    hash_out.buffer_real,
                                    hash_out.buffer_imag);
        break;
    case fft_precision_single:
        compute_buffer_hash<float>(buffer,
                                   btype,
                                   hash_in.length,
                                   blength,
                                   bstride,
                                   bdist,
                                   hash_in.nbatch,
                                   hash_out.buffer_real,
                                   hash_out.buffer_imag);
        break;
    default:
        abort();
    }
}

#endif // FFT_HASH_H