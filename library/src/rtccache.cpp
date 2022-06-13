// Copyright (C) 2021 - 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include "../../shared/environment.h"

#include "logging.h"
#include "rtccache.h"
#include "sqlite3.h"

namespace fs = std::filesystem;

std::unique_ptr<RTCCache> RTCCache::single;

// Get list of candidate paths to RTC cache DB, in decreasing order
// of preference.
static std::vector<fs::path> rtccache_db_paths()
{
    // use user-defined cache path if present
    std::vector<fs::path> paths;
    auto                  env_path = rocfft_getenv("ROCFFT_RTC_CACHE_PATH");

    if(!env_path.empty())
        paths.push_back(env_path);
    else
    {
        static const char* default_cache_filename = "rocfft_kernel_cache.db";

        // try standard cache dirs
#ifdef WIN32
        auto localappdata = rocfft_getenv("LOCALAPPDATA");
        if(!localappdata.empty())
        {
            auto dir = fs::path(localappdata) / "rocFFT";
            fs::create_directories(dir);
            paths.push_back(dir / default_cache_filename);
        }
#else
        auto xdg_cache_home = rocfft_getenv("XDG_CACHE_HOME");
        if(!xdg_cache_home.empty())
        {
            auto dir = fs::path(xdg_cache_home) / "rocFFT";
            fs::create_directories(dir);
            paths.push_back(dir / default_cache_filename);
        }
#endif

        auto home_path = rocfft_getenv("HOME");
        // try persistent home directory location if no cache dir
        if(paths.empty() && !home_path.empty())
        {
            auto dir = fs::path(home_path) / ".cache" / "rocFFT";
            fs::create_directories(dir);
            paths.push_back(dir / default_cache_filename);
        }

        // otherwise, temp directory, which you'd expect to be less
        // persistent but still usable
        paths.push_back(fs::temp_directory_path() / default_cache_filename);
    }

    // finally, fall back to in-memory db if all else fails
    paths.push_back({});
    return paths;
}

static sqlite3_stmt_ptr prepare_stmt(sqlite3_ptr& db, const char* sql)
{
    sqlite3_stmt* stmt = nullptr;
    if(sqlite3_prepare_v2(db.get(), sql, -1, &stmt, nullptr) == SQLITE_OK)
        return sqlite3_stmt_ptr(stmt);
    throw std::runtime_error(std::string("sqlite_prepare_v2 failed: ") + sqlite3_errmsg(db.get()));
}

sqlite3_ptr RTCCache::connect_db(const fs::path& path)
{
    sqlite3* db_raw = nullptr;
    int      flags  = SQLITE_OPEN_FULLMUTEX | SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE;
    if(path.empty())
    {
        // open in-memory
        flags |= SQLITE_OPEN_MEMORY;
    }
    if(sqlite3_open_v2(path.string().c_str(), &db_raw, flags, nullptr) != SQLITE_OK)
        return nullptr;

    sqlite3_ptr db(db_raw);

    // we can potentially want to write a bunch of kernels in
    // parallel (when doing mass compilation).  set a bigger busy
    // timeout (5s) so that concurrent modifications will wait for one
    // another
    sqlite3_busy_timeout(db_raw, 5000);

    // create the default table
    auto create = prepare_stmt(db,
                               "CREATE TABLE IF NOT EXISTS cache_v1 ("
                               "  kernel_name TEXT NOT NULL,"
                               "  arch TEXT NOT NULL,"
                               "  hip_version INTEGER NOT NULL,"
                               "  generator_sum BLOB NOT NULL,"
                               "  code BLOB NOT NULL,"
                               "  timestamp INTEGER NOT NULL,"
                               "  PRIMARY KEY ("
                               "      kernel_name, arch, hip_version, generator_sum"
                               "      ))");
    if(sqlite3_step(create.get()) != SQLITE_DONE)
        return nullptr;

    return db;
}

RTCCache::RTCCache()
{
    auto paths = rtccache_db_paths();
    for(const auto& p : paths)
    {
        db = connect_db(p);
        if(db)
            break;
    }

    if(!db)
        throw std::runtime_error("failed to open cache");

    // prepare get/store statements once so they can be called many
    // times
    get_stmt = prepare_stmt(db,
                            "SELECT code "
                            "FROM cache_v1 "
                            "WHERE"
                            "  kernel_name = :kernel_name "
                            "  AND arch = :arch "
                            "  AND hip_version = :hip_version "
                            "  AND generator_sum = :generator_sum ");

    store_stmt = prepare_stmt(db,
                              "INSERT OR REPLACE INTO cache_v1 ("
                              "    kernel_name,"
                              "    arch,"
                              "    hip_version,"
                              "    generator_sum,"
                              "    code,"
                              "    timestamp"
                              ")"
                              "VALUES ("
                              "    :kernel_name,"
                              "    :arch,"
                              "    :hip_version,"
                              "    :generator_sum,"
                              "    :code,"
                              "    CAST(STRFTIME('%s','now') AS INTEGER)"
                              ")");
}

std::vector<char> RTCCache::get_code_object(const std::string&       kernel_name,
                                            const std::string&       gpu_arch,
                                            int                      hip_version,
                                            const std::vector<char>& generator_sum)
{
    std::vector<char> code;

    // allow env variable to disable reads
    if(!rocfft_getenv("ROCFFT_RTC_CACHE_READ_DISABLE").empty())
        return code;

    std::lock_guard<std::mutex> lock(get_mutex);

    auto s = get_stmt.get();
    sqlite3_reset(s);

    // bind arguments to the query and execute
    if(sqlite3_bind_text(s, 1, kernel_name.c_str(), kernel_name.size(), SQLITE_TRANSIENT)
           != SQLITE_OK
       || sqlite3_bind_text(s, 2, gpu_arch.c_str(), gpu_arch.size(), SQLITE_TRANSIENT) != SQLITE_OK
       || sqlite3_bind_int(s, 3, hip_version) != SQLITE_OK
       || sqlite3_bind_blob(s, 4, generator_sum.data(), generator_sum.size(), SQLITE_TRANSIENT)
              != SQLITE_OK)
    {
        throw std::runtime_error(std::string("get_code_object bind: ") + sqlite3_errmsg(db.get()));
    }
    if(sqlite3_step(s) == SQLITE_ROW)
    {
        // cache hit, get the value out
        int         nbytes = sqlite3_column_bytes(s, 0);
        const char* data   = static_cast<const char*>(sqlite3_column_blob(s, 0));
        std::copy(data, data + nbytes, std::back_inserter(code));
    }
    sqlite3_reset(s);
    return code;
}

void RTCCache::store_code_object(const std::string&       kernel_name,
                                 const std::string&       gpu_arch,
                                 int                      hip_version,
                                 const std::vector<char>& generator_sum,
                                 const std::vector<char>& code)
{
    // allow env variable to disable writes
    if(!rocfft_getenv("ROCFFT_RTC_CACHE_WRITE_DISABLE").empty())
        return;

    std::lock_guard<std::mutex> lock(store_mutex);

    auto s = store_stmt.get();
    sqlite3_reset(s);

    // bind arguments to the query and execute
    if(sqlite3_bind_text(s, 1, kernel_name.c_str(), kernel_name.size(), SQLITE_TRANSIENT)
           != SQLITE_OK
       || sqlite3_bind_text(s, 2, gpu_arch.c_str(), gpu_arch.size(), SQLITE_TRANSIENT) != SQLITE_OK
       || sqlite3_bind_int(s, 3, hip_version) != SQLITE_OK
       || sqlite3_bind_blob(s, 4, generator_sum.data(), generator_sum.size(), SQLITE_TRANSIENT)
              != SQLITE_OK
       || sqlite3_bind_blob(s, 5, code.data(), code.size(), SQLITE_TRANSIENT))
    {
        throw std::runtime_error(std::string("store_code_object bind: ")
                                 + sqlite3_errmsg(db.get()));
    }
    if(sqlite3_step(s) != SQLITE_DONE)
    {
        std::cerr << "Error: failed to store code object for " << kernel_name << std::endl;
        // some kind of problem storing the row?  log it
        if(LOG_RTC_ENABLED())
            (*LogSingleton::GetInstance().GetRTCOS())
                << "Error: failed to store code object for " << kernel_name << std::flush;
    }
    sqlite3_reset(s);
}

rocfft_status RTCCache::serialize(void** buffer, size_t* buffer_len_bytes)
{
    sqlite3_int64 db_size = 0;
    auto          ptr     = sqlite3_serialize(db.get(), "main", &db_size, 0);
    if(ptr)
    {
        *buffer           = ptr;
        *buffer_len_bytes = db_size;
        return rocfft_status_success;
    }
    return rocfft_status_failure;
}

void RTCCache::serialize_free(void* buffer)
{
    sqlite3_free(buffer);
}

rocfft_status RTCCache::deserialize(const void* buffer, size_t buffer_len_bytes)
{
    std::lock_guard<std::mutex> lock(deserialize_mutex);

    // attach an empty database named "deserialized"
    sqlite3_exec(db.get(), "ATTACH DATABASE ':memory:' AS deserialized", nullptr, nullptr, nullptr);
    // the attach might fail if somehow this is our second
    // deserialize and the db already existed.  later steps will
    // notice this, so we can skip this error check

    // sqlite's API is prepared to write to the pointer, but we tell
    // it to be read-only
    auto buffer_mut = const_cast<unsigned char*>(static_cast<const unsigned char*>(buffer));

    int sql_err = sqlite3_deserialize(db.get(),
                                      "deserialized",
                                      buffer_mut,
                                      buffer_len_bytes,
                                      buffer_len_bytes,
                                      SQLITE_DESERIALIZE_READONLY);
    if(sql_err != SQLITE_OK)
        return rocfft_status_failure;

    // now the deserialized db is in memory.  run an additive query to
    // update the real db with the temp contents.
    sql_err           = sqlite3_exec(db.get(),
                           "INSERT OR REPLACE INTO cache_v1 ("
                           "    kernel_name,"
                           "    arch,"
                           "    hip_version,"
                           "    generator_sum,"
                           "    timestamp,"
                           "    code"
                           ")"
                           "SELECT"
                           "    kernel_name,"
                           "    arch,"
                           "    hip_version,"
                           "    generator_sum,"
                           "    timestamp,"
                           "    code "
                           "FROM deserialized.cache_v1",
                           nullptr,
                           nullptr,
                           nullptr);
    rocfft_status ret = sql_err == SQLITE_OK ? rocfft_status_success : rocfft_status_failure;

    // detach the temp db
    sqlite3_exec(db.get(), "DETACH DATABASE deserialized", nullptr, nullptr, nullptr);

    return ret;
}
