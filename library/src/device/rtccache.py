import sqlite3
import zlib
import os
import hashlib
import pickle
import tempfile

def init_generator(generator_code):
    '''Initialize the generator and return a checksum of the generator code.

    The checksum is used by the kernel cache to tell whether the
    generator has changed, which would invalidate the contents of the
    cache.  Returns bytes containing the checksum.
    '''

    return hashlib.sha256(generator_code.encode('utf-8')).digest()

def cache_db_paths(env_cache_path):
    '''Get list of candidate paths to RTC cache DB, in decreasing order
       of preference.
    '''

    default_cache_filename = 'rocfft_kernel_cache.db'

    paths = []
    if env_cache_path is not None:
        paths.append(env_cache_path)
    else:
        # come up with other candidate locations
        try:
            # try persistent home directory location
            cachedir = os.path.join(os.environ['HOME'], '.cache', 'rocFFT')
            os.makedirs(cachedir, exist_ok=True)
            paths.append(os.path.join(cachedir, default_cache_filename))
        except:
            pass

        # otherwise, temp directory, which you'd expect to be less
        # persistent but still usable
        paths.append(os.path.join(tempfile.gettempdir(), default_cache_filename))

    # finally, fall back to in-memory db if all else fails
    paths.append(':memory:')
    return paths

def connect_db(path):
    '''Connect to the cache DB using a given path

    Returns None if the DB could not be opened.
    '''
    try:
        db = sqlite3.connect(path)
        db.text_factory = bytes
        db.execute(
            '''
            CREATE TABLE IF NOT EXISTS stockham_v1 (
              kernel_name TEXT NOT NULL,
              arch TEXT NOT NULL,
              hip_version INTEGER NOT NULL,
              generator_sum BLOB NOT NULL,
              code BLOB NOT NULL,
              PRIMARY KEY (
                  kernel_name, arch, hip_version, generator_sum
                  )
             );
            ''')
        return db
    # just return None on error
    except:
        pass

def open_db(env_cache_path):
    '''Open the RTC cache DB
    '''

    cache_paths = cache_db_paths(env_cache_path)
    for cache_path in cache_paths:
        db = connect_db(cache_path)

        if db is not None:
            return db

def get_code_object(db, specs):
    '''Retrieve a code object from the RTC cache DB.

    Returns None if the DB is not open or a kernel with the provided
    specs was not found.  May throw an exception on error.

    Returns code object on success.
    '''
    if 'ROCFFT_RTC_CACHE_READ_DISABLE' in os.environ:
        return
    db_params = {
        'kernel_name': specs['kernel_name'],
        'arch': specs['arch'],
        'hip_version': specs['hip_version'],
        'generator_sum': specs['generator_sum'],
    }
    for row in db.execute(
    '''
        SELECT code
        FROM stockham_v1
        WHERE
          kernel_name = :kernel_name
          AND arch = :arch
          AND hip_version = :hip_version
          AND generator_sum = :generator_sum
        ''', db_params):
        return zlib.decompress(row[0])

def store_code_object(db, specs, code):
    '''Put a code object into the RTC cache DB.

    Normally returns None, but may throw an exception on error.
    '''
    if 'ROCFFT_RTC_CACHE_WRITE_DISABLE' in os.environ:
        return
    db_params = {
        'kernel_name': specs['kernel_name'],
        'arch': specs['arch'],
        'hip_version': specs['hip_version'],
        'generator_sum': specs['generator_sum'],
        'code': zlib.compress(code),
    }
    db.execute('''
        INSERT OR REPLACE INTO stockham_v1 (
            kernel_name,
            arch,
            hip_version,
            generator_sum,
            code
        )
        VALUES (
            :kernel_name,
            :arch,
            :hip_version,
            :generator_sum,
            :code
        )
        ''', db_params)
    db.commit()

def serialize_cache(db):
    '''Serialize the cache db into a returned bytes object.
    '''

    # the cache format is a dict of tables, where the key is the
    # table name and the value is a list of rows.
    cachedict = {}
    stockham_v1_rows = list(db.execute('''
     SELECT
         kernel_name,
         arch,
         hip_version,
         generator_sum,
         code
     FROM stockham_v1
    '''))
    cachedict['stockham_v1'] = stockham_v1_rows
    return pickle.dumps(cachedict)

def deserialize_cache(db, serialized):
    '''Deserialize a bytes object from serialize_cache into the db.
    '''

    cachedict = pickle.loads(serialized)
    # load all the rows in a transaction for efficiency
    with db:
        for stockham_v1_row in cachedict.get('stockham_v1', []):
            db.execute('''
              INSERT OR REPLACE INTO stockham_v1 (
                  kernel_name,
                  arch,
                  hip_version,
                  generator_sum,
                  code
              )
              VALUES (
                  ?, ?, ?, ?, ?
              )''', stockham_v1_row)
