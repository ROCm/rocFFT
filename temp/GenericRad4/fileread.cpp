#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>

#define LINE_MAX 19999

void CheckFile(bool open, const char *name)
{
    if(!open)
    {
        std::cout << "File: " << name << " could not be opened" << std::endl;
        exit(-1);
    }
}

extern "C" size_t file_lines_get(const char *file_name, char ***retp)
{
    std::ifstream pf(file_name);
    CheckFile(pf.is_open(), file_name);

    char **p;

    p = (char **) malloc(LINE_MAX * sizeof(char **));

    std::string line;
    size_t i = 0;
    while(std::getline(pf, line))
    {
        size_t l = line.length();
        char *t = (char *) malloc((l+2) * sizeof(char));
        line.copy(t, l);
        t[l] = '\n';
        t[l+1] = '\0';
        *(p + i) = t;
        i++;
    }

    *retp = p;
    return i;
}

extern "C" void file_lines_clear(size_t len, char **p)
{
    size_t i;
    for(i=0; i<len; i++)
    {
        free( *(p + i) );
    }

    free(p);
}
