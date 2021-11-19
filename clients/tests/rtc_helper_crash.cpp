// just crash
int main()
{
    char* a = 0;
    // NOTE: this is supposed to crash, since it's used in a test
    // that checks crashing child processes.
    //
    // cppcheck-suppress nullPointer
    *a = 0;
    return 0;
}
