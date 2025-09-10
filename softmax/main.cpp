#include "main.h"


static inline void skipSpaces(const char*& p) {
    while (*p && static_cast<unsigned char>(*p) <= ' ') ++p;
}

static inline int readInt(const char*& p) {
    skipSpaces(p);
    int sign = 1;
    if (*p == '-') { sign = -1; ++p; }
    else if (*p == '+') { ++p; }
    int value = 0;
    while (*p >= '0' && *p <= '9') {
        value = value * 10 + (*p - '0');
        ++p;
    }
    return sign * value;
}

static inline float readFloat(const char*& p) {
    skipSpaces(p);
    int sign = 1;
    if (*p == '-') { sign = -1; ++p; }
    else if (*p == '+') { ++p; }

    double value = 0.0;
    while (*p >= '0' && *p <= '9') {
        value = value * 10.0 + static_cast<double>(*p - '0');
        ++p;
    }
    if (*p == '.') {
        ++p;
        double frac = 1.0;
        while (*p >= '0' && *p <= '9') {
            frac *= 0.1;
            value += static_cast<double>(*p - '0') * frac;
            ++p;
        }
    }
    if (*p == 'e' || *p == 'E') {
        ++p;
        int expSign = 1;
        if (*p == '-') { expSign = -1; ++p; }
        else if (*p == '+') { ++p; }
        int expVal = 0;
        while (*p >= '0' && *p <= '9') { expVal = expVal * 10 + (*p - '0'); ++p; }
        if (expVal != 0) {
            value = value * pow(10.0, static_cast<double>(expSign * expVal));
        }
    }
    return static_cast<float>(sign < 0 ? -value : value);
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "usage: " << argv[0] << " <input_file>" << std::endl;
        return 1;
    }
    std::string filename = argv[1];

    int fd = open(filename.c_str(), O_RDONLY);
    if (fd < 0) {
        std::cerr << "fileopen error" << filename << std::endl;
        return 1;
    }
    struct stat st;
    if (fstat(fd, &st) != 0) {
        std::cerr << "fstat error" << std::endl;
        close(fd);
        return 1;
    }
    size_t fsize = static_cast<size_t>(st.st_size);
    std::vector<char> buf(fsize + 1);
    size_t off = 0;
    while (off < fsize) {
        ssize_t n = read(fd, buf.data() + off, fsize - off);
        if (n <= 0) break;
        off += static_cast<size_t>(n);
    }
    close(fd);
    buf[off] = '\0';

    const char* p = buf.data();
    int N = readInt(p);
    std::vector<float> input(N), output(N);
    for (int i = 0; i < N; ++i) input[i] = readFloat(p);

    solve(input.data(), output.data(), N);

    static char outbuf[1 << 20];
    setvbuf(stdout, outbuf, _IOFBF, sizeof(outbuf));
    for (int i = 0; i < N; ++i) {
        std::printf("%.9g%c", output[i], (i + 1 == N) ? '\n' : ' ');
    }
}
