#include "main.h"

// ---------- 快读 ----------
struct FastInput {
    static const int BUFSIZE = 1 << 20;
    char buf[BUFSIZE];
    int idx, size;

    FastInput() : idx(0), size(0) {}
    inline char read() {
        if (idx >= size) {
            size = fread(buf, 1, BUFSIZE, stdin);
            idx = 0;
            if (size == 0) return EOF;
        }
        return buf[idx++];
    }

    template <typename T>
    bool readInt(T &out) {
        char c;
        T sign = 1;
        T num = 0;
        c = read();
        if (c == EOF) return false;
        while (c != '-' && (c < '0' || c > '9')) {
            c = read();
            if (c == EOF) return false;
        }
        if (c == '-') { sign = -1; c = read(); }
        for (; c >= '0' && c <= '9'; c = read())
            num = num * 10 + (c - '0');
        out = num * sign;
        return true;
    }
};

// ---------- 快写 ----------
struct FastOutput {
    static const int BUFSIZE = 1 << 20;
    char buf[BUFSIZE];
    int idx;

    FastOutput() : idx(0) {}
    ~FastOutput() { flush(); }

    inline void pushChar(char c) {
        if (idx == BUFSIZE) flush();
        buf[idx++] = c;
    }

    template <typename T>
    inline void writeInt(T x, char end = '\n') {
        if (x == 0) {
            pushChar('0');
        } else {
            if (x < 0) { pushChar('-'); x = -x; }
            char s[24];
            int n = 0;
            while (x > 0) { s[n++] = char('0' + x % 10); x /= 10; }
            while (n--) pushChar(s[n]);
        }
        if (end) pushChar(end);
    }

    inline void flush() {
        if (idx) {
            fwrite(buf, 1, idx, stdout);
            idx = 0;
        }
    }
};

int main(int argc, char* argv[]) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <input_file>\n", argv[0]);
        return 1;
    }

    // 重定向文件到 stdin
    if (freopen(argv[1], "r", stdin) == nullptr) {
        fprintf(stderr, "fileopen error %s\n", argv[1]);
        return 1;
    }

    FastInput in;
    FastOutput out;

    int N;
    in.readInt(N);

    // 使用 pinned host memory，减少 hipMemcpy 开销
    int *h_in = nullptr, *h_out = nullptr;
    hipHostMalloc(&h_in, sizeof(int) * N);
    hipHostMalloc(&h_out, sizeof(int) * N);

    for (int i = 0; i < N; ++i)
        in.readInt(h_in[i]);

    // 调用 GPU 核函数
    solve(h_in, h_out, N);

    // 输出
    for (int i = 0; i < N; ++i)
        out.writeInt(h_out[i], i + 1 == N ? '\n' : ' ');

    out.flush();

    hipHostFree(h_in);
    hipHostFree(h_out);

    return 0;
}

