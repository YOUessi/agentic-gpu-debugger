#ifndef GPU_AGENT_VECTOR_API_H
#define GPU_AGENT_VECTOR_API_H

#include <cstddef>

// The trusted harness owns validation and JSON I/O. Candidates provide only
// this C++ symbol: write n float32 sums to out, returning zero on success.
// The caller guarantees non-null pointers and 1 <= n <= 65536.
int run_vector_add(const float* a, const float* b, float* out, std::size_t n);

#endif
