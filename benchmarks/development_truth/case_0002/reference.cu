// Reviewer-readable host CPU reference. Never compiled or mounted with candidate code.
#include <cstddef>

void reference_add(const float* a, const float* b, float* out, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) out[i] = a[i] + b[i];
}
