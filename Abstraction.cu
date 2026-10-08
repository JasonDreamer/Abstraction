#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <queue>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

struct Pixel8 {
    unsigned char alpha;
    unsigned char red;
    unsigned char green;
    unsigned char blue;
};

struct Pixel16 {
    unsigned short alpha;
    unsigned short red;
    unsigned short green;
    unsigned short blue;
};

struct Pixel32 {
    float alpha;
    float red;
    float green;
    float blue;
};

struct PixelF {
    float r;
    float g;
    float b;
    float a;
};

struct LabColor {
    float l;
    float a;
    float b;
};

struct AnalysisSample {
    float l;
    float labA;
    float labB;
    float r;
    float g;
    float b;
    float alpha;
};

struct PaletteEntry {
    float l;
    float labA;
    float labB;
    float r;
    float g;
    float b;
};

struct PointF {
    float x;
    float y;
};

struct BoundaryEdge {
    int x0;
    int y0;
    int x1;
    int y1;
    int direction;
    bool used;
};

struct VectorTriangle {
    float x0;
    float y0;
    float x1;
    float y1;
    float x2;
    float y2;
    float r;
    float g;
    float b;
    float alpha;
    float priority;
    unsigned int hash;
};

struct RegionInfo {
    int cells = 0;
    double sumL = 0.0;
    double sumLabA = 0.0;
    double sumLabB = 0.0;
    double sumR = 0.0;
    double sumG = 0.0;
    double sumB = 0.0;
    double colorWeight = 0.0;
    double sumAlpha = 0.0;
    unsigned int hash = 0;
    double pixelWeight = 0.0;
    double sumX = 0.0;
    double sumY = 0.0;
    float minimumX = std::numeric_limits<float>::max();
    float minimumY = std::numeric_limits<float>::max();
    float maximumX = -std::numeric_limits<float>::max();
    float maximumY = -std::numeric_limits<float>::max();
    bool transparent = false;
};

struct AlphaIslandInfo {
    double pixelWeight = 0.0;
    double sumX = 0.0;
    double sumY = 0.0;
    float minimumX = std::numeric_limits<float>::max();
    float minimumY = std::numeric_limits<float>::max();
    float maximumX = -std::numeric_limits<float>::max();
    float maximumY = -std::numeric_limits<float>::max();
};

__host__ __device__ float Clamp01(float value)
{
    return fminf(1.0f, fmaxf(0.0f, value));
}

__host__ __device__ unsigned int Hash32(unsigned int value)
{
    value ^= value >> 16;
    value *= 0x7feb352dU;
    value ^= value >> 15;
    value *= 0x846ca68bU;
    value ^= value >> 16;
    return value;
}

__host__ __device__ float Hash01(unsigned int value)
{
    return static_cast<float>(Hash32(value) & 0x00FFFFFFU) / 16777215.0f;
}

template <typename PixelType>
__device__ PixelF ReadPixel(const char* input, int rowBytes, int x, int y);

template <>
__device__ PixelF ReadPixel<Pixel8>(const char* input, int rowBytes, int x, int y)
{
    const Pixel8* row = reinterpret_cast<const Pixel8*>(input + static_cast<size_t>(y) * rowBytes);
    const Pixel8 p = row[x];
    return PixelF{p.red / 255.0f, p.green / 255.0f, p.blue / 255.0f, p.alpha / 255.0f};
}

template <>
__device__ PixelF ReadPixel<Pixel16>(const char* input, int rowBytes, int x, int y)
{
    const Pixel16* row = reinterpret_cast<const Pixel16*>(input + static_cast<size_t>(y) * rowBytes);
    const Pixel16 p = row[x];
    return PixelF{p.red / 32768.0f, p.green / 32768.0f, p.blue / 32768.0f, p.alpha / 32768.0f};
}

template <>
__device__ PixelF ReadPixel<Pixel32>(const char* input, int rowBytes, int x, int y)
{
    const Pixel32* row = reinterpret_cast<const Pixel32*>(input + static_cast<size_t>(y) * rowBytes);
    const Pixel32 p = row[x];
    return PixelF{Clamp01(p.red), Clamp01(p.green), Clamp01(p.blue), Clamp01(p.alpha)};
}

template <typename PixelType>
__device__ void WritePixel(char* output, int rowBytes, int x, int y, PixelF pixel);

template <>
__device__ void WritePixel<Pixel8>(char* output, int rowBytes, int x, int y, PixelF p)
{
    Pixel8* row = reinterpret_cast<Pixel8*>(output + static_cast<size_t>(y) * rowBytes);
    row[x].red = static_cast<unsigned char>(roundf(Clamp01(p.r) * 255.0f));
    row[x].green = static_cast<unsigned char>(roundf(Clamp01(p.g) * 255.0f));
    row[x].blue = static_cast<unsigned char>(roundf(Clamp01(p.b) * 255.0f));
    row[x].alpha = static_cast<unsigned char>(roundf(Clamp01(p.a) * 255.0f));
}

template <>
__device__ void WritePixel<Pixel16>(char* output, int rowBytes, int x, int y, PixelF p)
{
    Pixel16* row = reinterpret_cast<Pixel16*>(output + static_cast<size_t>(y) * rowBytes);
    row[x].red = static_cast<unsigned short>(roundf(Clamp01(p.r) * 32768.0f));
    row[x].green = static_cast<unsigned short>(roundf(Clamp01(p.g) * 32768.0f));
    row[x].blue = static_cast<unsigned short>(roundf(Clamp01(p.b) * 32768.0f));
    row[x].alpha = static_cast<unsigned short>(roundf(Clamp01(p.a) * 32768.0f));
}

template <>
__device__ void WritePixel<Pixel32>(char* output, int rowBytes, int x, int y, PixelF p)
{
    Pixel32* row = reinterpret_cast<Pixel32*>(output + static_cast<size_t>(y) * rowBytes);
    row[x].red = Clamp01(p.r);
    row[x].green = Clamp01(p.g);
    row[x].blue = Clamp01(p.b);
    row[x].alpha = Clamp01(p.a);
}

__host__ __device__ PixelF StraightFromPremultiplied(PixelF p)
{
    if (p.a > 1.0e-5f) {
        const float inverseAlpha = 1.0f / p.a;
        p.r = Clamp01(p.r * inverseAlpha);
        p.g = Clamp01(p.g * inverseAlpha);
        p.b = Clamp01(p.b * inverseAlpha);
    } else {
        p.r = Clamp01(p.r);
        p.g = Clamp01(p.g);
        p.b = Clamp01(p.b);
    }
    return p;
}

__host__ __device__ PixelF Premultiply(PixelF p)
{
    p.r *= p.a;
    p.g *= p.a;
    p.b *= p.a;
    return p;
}

__host__ __device__ LabColor RgbToLab(PixelF straight)
{
    const float r = straight.r > 0.04045f ? powf((straight.r + 0.055f) / 1.055f, 2.4f) : straight.r / 12.92f;
    const float g = straight.g > 0.04045f ? powf((straight.g + 0.055f) / 1.055f, 2.4f) : straight.g / 12.92f;
    const float b = straight.b > 0.04045f ? powf((straight.b + 0.055f) / 1.055f, 2.4f) : straight.b / 12.92f;
    const float x = (r * 0.4124f + g * 0.3576f + b * 0.1805f) / 0.95047f;
    const float y = r * 0.2126f + g * 0.7152f + b * 0.0722f;
    const float z = (r * 0.0193f + g * 0.1192f + b * 0.9505f) / 1.08883f;
    const float fx = x > 0.008856f ? powf(x, 1.0f / 3.0f) : 7.787f * x + 16.0f / 116.0f;
    const float fy = y > 0.008856f ? powf(y, 1.0f / 3.0f) : 7.787f * y + 16.0f / 116.0f;
    const float fz = z > 0.008856f ? powf(z, 1.0f / 3.0f) : 7.787f * z + 16.0f / 116.0f;
    return LabColor{116.0f * fy - 16.0f, 500.0f * (fx - fy), 200.0f * (fy - fz)};
}

template <typename PixelType>
__global__ void AnalyzeKernel(
    const char* input,
    int rowBytes,
    int width,
    int height,
    int analysisStep,
    int analysisCols,
    int analysisRows,
    AnalysisSample* samples)
{
    const int cellX = blockIdx.x * blockDim.x + threadIdx.x;
    const int cellY = blockIdx.y * blockDim.y + threadIdx.y;
    if (cellX >= analysisCols || cellY >= analysisRows) return;

    const int left = cellX * analysisStep;
    const int top = cellY * analysisStep;
    const int cellWidth = min(analysisStep, width - left);
    const int cellHeight = min(analysisStep, height - top);
    float sumL = 0.0f;
    float sumLabA = 0.0f;
    float sumLabB = 0.0f;
    float sumR = 0.0f;
    float sumG = 0.0f;
    float sumB = 0.0f;
    float sumWeight = 0.0f;
    float sumAlpha = 0.0f;

    for (int sampleY = 0; sampleY < 3; ++sampleY) {
        const int y = min(height - 1, top + ((sampleY * 2 + 1) * cellHeight) / 6);
        for (int sampleX = 0; sampleX < 3; ++sampleX) {
            const int x = min(width - 1, left + ((sampleX * 2 + 1) * cellWidth) / 6);
            PixelF straight = StraightFromPremultiplied(ReadPixel<PixelType>(input, rowBytes, x, y));
            const LabColor lab = RgbToLab(straight);
            const float weight = fmaxf(straight.a, 0.01f);
            sumL += lab.l * weight;
            sumLabA += lab.a * weight;
            sumLabB += lab.b * weight;
            sumR += straight.r * weight;
            sumG += straight.g * weight;
            sumB += straight.b * weight;
            sumWeight += weight;
            sumAlpha += straight.a;
        }
    }

    const float inverseWeight = 1.0f / fmaxf(sumWeight, 1.0e-6f);
    AnalysisSample result;
    result.l = sumL * inverseWeight;
    result.labA = sumLabA * inverseWeight;
    result.labB = sumLabB * inverseWeight;
    result.r = sumR * inverseWeight;
    result.g = sumG * inverseWeight;
    result.b = sumB * inverseWeight;
    result.alpha = sumAlpha / 9.0f;
    samples[static_cast<size_t>(cellY) * analysisCols + cellX] = result;
}

__host__ __device__ float LabDistanceSquared(
    float l0,
    float a0,
    float b0,
    float l1,
    float a1,
    float b1)
{
    const float dl = l0 - l1;
    const float da = a0 - a1;
    const float db = b0 - b1;
    return dl * dl + da * da + db * db;
}

__device__ float Cross2D(float ax, float ay, float bx, float by, float px, float py)
{
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax);
}

__device__ bool PointInTriangle(float px, float py, const VectorTriangle& triangle)
{
    const float c0 = Cross2D(triangle.x0, triangle.y0, triangle.x1, triangle.y1, px, py);
    const float c1 = Cross2D(triangle.x1, triangle.y1, triangle.x2, triangle.y2, px, py);
    const float c2 = Cross2D(triangle.x2, triangle.y2, triangle.x0, triangle.y0, px, py);
    const bool hasNegative = c0 < -1.0e-4f || c1 < -1.0e-4f || c2 < -1.0e-4f;
    const bool hasPositive = c0 > 1.0e-4f || c1 > 1.0e-4f || c2 > 1.0e-4f;
    return !(hasNegative && hasPositive);
}

__device__ float SegmentDistanceSquared(float px, float py, float ax, float ay, float bx, float by)
{
    const float dx = bx-ax, dy = by-ay;
    const float length = dx*dx+dy*dy;
    const float t = length > 0 ? Clamp01(((px-ax)*dx+(py-ay)*dy)/length) : 0;
    const float x = px-ax-t*dx, y = py-ay-t*dy;
    return x*x+y*y;
}

__device__ float TriangleDistanceSquared(float px, float py, const VectorTriangle& t)
{
    if (PointInTriangle(px,py,t)) return 0;
    return fminf(SegmentDistanceSquared(px,py,t.x0,t.y0,t.x1,t.y1),
        fminf(SegmentDistanceSquared(px,py,t.x1,t.y1,t.x2,t.y2),
            SegmentDistanceSquared(px,py,t.x2,t.y2,t.x0,t.y0)));
}

template <typename PixelType>
__global__ void RenderKernel(
    const char* input,
    char* output,
    int inputWidth,
    int inputHeight,
    int outputWidth,
    int outputHeight,
    int inputRowBytes,
    int outputRowBytes,
    int inputOriginX,
    int inputOriginY,
    int outputOriginX,
    int outputOriginY,
    const VectorTriangle* triangles,
    int triangleCount,
    const int* binOffsets,
    const int* binTriangleIndices,
    int binSize,
    int binCols,
    float mix,
    bool sealGaps,
    PixelF fallbackColor)
{
    const int outputX = blockIdx.x * blockDim.x + threadIdx.x;
    const int outputY = blockIdx.y * blockDim.y + threadIdx.y;
    if (outputX >= outputWidth || outputY >= outputHeight) return;

    const int globalX = outputOriginX + outputX;
    const int globalY = outputOriginY + outputY;
    const int inputX = globalX - inputOriginX;
    const int inputY = globalY - inputOriginY;
    PixelF original = PixelF{0.0f, 0.0f, 0.0f, 0.0f};
    if (inputX >= 0 && inputX < inputWidth && inputY >= 0 && inputY < inputHeight) {
        original = ReadPixel<PixelType>(input, inputRowBytes, inputX, inputY);
    }

    PixelF abstractPixel = PixelF{0.0f, 0.0f, 0.0f, 0.0f};
    // In attached mode the source provides only the alpha silhouette; RGB always
    // comes from flat vector regions. Detached mode retains intentional gaps.

    const float pixelX = static_cast<float>(globalX) + 0.5f;
    const float pixelY = static_cast<float>(globalY) + 0.5f;
    const int binX = min(binCols - 1, max(0, outputX / binSize));
    const int binY = max(0, outputY / binSize);
    const int binIndex = binY * binCols + binX;
    int selectedTriangle = -1;
    float selectedPriority = -1.0e30f;
    const int begin = binOffsets[binIndex];
    const int end = binOffsets[binIndex + 1];
    for (int item = begin; item < end; ++item) {
        const int triangleIndex = binTriangleIndices[item];
        if (triangleIndex < 0 || triangleIndex >= triangleCount) continue;
        const VectorTriangle triangle = triangles[triangleIndex];
        if (sealGaps && triangle.alpha <= 0.005f) continue;
        if (triangle.priority >= selectedPriority && PointInTriangle(pixelX, pixelY, triangle)) {
            selectedPriority = triangle.priority;
            selectedTriangle = triangleIndex;
        }
    }

    if (sealGaps && original.a > 0 && selectedTriangle < 0) {
        // Extend the nearest polygon into the seam. Search spatial bins outward,
        // stopping once unvisited bins cannot contain a closer vector boundary.
        float bestDistance = 1.0e30f;
        const int binRows = (outputHeight + binSize - 1) / binSize;
        const float localX = outputX + 0.5f, localY = outputY + 0.5f;
        for (int radius = 0; radius < max(binCols,binRows); ++radius) {
            const int left = max(0,binX-radius), right = min(binCols-1,binX+radius);
            const int top = max(0,binY-radius), bottom = min(binRows-1,binY+radius);
            for (int y = top; y <= bottom; ++y) for (int x = left; x <= right; ++x) {
                if (max(abs(x-binX),abs(y-binY)) != radius) continue;
                const int bin = y*binCols+x;
                for (int item = binOffsets[bin]; item < binOffsets[bin+1]; ++item) {
                    const int candidate = binTriangleIndices[item];
                    const VectorTriangle& triangle = triangles[candidate];
                    if (triangle.alpha <= 0.005f) continue;
                    const float distance = TriangleDistanceSquared(pixelX,pixelY,triangle);
                    if (distance < bestDistance || (distance == bestDistance && candidate < selectedTriangle)) {
                        bestDistance = distance;
                        selectedTriangle = candidate;
                    }
                }
            }
            float unsearchedDistance = 1.0e15f;
            if (left > 0) unsearchedDistance = fminf(unsearchedDistance,localX-left*binSize);
            if (right+1 < binCols) unsearchedDistance = fminf(unsearchedDistance,(right+1)*binSize-localX);
            if (top > 0) unsearchedDistance = fminf(unsearchedDistance,localY-top*binSize);
            if (bottom+1 < binRows) unsearchedDistance = fminf(unsearchedDistance,(bottom+1)*binSize-localY);
            if (selectedTriangle >= 0 && bestDistance <= unsearchedDistance*unsearchedDistance) break;
        }
    }

    if (selectedTriangle >= 0) {
        const VectorTriangle triangle = triangles[selectedTriangle];
        PixelF region = PixelF{triangle.r, triangle.g, triangle.b, triangle.alpha};
        abstractPixel = Premultiply(region);
    }

    if (sealGaps) {
        const PixelF color = selectedTriangle >= 0
            ? PixelF{triangles[selectedTriangle].r,triangles[selectedTriangle].g,triangles[selectedTriangle].b,1}
            : fallbackColor;
        abstractPixel = Premultiply(PixelF{color.r,color.g,color.b,original.a});
    }

    PixelF result;
    result.r = original.r + (abstractPixel.r - original.r) * mix;
    result.g = original.g + (abstractPixel.g - original.g) * mix;
    result.b = original.b + (abstractPixel.b - original.b) * mix;
    result.a = original.a + (abstractPixel.a - original.a) * mix;
    WritePixel<PixelType>(output, outputRowBytes, outputX, outputY, result);
}

class DisjointSet {
public:
    explicit DisjointSet(int count) : parent_(count), size_(count, 1)
    {
        for (int i = 0; i < count; ++i) parent_[i] = i;
    }

    int Find(int value)
    {
        int root = value;
        while (parent_[root] != root) root = parent_[root];
        while (parent_[value] != value) {
            const int next = parent_[value];
            parent_[value] = root;
            value = next;
        }
        return root;
    }

    void Unite(int first, int second)
    {
        first = Find(first);
        second = Find(second);
        if (first == second) return;
        if (size_[first] < size_[second]) std::swap(first, second);
        parent_[second] = first;
        size_[first] += size_[second];
    }

private:
    std::vector<int> parent_;
    std::vector<int> size_;
};

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() : pointer_(nullptr) {}
    ~DeviceBuffer()
    {
        if (pointer_) cudaFree(pointer_);
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    bool Allocate(size_t count)
    {
        if (count == 0) return true;
        return cudaMalloc(reinterpret_cast<void**>(&pointer_), count * sizeof(T)) == cudaSuccess;
    }

    T* Get() { return pointer_; }
    const T* Get() const { return pointer_; }

private:
    T* pointer_;
};

static int NearestPaletteHost(const AnalysisSample& sample, const std::vector<PaletteEntry>& palette)
{
    float bestDistance = std::numeric_limits<float>::max();
    int bestIndex = 0;
    for (int i = 0; i < static_cast<int>(palette.size()); ++i) {
        const float distance = LabDistanceSquared(
            sample.l,
            sample.labA,
            sample.labB,
            palette[i].l,
            palette[i].labA,
            palette[i].labB);
        if (distance < bestDistance) {
            bestDistance = distance;
            bestIndex = i;
        }
    }
    return bestIndex;
}

static std::vector<PaletteEntry> BuildPalette(
    const std::vector<AnalysisSample>& samples,
    int requestedColors,
    int seed)
{
    std::vector<int> candidates;
    constexpr size_t kMaximumPaletteSamples = 32768U;
    const size_t sampleStride = std::max<size_t>(1U, (samples.size() + kMaximumPaletteSamples - 1U) / kMaximumPaletteSamples);
    const size_t sampleOffset = sampleStride > 1U
        ? Hash32(static_cast<unsigned int>(seed) ^ 0x3c6ef372U) % sampleStride
        : 0U;
    candidates.reserve(std::min(samples.size(), kMaximumPaletteSamples));
    for (size_t i = sampleOffset; i < samples.size(); i += sampleStride) {
        if (samples[i].alpha > 0.005f) candidates.push_back(static_cast<int>(i));
    }
    if (candidates.empty() && sampleStride > 1U) {
        for (size_t i = 0; i < samples.size() && candidates.size() < kMaximumPaletteSamples; ++i) {
            if (samples[i].alpha > 0.005f) candidates.push_back(static_cast<int>(i));
        }
    }
    if (candidates.empty()) {
        return std::vector<PaletteEntry>{PaletteEntry{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}};
    }

    const int colorCount = std::max(1, std::min(requestedColors, static_cast<int>(candidates.size())));
    std::vector<PaletteEntry> palette;
    palette.reserve(colorCount);
    const unsigned int firstHash = Hash32(static_cast<unsigned int>(seed) ^ 0x51ed270bU);
    const AnalysisSample& first = samples[candidates[firstHash % candidates.size()]];
    palette.push_back(PaletteEntry{first.l, first.labA, first.labB, first.r, first.g, first.b});

    std::vector<float> nearestDistances(candidates.size(), std::numeric_limits<float>::max());
    while (static_cast<int>(palette.size()) < colorCount) {
        float greatestDistance = -1.0f;
        int selected = candidates.front();
        const PaletteEntry& entry = palette.back();
        for (size_t candidateIndex = 0; candidateIndex < candidates.size(); ++candidateIndex) {
            const int candidate = candidates[candidateIndex];
            const AnalysisSample& sample = samples[candidate];
            float& nearestDistance = nearestDistances[candidateIndex];
            nearestDistance = std::min(nearestDistance, LabDistanceSquared(
                    sample.l,
                    sample.labA,
                    sample.labB,
                    entry.l,
                    entry.labA,
                    entry.labB));
            const float weightedDistance = nearestDistance * std::max(sample.alpha, 0.05f);
            if (weightedDistance > greatestDistance) {
                greatestDistance = weightedDistance;
                selected = candidate;
            }
        }
        const AnalysisSample& sample = samples[selected];
        palette.push_back(PaletteEntry{sample.l, sample.labA, sample.labB, sample.r, sample.g, sample.b});
    }

    for (int iteration = 0; iteration < 10; ++iteration) {
        struct Accumulator {
            double l = 0.0;
            double labA = 0.0;
            double labB = 0.0;
            double r = 0.0;
            double g = 0.0;
            double b = 0.0;
            double weight = 0.0;
        };
        std::vector<Accumulator> accumulators(palette.size());
        for (const int candidate : candidates) {
            const AnalysisSample& sample = samples[candidate];
            const int nearest = NearestPaletteHost(sample, palette);
            const double weight = std::max(sample.alpha, 0.05f);
            Accumulator& accumulator = accumulators[nearest];
            accumulator.l += sample.l * weight;
            accumulator.labA += sample.labA * weight;
            accumulator.labB += sample.labB * weight;
            accumulator.r += sample.r * weight;
            accumulator.g += sample.g * weight;
            accumulator.b += sample.b * weight;
            accumulator.weight += weight;
        }
        for (int i = 0; i < static_cast<int>(palette.size()); ++i) {
            if (accumulators[i].weight <= 0.0) continue;
            const double inverseWeight = 1.0 / accumulators[i].weight;
            palette[i].l = static_cast<float>(accumulators[i].l * inverseWeight);
            palette[i].labA = static_cast<float>(accumulators[i].labA * inverseWeight);
            palette[i].labB = static_cast<float>(accumulators[i].labB * inverseWeight);
            palette[i].r = static_cast<float>(accumulators[i].r * inverseWeight);
            palette[i].g = static_cast<float>(accumulators[i].g * inverseWeight);
            palette[i].b = static_cast<float>(accumulators[i].b * inverseWeight);
        }
    }
    return palette;
}

static float SignedArea(const std::vector<PointF>& polygon)
{
    double area = 0.0;
    for (size_t i = 0; i < polygon.size(); ++i) {
        const PointF& current = polygon[i];
        const PointF& next = polygon[(i + 1) % polygon.size()];
        area += static_cast<double>(current.x) * next.y - static_cast<double>(next.x) * current.y;
    }
    return static_cast<float>(area * 0.5);
}

static float Cross(const PointF& first, const PointF& second, const PointF& third)
{
    return (second.x - first.x) * (third.y - first.y) -
        (second.y - first.y) * (third.x - first.x);
}

static float DistanceSquaredToSegment(const PointF& point, const PointF& first, const PointF& second)
{
    const float dx = second.x - first.x;
    const float dy = second.y - first.y;
    const float denominator = dx * dx + dy * dy;
    float t = denominator > 1.0e-8f
        ? ((point.x - first.x) * dx + (point.y - first.y) * dy) / denominator
        : 0.0f;
    t = std::max(0.0f, std::min(1.0f, t));
    const float nearestX = first.x + dx * t;
    const float nearestY = first.y + dy * t;
    const float differenceX = point.x - nearestX;
    const float differenceY = point.y - nearestY;
    return differenceX * differenceX + differenceY * differenceY;
}

struct SimplifyCandidate {
    float distanceSquared;
    int index;
    int version;
};

struct SimplifyCandidateGreater {
    bool operator()(const SimplifyCandidate& first, const SimplifyCandidate& second) const
    {
        return first.distanceSquared > second.distanceSquared;
    }
};

static std::vector<PointF> SimplifyPolygon(
    const std::vector<PointF>& source,
    float tolerance,
    int maximumVertices)
{
    if (source.size() <= 3) return source;
    const int count = static_cast<int>(source.size());
    std::vector<int> previous(count);
    std::vector<int> next(count);
    std::vector<int> version(count, 0);
    std::vector<unsigned char> alive(count, 1);
    for (int i = 0; i < count; ++i) {
        previous[i] = (i + count - 1) % count;
        next[i] = (i + 1) % count;
    }

    std::priority_queue<
        SimplifyCandidate,
        std::vector<SimplifyCandidate>,
        SimplifyCandidateGreater> candidates;
    const auto addCandidate = [&](int index) {
        candidates.push(SimplifyCandidate{
            DistanceSquaredToSegment(source[index], source[previous[index]], source[next[index]]),
            index,
            version[index]});
    };
    for (int i = 0; i < count; ++i) addCandidate(i);

    int aliveCount = count;
    const float toleranceSquared = tolerance * tolerance;
    while (aliveCount > 3 && !candidates.empty()) {
        const SimplifyCandidate candidate = candidates.top();
        candidates.pop();
        if (!alive[candidate.index] || version[candidate.index] != candidate.version) continue;
        if (aliveCount <= maximumVertices && candidate.distanceSquared > toleranceSquared) break;

        const int before = previous[candidate.index];
        const int after = next[candidate.index];
        alive[candidate.index] = 0;
        next[before] = after;
        previous[after] = before;
        --aliveCount;
        ++version[before];
        ++version[after];
        addCandidate(before);
        addCandidate(after);
    }

    std::vector<PointF> result;
    result.reserve(aliveCount);
    int start = 0;
    while (start < count && !alive[start]) ++start;
    if (start >= count) return result;
    int current = start;
    do {
        result.push_back(source[current]);
        current = next[current];
    } while (current != start && static_cast<int>(result.size()) <= aliveCount);
    return result;
}

static unsigned long long PointKey(int x, int y)
{
    return (static_cast<unsigned long long>(static_cast<unsigned int>(y)) << 32) |
        static_cast<unsigned int>(x);
}

static std::vector<std::vector<std::pair<int, int>>> TraceLoops(std::vector<BoundaryEdge>& edges)
{
    std::unordered_map<unsigned long long, std::vector<int>> outgoing;
    outgoing.reserve(edges.size() * 2);
    for (int i = 0; i < static_cast<int>(edges.size()); ++i) {
        outgoing[PointKey(edges[i].x0, edges[i].y0)].push_back(i);
    }

    std::vector<std::vector<std::pair<int, int>>> loops;
    for (int firstEdge = 0; firstEdge < static_cast<int>(edges.size()); ++firstEdge) {
        if (edges[firstEdge].used) continue;
        std::vector<std::pair<int, int>> loop;
        const int startX = edges[firstEdge].x0;
        const int startY = edges[firstEdge].y0;
        int currentEdge = firstEdge;
        bool closed = false;

        for (size_t iteration = 0; iteration <= edges.size(); ++iteration) {
            BoundaryEdge& edge = edges[currentEdge];
            if (edge.used) break;
            edge.used = true;
            loop.push_back(std::make_pair(edge.x0, edge.y0));
            if (edge.x1 == startX && edge.y1 == startY) {
                closed = true;
                break;
            }

            const auto found = outgoing.find(PointKey(edge.x1, edge.y1));
            if (found == outgoing.end()) break;
            int selected = -1;
            int selectedTurnScore = 5;
            for (const int candidateIndex : found->second) {
                const BoundaryEdge& candidate = edges[candidateIndex];
                if (candidate.used) continue;
                const int turn = (candidate.direction - edge.direction + 4) % 4;
                int turnScore = 3;
                if (turn == 1) turnScore = 0;
                else if (turn == 0) turnScore = 1;
                else if (turn == 3) turnScore = 2;
                if (turnScore < selectedTurnScore) {
                    selectedTurnScore = turnScore;
                    selected = candidateIndex;
                }
            }
            if (selected < 0) break;
            currentEdge = selected;
        }
        if (closed && loop.size() >= 3) loops.push_back(std::move(loop));
    }
    return loops;
}

static bool PointInsideHostTriangle(
    const PointF& point,
    const PointF& first,
    const PointF& second,
    const PointF& third)
{
    const float firstCross = Cross(first, second, point);
    const float secondCross = Cross(second, third, point);
    const float thirdCross = Cross(third, first, point);
    return firstCross >= -1.0e-4f && secondCross >= -1.0e-4f && thirdCross >= -1.0e-4f;
}

static void AddTriangle(
    const std::vector<PointF>& polygon,
    int first,
    int second,
    int third,
    const PaletteEntry& color,
    float alpha,
    float priority,
    unsigned int hash,
    std::vector<VectorTriangle>& triangles)
{
    // Degenerate fill triangles otherwise pass every half-plane test on their line.
    if (std::fabs(Cross(polygon[first], polygon[second], polygon[third])) <= 1.0e-8f) return;
    triangles.push_back(VectorTriangle{
        polygon[first].x,
        polygon[first].y,
        polygon[second].x,
        polygon[second].y,
        polygon[third].x,
        polygon[third].y,
        color.r,
        color.g,
        color.b,
        alpha,
        priority,
        hash});
}

static bool EarClipPolygon(
    std::vector<PointF> polygon,
    const PaletteEntry& color,
    float alpha,
    float priority,
    unsigned int hash,
    std::vector<VectorTriangle>& triangles)
{
    if (polygon.size() < 3) return false;
    if (SignedArea(polygon) < 0.0f) std::reverse(polygon.begin(), polygon.end());
    std::vector<int> indices(polygon.size());
    for (int i = 0; i < static_cast<int>(indices.size()); ++i) indices[i] = i;

    while (indices.size() > 3) {
        bool clipped = false;
        for (int cursor = 0; cursor < static_cast<int>(indices.size()); ++cursor) {
            const int previous = indices[(cursor + indices.size() - 1) % indices.size()];
            const int current = indices[cursor];
            const int next = indices[(cursor + 1) % indices.size()];
            if (Cross(polygon[previous], polygon[current], polygon[next]) <= 1.0e-5f) continue;

            bool containsVertex = false;
            for (const int candidate : indices) {
                if (candidate == previous || candidate == current || candidate == next) continue;
                if (PointInsideHostTriangle(
                    polygon[candidate],
                    polygon[previous],
                    polygon[current],
                    polygon[next])) {
                    containsVertex = true;
                    break;
                }
            }
            if (containsVertex) continue;
            AddTriangle(polygon, previous, current, next, color, alpha, priority, hash, triangles);
            indices.erase(indices.begin() + cursor);
            clipped = true;
            break;
        }
        // Retrying identical geometry cannot find a new ear.
        if (!clipped) return false;
    }

    if (indices.size() == 3) {
        AddTriangle(polygon, indices[0], indices[1], indices[2], color, alpha, priority, hash, triangles);
        return true;
    }
    return false;
}

// Tessellate the same vector contour with even-odd fill when ear clipping cannot
// handle a repeated vertex, a touching edge, or an intersection introduced by
// simplification/jitter. No original-image pixels are used here.
static bool TessellateContourBands(
    const std::vector<PointF>& polygon,
    const PaletteEntry& color, float alpha, float priority, unsigned int hash,
    std::vector<VectorTriangle>& triangles)
{
    struct Edge {
        double x0, y0, x1, y1;
        double X(double y) const { return x0 + (x1 - x0) * ((y - y0) / (y1 - y0)); }
    };
    std::vector<Edge> edges;
    std::vector<double> levels;
    for (size_t i = 0; i < polygon.size(); ++i) {
        const PointF a = polygon[i], b = polygon[(i + 1) % polygon.size()];
        levels.push_back(a.y);
        if (a.y == b.y) continue;
        edges.push_back(a.y < b.y ? Edge{a.x,a.y,b.x,b.y} : Edge{b.x,b.y,a.x,a.y});
    }
    // Split at crossings as well as vertices so edge order is constant in each band.
    std::sort(edges.begin(), edges.end(), [](const Edge& a, const Edge& b) { return a.y0 < b.y0; });
    for (size_t i = 0; i < edges.size(); ++i) {
        const Edge& a = edges[i];
        for (size_t j = i + 1; j < edges.size() && edges[j].y0 < a.y1; ++j) {
            const Edge& b = edges[j];
            if (std::max(a.x0,a.x1) < std::min(b.x0,b.x1) ||
                std::max(b.x0,b.x1) < std::min(a.x0,a.x1)) continue;
            const double low = std::max(a.y0,b.y0), high = std::min(a.y1,b.y1);
            const double d0 = a.X(low) - b.X(low), d1 = a.X(high) - b.X(high);
            if ((d0 < 0 && d1 > 0) || (d0 > 0 && d1 < 0)) {
                levels.push_back(low + (high - low) * d0 / (d0 - d1));
            }
        }
    }
    std::sort(levels.begin(), levels.end());
    levels.erase(std::unique(levels.begin(), levels.end()), levels.end());
    std::vector<std::pair<double, size_t>> crossings;
    std::vector<PointF> quad(4);
    for (size_t band = 1; band < levels.size(); ++band) {
        const double top = levels[band - 1], bottom = levels[band];
        if (bottom - top <= 1.0e-9) continue;
        const double middle = (top + bottom) * 0.5;
        crossings.clear();
        for (size_t i = 0; i < edges.size(); ++i) {
            if (edges[i].y0 <= middle && middle < edges[i].y1)
                crossings.emplace_back(edges[i].X(middle), i);
        }
        std::sort(crossings.begin(), crossings.end());
        if (crossings.size() % 2 != 0) return false;
        for (size_t i = 0; i + 1 < crossings.size(); i += 2) {
            if (crossings[i + 1].first - crossings[i].first <= 1.0e-9) continue;
            const Edge& left = edges[crossings[i].second];
            const Edge& right = edges[crossings[i + 1].second];
            quad[0] = PointF{static_cast<float>(left.X(top)), static_cast<float>(top)};
            quad[1] = PointF{static_cast<float>(right.X(top)), static_cast<float>(top)};
            quad[2] = PointF{static_cast<float>(right.X(bottom)), static_cast<float>(bottom)};
            quad[3] = PointF{static_cast<float>(left.X(bottom)), static_cast<float>(bottom)};
            AddTriangle(quad, 0, 1, 2, color, alpha, priority, hash, triangles);
            AddTriangle(quad, 0, 2, 3, color, alpha, priority, hash, triangles);
        }
    }
    return true;
}

static bool TriangulatePolygon(
    const std::vector<PointF>& polygon,
    const PaletteEntry& color, float alpha, float priority, unsigned int hash,
    std::vector<VectorTriangle>& triangles)
{
    const size_t start = triangles.size();
    if (EarClipPolygon(polygon, color, alpha, priority, hash, triangles)) {
        double filledArea = 0;
        for (size_t i = start; i < triangles.size(); ++i) {
            const auto& t = triangles[i];
            filledArea += std::fabs((static_cast<double>(t.x1)-t.x0)*(t.y2-t.y0) -
                (static_cast<double>(t.y1)-t.y0)*(t.x2-t.x0)) * 0.5;
        }
        const double contourArea = std::fabs(static_cast<double>(SignedArea(polygon)));
        // An ear clip can finish on a crossing contour yet produce overlapping triangles.
        if (contourArea > 1.0e-8 && std::fabs(filledArea-contourArea) <= std::max(1.0e-5,contourArea*1.0e-5)) return true;
    }
    triangles.resize(start);
    if (TessellateContourBands(polygon, color, alpha, priority, hash, triangles)) return true;
    triangles.resize(start);
    return false;
}

static bool ComponentsCanConnect(
    const AnalysisSample& first,
    int firstLabel,
    const AnalysisSample& second,
    int secondLabel,
    int transparentLabel,
    float toleranceSquared)
{
    const bool firstTransparent = firstLabel == transparentLabel;
    const bool secondTransparent = secondLabel == transparentLabel;
    if (firstTransparent || secondTransparent) return firstTransparent && secondTransparent;
    if (firstLabel == secondLabel) return true;
    if (toleranceSquared <= 0.0f) return false;
    return LabDistanceSquared(
        first.l,
        first.labA,
        first.labB,
        second.l,
        second.labA,
        second.labB) <= toleranceSquared;
}

static std::vector<PointF> BuildRelaxOffsets(
    const std::vector<AlphaIslandInfo>& islands,
    const std::vector<PointF>& initialOffsets,
    float frameCenterX,
    float frameCenterY,
    float requestedGap,
    float applicationStrength)
{
    std::vector<PointF> offsets(islands.size(), PointF{0.0f, 0.0f});
    if (initialOffsets.size() != islands.size()) return offsets;
    offsets = initialOffsets;
    if (islands.size() <= 1 || requestedGap <= 0.0f) return offsets;

    std::vector<PointF> originalCenters(islands.size());
    std::vector<PointF> relaxedCenters(islands.size());
    std::vector<PointF> halfExtents(islands.size());
    for (int i = 0; i < static_cast<int>(islands.size()); ++i) {
        const AlphaIslandInfo& island = islands[i];
        const double inverseWeight = 1.0 / std::max(1.0, island.pixelWeight);
        const PointF center = PointF{
            static_cast<float>(island.sumX * inverseWeight),
            static_cast<float>(island.sumY * inverseWeight)};
        originalCenters[i] = center;
        halfExtents[i] = PointF{
            std::max(0.5f, (island.maximumX - island.minimumX) * 0.5f),
            std::max(0.5f, (island.maximumY - island.minimumY) * 0.5f)};

        float directionX = center.x - frameCenterX;
        float directionY = center.y - frameCenterY;
        float directionLength = std::sqrt(directionX * directionX + directionY * directionY);
        if (directionLength <= 1.0e-4f) {
            const float angle = 6.28318530718f * static_cast<float>(i) / static_cast<float>(islands.size());
            directionX = std::cos(angle);
            directionY = std::sin(angle);
        } else {
            directionX /= directionLength;
            directionY /= directionLength;
        }
        relaxedCenters[i] = PointF{
            center.x + initialOffsets[i].x + directionX * requestedGap * 0.5f,
            center.y + initialOffsets[i].y + directionY * requestedGap * 0.5f};
    }

    const int iterationCount = islands.size() <= 512U ? 32 : (islands.size() <= 2048U ? 16 : 8);
    for (int iteration = 0; iteration < iterationCount; ++iteration) {
        bool changed = false;
        for (int first = 0; first < static_cast<int>(islands.size()); ++first) {
            for (int second = first + 1; second < static_cast<int>(islands.size()); ++second) {
                const float differenceX = relaxedCenters[second].x - relaxedCenters[first].x;
                const float differenceY = relaxedCenters[second].y - relaxedCenters[first].y;
                const float overlapX = halfExtents[first].x + halfExtents[second].x + requestedGap - std::fabs(differenceX);
                const float overlapY = halfExtents[first].y + halfExtents[second].y + requestedGap - std::fabs(differenceY);
                if (overlapX <= 0.0f || overlapY <= 0.0f) continue;

                changed = true;
                if (overlapX <= overlapY) {
                    const float direction = std::fabs(differenceX) > 1.0e-4f
                        ? (differenceX > 0.0f ? 1.0f : -1.0f)
                        : (first < second ? 1.0f : -1.0f);
                    const float correction = (overlapX + 0.01f) * 0.5f;
                    relaxedCenters[first].x -= direction * correction;
                    relaxedCenters[second].x += direction * correction;
                } else {
                    const float direction = std::fabs(differenceY) > 1.0e-4f
                        ? (differenceY > 0.0f ? 1.0f : -1.0f)
                        : (first < second ? 1.0f : -1.0f);
                    const float correction = (overlapY + 0.01f) * 0.5f;
                    relaxedCenters[first].y -= direction * correction;
                    relaxedCenters[second].y += direction * correction;
                }
            }
        }
        if (!changed) break;
    }

    applicationStrength = std::max(0.0f, std::min(1.0f, applicationStrength));
    for (int i = 0; i < static_cast<int>(islands.size()); ++i) {
        const PointF solvedOffset = PointF{
            relaxedCenters[i].x - originalCenters[i].x,
            relaxedCenters[i].y - originalCenters[i].y};
        offsets[i] = PointF{
            initialOffsets[i].x + (solvedOffset.x - initialOffsets[i].x) * applicationStrength,
            initialOffsets[i].y + (solvedOffset.y - initialOffsets[i].y) * applicationStrength};
    }
    return offsets;
}

static bool BuildVectorGeometry(
    const std::vector<AnalysisSample>& samples,
    int analysisCols,
    int analysisRows,
    int analysisStep,
    int inputWidth,
    int inputHeight,
    int inputOriginX,
    int inputOriginY,
    int regionSize,
    float colorTolerance,
    float simplify,
    float shapeJitter,
    float explode,
    float relax,
    int seed,
    const std::vector<PaletteEntry>& palette,
    std::vector<VectorTriangle>& triangles)
{
    const int cellCount = analysisCols * analysisRows;
    if (cellCount <= 0 || samples.size() != static_cast<size_t>(cellCount) || palette.empty()) return false;
    const int transparentLabel = static_cast<int>(palette.size());
    std::vector<int> labels(cellCount);
    for (int i = 0; i < cellCount; ++i) {
        labels[i] = samples[i].alpha <= 0.005f ? transparentLabel : NearestPaletteHost(samples[i], palette);
    }

    DisjointSet cellSets(cellCount);
    const float toleranceDelta = colorTolerance * 50.0f;
    const float toleranceSquared = toleranceDelta * toleranceDelta;
    for (int y = 0; y < analysisRows; ++y) {
        for (int x = 0; x < analysisCols; ++x) {
            const int index = y * analysisCols + x;
            if (x + 1 < analysisCols) {
                const int right = index + 1;
                if (ComponentsCanConnect(samples[index], labels[index], samples[right], labels[right], transparentLabel, toleranceSquared)) {
                    cellSets.Unite(index, right);
                }
            }
            if (y + 1 < analysisRows) {
                const int below = index + analysisCols;
                if (ComponentsCanConnect(samples[index], labels[index], samples[below], labels[below], transparentLabel, toleranceSquared)) {
                    cellSets.Unite(index, below);
                }
            }
        }
    }

    struct ComponentInfo {
        int cells = 0;
        double sumL = 0.0;
        double sumLabA = 0.0;
        double sumLabB = 0.0;
        bool transparent = false;
    };
    std::vector<int> rootToComponent(cellCount, -1);
    std::vector<int> componentOfCell(cellCount, -1);
    std::vector<ComponentInfo> components;
    for (int i = 0; i < cellCount; ++i) {
        const int root = cellSets.Find(i);
        if (rootToComponent[root] < 0) {
            rootToComponent[root] = static_cast<int>(components.size());
            components.push_back(ComponentInfo{});
            components.back().transparent = labels[i] == transparentLabel;
        }
        const int component = rootToComponent[root];
        componentOfCell[i] = component;
        components[component].cells += 1;
        components[component].sumL += samples[i].l;
        components[component].sumLabA += samples[i].labA;
        components[component].sumLabB += samples[i].labB;
    }

    std::vector<std::vector<int>> adjacency(components.size());
    for (int y = 0; y < analysisRows; ++y) {
        for (int x = 0; x < analysisCols; ++x) {
            const int index = y * analysisCols + x;
            const int component = componentOfCell[index];
            if (x + 1 < analysisCols) {
                const int other = componentOfCell[index + 1];
                if (component != other) {
                    adjacency[component].push_back(other);
                    adjacency[other].push_back(component);
                }
            }
            if (y + 1 < analysisRows) {
                const int other = componentOfCell[index + analysisCols];
                if (component != other) {
                    adjacency[component].push_back(other);
                    adjacency[other].push_back(component);
                }
            }
        }
    }
    for (std::vector<int>& neighbors : adjacency) {
        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
    }

    const int nominalDiameter = std::max(1, regionSize / std::max(1, analysisStep));
    // Size 1 preserves even single-cell color features.
    const int minimumRegionCells = std::max(1, nominalDiameter * nominalDiameter / 5);
    std::vector<int> mergeTarget(components.size(), -1);
    for (int component = 0; component < static_cast<int>(components.size()); ++component) {
        const ComponentInfo& current = components[component];
        if (current.cells >= minimumRegionCells) continue;
        const double currentInverse = 1.0 / std::max(1, current.cells);
        const float currentL = static_cast<float>(current.sumL * currentInverse);
        const float currentA = static_cast<float>(current.sumLabA * currentInverse);
        const float currentB = static_cast<float>(current.sumLabB * currentInverse);
        float bestScore = std::numeric_limits<float>::max();
        for (const int neighbor : adjacency[component]) {
            const ComponentInfo& candidate = components[neighbor];
            if (candidate.transparent != current.transparent) continue;
            if (candidate.cells < current.cells || (candidate.cells == current.cells && neighbor > component)) continue;
            const double neighborInverse = 1.0 / std::max(1, candidate.cells);
            const float distance = current.transparent ? 0.0f : LabDistanceSquared(
                currentL,
                currentA,
                currentB,
                static_cast<float>(candidate.sumL * neighborInverse),
                static_cast<float>(candidate.sumLabA * neighborInverse),
                static_cast<float>(candidate.sumLabB * neighborInverse));
            const float score = distance + 200.0f / static_cast<float>(candidate.cells);
            if (score < bestScore) {
                bestScore = score;
                mergeTarget[component] = neighbor;
            }
        }
    }

    DisjointSet componentSets(static_cast<int>(components.size()));
    for (int component = 0; component < static_cast<int>(mergeTarget.size()); ++component) {
        if (mergeTarget[component] >= 0) componentSets.Unite(component, mergeTarget[component]);
    }

    std::vector<int> mergedRootToRegion(components.size(), -1);
    std::vector<int> cellRegions(cellCount, -1);
    std::vector<RegionInfo> regions;
    for (int i = 0; i < cellCount; ++i) {
        const int mergedRoot = componentSets.Find(componentOfCell[i]);
        if (mergedRootToRegion[mergedRoot] < 0) {
            mergedRootToRegion[mergedRoot] = static_cast<int>(regions.size());
            regions.push_back(RegionInfo{});
            regions.back().hash = Hash32(static_cast<unsigned int>(mergedRoot) ^ static_cast<unsigned int>(seed));
        }
        const int region = mergedRootToRegion[mergedRoot];
        cellRegions[i] = region;
        RegionInfo& info = regions[region];
        const double colorWeight = std::max(samples[i].alpha, 0.01f);
        ++info.cells;
        info.sumL += samples[i].l * colorWeight;
        info.sumLabA += samples[i].labA * colorWeight;
        info.sumLabB += samples[i].labB * colorWeight;
        info.sumR += samples[i].r * colorWeight;
        info.sumG += samples[i].g * colorWeight;
        info.sumB += samples[i].b * colorWeight;
        info.colorWeight += colorWeight;
        info.sumAlpha += samples[i].alpha;
        info.transparent = labels[i] == transparentLabel;
        const int cellX = i % analysisCols;
        const int cellY = i / analysisCols;
        const int localLeft = cellX * analysisStep;
        const int localTop = cellY * analysisStep;
        const int localRight = std::min(inputWidth, localLeft + analysisStep);
        const int localBottom = std::min(inputHeight, localTop + analysisStep);
        const float left = static_cast<float>(inputOriginX + localLeft);
        const float top = static_cast<float>(inputOriginY + localTop);
        const float right = static_cast<float>(inputOriginX + localRight);
        const float bottom = static_cast<float>(inputOriginY + localBottom);
        const double pixelWeight = static_cast<double>(localRight - localLeft) * (localBottom - localTop);
        info.pixelWeight += pixelWeight;
        info.sumX += (left + right) * 0.5 * pixelWeight;
        info.sumY += (top + bottom) * 0.5 * pixelWeight;
        info.minimumX = std::min(info.minimumX, left);
        info.minimumY = std::min(info.minimumY, top);
        info.maximumX = std::max(info.maximumX, right);
        info.maximumY = std::max(info.maximumY, bottom);
    }

    std::vector<std::vector<BoundaryEdge>> regionEdges(regions.size());
    for (int y = 0; y < analysisRows; ++y) {
        for (int x = 0; x < analysisCols; ++x) {
            const int index = y * analysisCols + x;
            const int region = cellRegions[index];
            std::vector<BoundaryEdge>& edges = regionEdges[region];
            if (y == 0 || cellRegions[index - analysisCols] != region) {
                edges.push_back(BoundaryEdge{x, y, x + 1, y, 0, false});
            }
            if (x + 1 == analysisCols || cellRegions[index + 1] != region) {
                edges.push_back(BoundaryEdge{x + 1, y, x + 1, y + 1, 1, false});
            }
            if (y + 1 == analysisRows || cellRegions[index + analysisCols] != region) {
                edges.push_back(BoundaryEdge{x + 1, y + 1, x, y + 1, 2, false});
            }
            if (x == 0 || cellRegions[index - 1] != region) {
                edges.push_back(BoundaryEdge{x, y + 1, x, y, 3, false});
            }
        }
    }

    const float simplifyTolerance = analysisStep * simplify * 2.62f;
    const float jitterAmount = analysisStep * 0.75f * shapeJitter;
    const float explodeAmount = regionSize * 1.5f * explode;
    std::vector<int> regionToRelaxIsland(regions.size(), -1);
    std::vector<AlphaIslandInfo> relaxIslands;
    std::vector<PointF> initialOffsets;
    relaxIslands.reserve(regions.size());
    initialOffsets.reserve(regions.size());
    for (int region = 0; region < static_cast<int>(regions.size()); ++region) {
        const RegionInfo& info = regions[region];
        if (info.transparent || info.pixelWeight <= 0.0) continue;
        AlphaIslandInfo island;
        island.pixelWeight = info.pixelWeight;
        island.sumX = info.sumX;
        island.sumY = info.sumY;
        island.minimumX = info.minimumX;
        island.minimumY = info.minimumY;
        island.maximumX = info.maximumX;
        island.maximumY = info.maximumY;
        regionToRelaxIsland[region] = static_cast<int>(relaxIslands.size());
        relaxIslands.push_back(island);

        const float offsetAngle = Hash01(info.hash ^ 0xa511e9b3U) * 6.28318530718f;
        const float offsetScale = explodeAmount * (0.35f + Hash01(info.hash ^ 0x63d83595U) * 0.65f);
        initialOffsets.push_back(PointF{
            cosf(offsetAngle) * offsetScale,
            sinf(offsetAngle) * offsetScale});
    }
    const std::vector<PointF> relaxOffsets = BuildRelaxOffsets(
        relaxIslands,
        initialOffsets,
        inputOriginX + inputWidth * 0.5f,
        inputOriginY + inputHeight * 0.5f,
        regionSize * relax,
        relax);
    for (int region = 0; region < static_cast<int>(regions.size()); ++region) {
        RegionInfo& info = regions[region];
        if (info.cells <= 0 || info.colorWeight <= 0.0) continue;
        AnalysisSample regionMean;
        const double inverseColorWeight = 1.0 / info.colorWeight;
        regionMean.l = static_cast<float>(info.sumL * inverseColorWeight);
        regionMean.labA = static_cast<float>(info.sumLabA * inverseColorWeight);
        regionMean.labB = static_cast<float>(info.sumLabB * inverseColorWeight);
        regionMean.r = static_cast<float>(info.sumR * inverseColorWeight);
        regionMean.g = static_cast<float>(info.sumG * inverseColorWeight);
        regionMean.b = static_cast<float>(info.sumB * inverseColorWeight);
        regionMean.alpha = static_cast<float>(info.sumAlpha / info.cells);
        const PaletteEntry& regionColor = palette[NearestPaletteHost(regionMean, palette)];
        const int relaxIsland = regionToRelaxIsland[region];
        const PointF relaxOffset = relaxIsland >= 0 && relaxIsland < static_cast<int>(relaxOffsets.size())
            ? relaxOffsets[relaxIsland]
            : PointF{0.0f, 0.0f};
        const float offsetX = relaxOffset.x;
        const float offsetY = relaxOffset.y;
        std::vector<std::vector<std::pair<int, int>>> loops = TraceLoops(regionEdges[region]);
        for (const std::vector<std::pair<int, int>>& gridLoop : loops) {
            std::vector<PointF> polygon;
            polygon.reserve(gridLoop.size());
            for (const std::pair<int, int>& point : gridLoop) {
                PointF converted;
                converted.x = static_cast<float>(inputOriginX + std::min(inputWidth, point.first * analysisStep));
                converted.y = static_cast<float>(inputOriginY + std::min(inputHeight, point.second * analysisStep));
                polygon.push_back(converted);
            }
            const float originalArea = SignedArea(polygon);
            if (originalArea <= 0.5f) continue;
            // At zero, remove only exactly collinear points, without a forced vertex budget.
            const int vertexBudget = simplify <= 0.0f
                ? static_cast<int>(polygon.size())
                : static_cast<int>(4096.0f + (256.0f - 4096.0f) * simplify);
            polygon = SimplifyPolygon(polygon, simplifyTolerance, vertexBudget);
            if (polygon.size() < 3) continue;
            for (PointF& point : polygon) {
                const int quantizedX = static_cast<int>(std::lround(point.x * 16.0f));
                const int quantizedY = static_cast<int>(std::lround(point.y * 16.0f));
                const unsigned int pointHash = Hash32(
                    static_cast<unsigned int>(quantizedX) * 0x9e3779b9U ^
                    static_cast<unsigned int>(quantizedY) * 0x85ebca6bU ^
                    static_cast<unsigned int>(seed));
                point.x += (Hash01(pointHash) * 2.0f - 1.0f) * jitterAmount + offsetX;
                point.y += (Hash01(pointHash ^ 0xc2b2ae35U) * 2.0f - 1.0f) * jitterAmount + offsetY;
            }
            const float priority = -std::log(std::max(1.0f, std::fabs(originalArea))) + Hash01(info.hash) * 1.0e-5f;
            const size_t triangleStart = triangles.size();
            if (!TriangulatePolygon(
                polygon,
                regionColor,
                Clamp01(regionMean.alpha),
                priority,
                info.hash,
                triangles)) {
                triangles.resize(triangleStart);
                return false; // Report a real geometry failure instead of silently losing one region.
            }
            if (triangles.size() > 200000U) return false;
        }
    }
    return true;
}

static bool BuildSpatialBins(
    const std::vector<VectorTriangle>& triangles,
    int outputWidth,
    int outputHeight,
    int outputOriginX,
    int outputOriginY,
    int binSize,
    int& binCols,
    int& binRows,
    std::vector<int>& binOffsets,
    std::vector<int>& binTriangleIndices)
{
    binCols = std::max(1, (outputWidth + binSize - 1) / binSize);
    binRows = std::max(1, (outputHeight + binSize - 1) / binSize);
    const int binCount = binCols * binRows;
    std::vector<int> counts(binCount, 0);

    const auto triangleBinRange = [&](const VectorTriangle& triangle, int& left, int& top, int& right, int& bottom) {
        const float minimumX = std::min(triangle.x0, std::min(triangle.x1, triangle.x2));
        const float maximumX = std::max(triangle.x0, std::max(triangle.x1, triangle.x2));
        const float minimumY = std::min(triangle.y0, std::min(triangle.y1, triangle.y2));
        const float maximumY = std::max(triangle.y0, std::max(triangle.y1, triangle.y2));
        left = static_cast<int>(std::floor((minimumX - outputOriginX) / binSize));
        right = static_cast<int>(std::floor((maximumX - outputOriginX) / binSize));
        top = static_cast<int>(std::floor((minimumY - outputOriginY) / binSize));
        bottom = static_cast<int>(std::floor((maximumY - outputOriginY) / binSize));
        left = std::max(0, std::min(binCols - 1, left));
        right = std::max(0, std::min(binCols - 1, right));
        top = std::max(0, std::min(binRows - 1, top));
        bottom = std::max(0, std::min(binRows - 1, bottom));
        return maximumX >= outputOriginX && minimumX < outputOriginX + outputWidth &&
            maximumY >= outputOriginY && minimumY < outputOriginY + outputHeight;
    };

    for (const VectorTriangle& triangle : triangles) {
        int left = 0;
        int top = 0;
        int right = 0;
        int bottom = 0;
        if (!triangleBinRange(triangle, left, top, right, bottom)) continue;
        for (int y = top; y <= bottom; ++y) {
            for (int x = left; x <= right; ++x) ++counts[y * binCols + x];
        }
    }

    binOffsets.assign(binCount + 1, 0);
    for (int i = 0; i < binCount; ++i) binOffsets[i + 1] = binOffsets[i] + counts[i];
    if (binOffsets.back() > 8000000) return false;
    binTriangleIndices.assign(binOffsets.back(), 0);
    std::vector<int> cursor = binOffsets;
    for (int triangleIndex = 0; triangleIndex < static_cast<int>(triangles.size()); ++triangleIndex) {
        int left = 0;
        int top = 0;
        int right = 0;
        int bottom = 0;
        if (!triangleBinRange(triangles[triangleIndex], left, top, right, bottom)) continue;
        for (int y = top; y <= bottom; ++y) {
            for (int x = left; x <= right; ++x) {
                binTriangleIndices[cursor[y * binCols + x]++] = triangleIndex;
            }
        }
    }
    return true;
}

static bool RunAbstractionCUDA(
    const void* input,
    void* output,
    int inputWidth,
    int inputHeight,
    int outputWidth,
    int outputHeight,
    int inputRowBytes,
    int outputRowBytes,
    int inputOriginX,
    int inputOriginY,
    int outputOriginX,
    int outputOriginY,
    int depth,
    int regionSize,
    float colorTolerance,
    int paletteColors,
    float simplify,
    float shapeJitter,
    float explode,
    int seed,
    float mix,
    float relax)
{
    if (!input || !output || inputWidth <= 0 || inputHeight <= 0 ||
        outputWidth <= 0 || outputHeight <= 0 || inputRowBytes <= 0 || outputRowBytes <= 0) {
        return false;
    }
    if (depth != 8 && depth != 16 && depth != 32) return false;

    regionSize = std::max(1, std::min(256, regionSize));
    colorTolerance = std::max(0.0f, std::min(1.0f, colorTolerance));
    paletteColors = std::max(2, std::min(256, paletteColors));
    simplify = std::max(0.0f, std::min(1.0f, simplify));
    shapeJitter = std::max(0.0f, std::min(1.0f, shapeJitter));
    explode = std::max(0.0f, std::min(2.0f, explode));
    mix = std::max(0.0f, std::min(1.0f, mix));
    relax = std::max(0.0f, std::min(2.0f, relax));

    int analysisStep = std::max(1, regionSize / 12);
    constexpr size_t kMaximumAnalysisCells = 4194304U;
    while (static_cast<size_t>((inputWidth + analysisStep - 1) / analysisStep) *
        static_cast<size_t>((inputHeight + analysisStep - 1) / analysisStep) > kMaximumAnalysisCells) {
        ++analysisStep;
    }
    const int analysisCols = (inputWidth + analysisStep - 1) / analysisStep;
    const int analysisRows = (inputHeight + analysisStep - 1) / analysisStep;
    const size_t analysisCount = static_cast<size_t>(analysisCols) * analysisRows;
    const size_t inputBytes = static_cast<size_t>(inputRowBytes) * inputHeight;
    const size_t outputBytes = static_cast<size_t>(outputRowBytes) * outputHeight;

    DeviceBuffer<char> deviceInput;
    DeviceBuffer<char> deviceOutput;
    DeviceBuffer<AnalysisSample> deviceSamples;
    if (!deviceInput.Allocate(inputBytes) ||
        !deviceOutput.Allocate(outputBytes) ||
        !deviceSamples.Allocate(analysisCount)) {
        return false;
    }
    if (cudaMemcpy(deviceInput.Get(), input, inputBytes, cudaMemcpyHostToDevice) != cudaSuccess) return false;

    const dim3 analysisBlock(16, 16);
    const dim3 analysisGrid(
        (analysisCols + analysisBlock.x - 1) / analysisBlock.x,
        (analysisRows + analysisBlock.y - 1) / analysisBlock.y);
    if (depth == 8) {
        AnalyzeKernel<Pixel8><<<analysisGrid, analysisBlock>>>(
            deviceInput.Get(), inputRowBytes, inputWidth, inputHeight,
            analysisStep, analysisCols, analysisRows, deviceSamples.Get());
    } else if (depth == 16) {
        AnalyzeKernel<Pixel16><<<analysisGrid, analysisBlock>>>(
            deviceInput.Get(), inputRowBytes, inputWidth, inputHeight,
            analysisStep, analysisCols, analysisRows, deviceSamples.Get());
    } else {
        AnalyzeKernel<Pixel32><<<analysisGrid, analysisBlock>>>(
            deviceInput.Get(), inputRowBytes, inputWidth, inputHeight,
            analysisStep, analysisCols, analysisRows, deviceSamples.Get());
    }
    if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) return false;

    std::vector<AnalysisSample> samples(analysisCount);
    if (cudaMemcpy(
        samples.data(),
        deviceSamples.Get(),
        analysisCount * sizeof(AnalysisSample),
        cudaMemcpyDeviceToHost) != cudaSuccess) {
        return false;
    }

    std::vector<PaletteEntry> palette = BuildPalette(samples, paletteColors, seed);
    std::vector<VectorTriangle> triangles;
    if (!BuildVectorGeometry(
        samples,
        analysisCols,
        analysisRows,
        analysisStep,
        inputWidth,
        inputHeight,
        inputOriginX,
        inputOriginY,
        regionSize,
        colorTolerance,
        simplify,
        shapeJitter,
        explode,
        relax,
        seed,
        palette,
        triangles)) {
        return false;
    }

    const int binSize = std::max(16, regionSize);
    int binCols = 0;
    int binRows = 0;
    std::vector<int> binOffsets;
    std::vector<int> binTriangleIndices;
    if (!BuildSpatialBins(
        triangles,
        outputWidth,
        outputHeight,
        outputOriginX,
        outputOriginY,
        binSize,
        binCols,
        binRows,
        binOffsets,
        binTriangleIndices)) {
        return false;
    }

    DeviceBuffer<VectorTriangle> deviceTriangles;
    DeviceBuffer<int> deviceBinOffsets;
    DeviceBuffer<int> deviceBinTriangleIndices;
    if (!deviceTriangles.Allocate(triangles.size()) ||
        !deviceBinOffsets.Allocate(binOffsets.size()) ||
        !deviceBinTriangleIndices.Allocate(binTriangleIndices.size())) {
        return false;
    }
    if (!triangles.empty() && cudaMemcpy(
        deviceTriangles.Get(),
        triangles.data(),
        triangles.size() * sizeof(VectorTriangle),
        cudaMemcpyHostToDevice) != cudaSuccess) {
        return false;
    }
    if (cudaMemcpy(
        deviceBinOffsets.Get(),
        binOffsets.data(),
        binOffsets.size() * sizeof(int),
        cudaMemcpyHostToDevice) != cudaSuccess) {
        return false;
    }
    if (!binTriangleIndices.empty() && cudaMemcpy(
        deviceBinTriangleIndices.Get(),
        binTriangleIndices.data(),
        binTriangleIndices.size() * sizeof(int),
        cudaMemcpyHostToDevice) != cudaSuccess) {
        return false;
    }

    const dim3 renderBlock(16, 16);
    const dim3 renderGrid(
        (outputWidth + renderBlock.x - 1) / renderBlock.x,
        (outputHeight + renderBlock.y - 1) / renderBlock.y);
    const bool sealGaps = explode == 0.0f && relax == 0.0f;
    const PixelF fallbackColor{palette.front().r,palette.front().g,palette.front().b,1};
#define LAUNCH_RENDER(PixelType) \
    RenderKernel<PixelType><<<renderGrid, renderBlock>>>( \
        deviceInput.Get(), deviceOutput.Get(), \
        inputWidth, inputHeight, outputWidth, outputHeight, \
        inputRowBytes, outputRowBytes, \
        inputOriginX, inputOriginY, outputOriginX, outputOriginY, \
        deviceTriangles.Get(), static_cast<int>(triangles.size()), \
        deviceBinOffsets.Get(), deviceBinTriangleIndices.Get(), \
        binSize, binCols, mix, sealGaps, fallbackColor)
    if (depth == 8) {
        LAUNCH_RENDER(Pixel8);
    } else if (depth == 16) {
        LAUNCH_RENDER(Pixel16);
    } else {
        LAUNCH_RENDER(Pixel32);
    }
#undef LAUNCH_RENDER

    if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) return false;
    if (cudaMemcpy(output, deviceOutput.Get(), outputBytes, cudaMemcpyDeviceToHost) != cudaSuccess) return false;
    return true;
}

} // namespace

extern "C" bool LaunchAbstractionCUDA(
    const void* input,
    void* output,
    int inputWidth,
    int inputHeight,
    int outputWidth,
    int outputHeight,
    int inputRowBytes,
    int outputRowBytes,
    int inputOriginX,
    int inputOriginY,
    int outputOriginX,
    int outputOriginY,
    int depth,
    int regionSize,
    float colorTolerance,
    int paletteColors,
    float simplify,
    float shapeJitter,
    float explode,
    int seed,
    float mix,
    float relax)
{
    try {
        return RunAbstractionCUDA(
            input,
            output,
            inputWidth,
            inputHeight,
            outputWidth,
            outputHeight,
            inputRowBytes,
            outputRowBytes,
            inputOriginX,
            inputOriginY,
            outputOriginX,
            outputOriginY,
            depth,
            regionSize,
            colorTolerance,
            paletteColors,
            simplify,
            shapeJitter,
            explode,
            seed,
            mix,
            relax);
    } catch (...) {
        return false;
    }
}
