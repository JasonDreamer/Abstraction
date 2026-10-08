#include "../Abstraction.cu"
#include <cstdio>
#include <cstdlib>

static void Check(bool condition, const char* message)
{
    if (!condition) { std::fprintf(stderr, "FAIL: %s\n", message); std::exit(1); }
}

template <typename P>
static void CheckDepth(float maximum)
{
    using Channel = decltype(P{}.alpha);
    // Opaque source with padded scanlines, a single simplified polygon and exposed gaps.
    std::vector<P> source(24, P{static_cast<Channel>(maximum), static_cast<Channel>(maximum), 0, 0});
    source[2*6+3] = P{0,0,0,0};
    source[3*6+2] = P{static_cast<Channel>(maximum*.5f),static_cast<Channel>(maximum*.5f),0,0};
    std::vector<P> output(24);
    const int rowBytes = 6 * sizeof(P);
    VectorTriangle triangle{0, 0, 2, 0, 0, 2, 0, 1, 0, 1, 1, 0};
    const int offsets[5] = {0, 1, 1, 1, 1};
    const int index = 0;
    DeviceBuffer<char> inputBuffer, outputBuffer;
    DeviceBuffer<VectorTriangle> triangleBuffer;
    DeviceBuffer<int> offsetBuffer, indexBuffer;
    Check(inputBuffer.Allocate(rowBytes * 4) && outputBuffer.Allocate(rowBytes * 4) &&
        triangleBuffer.Allocate(1) && offsetBuffer.Allocate(5) && indexBuffer.Allocate(1), "CUDA allocation");
    Check(cudaMemcpy(inputBuffer.Get(), source.data(), rowBytes * 4, cudaMemcpyHostToDevice) == cudaSuccess, "source upload");
    Check(cudaMemcpy(triangleBuffer.Get(), &triangle, sizeof(triangle), cudaMemcpyHostToDevice) == cudaSuccess, "geometry upload");
    Check(cudaMemcpy(offsetBuffer.Get(), offsets, sizeof(offsets), cudaMemcpyHostToDevice) == cudaSuccess, "bins upload");
    Check(cudaMemcpy(indexBuffer.Get(), &index, sizeof(index), cudaMemcpyHostToDevice) == cudaSuccess, "indices upload");
    for (const bool seal : {false,true}) for (const float mix : {1.0f, 0.5f, 0.0f}) {
        RenderKernel<P><<<dim3(1, 1), dim3(4, 4)>>>(inputBuffer.Get(), outputBuffer.Get(),
            4, 4, 4, 4, rowBytes, rowBytes, 0, 0, 0, 0,
            triangleBuffer.Get(), 1, offsetBuffer.Get(), indexBuffer.Get(), 2, 2, mix, seal, PixelF{0,0,1,1});
        Check(cudaGetLastError() == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess, "render kernel");
        Check(cudaMemcpy(output.data(), outputBuffer.Get(), rowBytes * 4, cudaMemcpyDeviceToHost) == cudaSuccess, "output download");
        const P gap = output[3 * 6 + 3];
        const float tolerance = maximum > 1 ? 1.1f / maximum : 1.0e-6f;
        Check(std::fabs(gap.alpha / maximum - (seal ? 1 : 1-mix)) < tolerance, "attached alpha preserved; detached gap controlled by Mix");
        Check(std::fabs(gap.red / maximum - (1 - mix)) < tolerance, "uncovered premultiplied RGB only comes from Mix");
        Check(std::fabs(gap.green / maximum - (seal ? mix : 0)) < tolerance, "seam uses nearest vector color across bins, never source RGB");
        if (mix == 1) {
            if (!seal) Check(gap.alpha == 0 && gap.red == 0 && gap.green == 0 && gap.blue == 0, "detached gap is transparent black");
            Check(output[0].green == maximum && output[0].alpha == maximum, "covered polygon remains opaque");
            if (seal) {
                Check(output[2*6+3].alpha == 0 && output[2*6+3].green == 0, "original transparent silhouette preserved");
                Check(output[3*6+2].alpha == source[3*6+2].alpha, "partial source alpha preserved");
            }
        }
    }
}

int main()
{
    CheckDepth<Pixel8>(255);
    CheckDepth<Pixel16>(32768);
    CheckDepth<Pixel32>(1);
    std::vector<Pixel32> source(64*64), output(source.size());
    for (int y=0;y<64;++y) for (int x=0;x<64;++x) {
        const float a = x<3 ? 0 : (x==3 ? .5f : 1);
        source[y*64+x] = Pixel32{a, a*(.5f+.4f*std::sin(x*.3f)), a*(.5f+.4f*std::cos(y*.2f)), a*.3f};
    }
    for (int size : {1,48,256}) for (float simplify : {0.0f,.1f,1.0f}) {
        Check(LaunchAbstractionCUDA(source.data(),output.data(),64,64,64,64,64*sizeof(Pixel32),64*sizeof(Pixel32),
            0,0,0,0,32,size,0,16,simplify,0,0,1,1,0), "full pipeline succeeds");
        for (size_t i=0;i<source.size();++i) Check(output[i].alpha==source[i].alpha,"Size/Simplify sweep preserves all input alpha");
    }
    std::puts("PASS: CUDA 8/16/32bpc gap sealing/detachment, nearest vector colors, alpha silhouette, Mix, rowbytes, full Size/Simplify sweep.");
}
