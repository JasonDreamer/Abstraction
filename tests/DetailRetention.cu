// Standalone host-side regression checks using the actual vectorization code.
#include "../Abstraction.cu"
#include <cstdio>
#include <cstdlib>

static void Check(bool condition, const char* message)
{
    if (!condition) { std::fprintf(stderr, "FAIL: %s\n", message); std::exit(1); }
}

int main()
{
    std::vector<AnalysisSample> colors;
    for (int i = 0; i < 128; ++i) {
        const PixelF rgb{(i % 8) / 7.0f, ((i / 8) % 4) / 3.0f, (i / 32) / 3.0f, 1.0f};
        const LabColor lab = RgbToLab(rgb);
        colors.push_back(AnalysisSample{lab.l, lab.a, lab.b, rgb.r, rgb.g, rgb.b, 1.0f});
    }
    const auto palette = BuildPalette(colors, 128, 1);
    Check(palette.size() == 128, "128-color palette retained");
    for (const auto& sample : colors) {
        const auto& color = palette[NearestPaletteHost(sample, palette)];
        Check(LabDistanceSquared(sample.l, sample.labA, sample.labB, color.l, color.labA, color.labB) < 0.001f,
            "distinct source colors survive palette construction");
    }

    // Different palette labels only one Lab unit apart must not be merged at tolerance zero.
    std::vector<PaletteEntry> closePalette{{50, 0, 0, .4f, .4f, .4f}, {51, 0, 0, .41f, .41f, .41f}};
    std::vector<AnalysisSample> cells;
    for (int y = 0; y < 16; ++y) for (int x = 0; x < 16; ++x) {
        const auto& c = closePalette[(x + y) % 2];
        cells.push_back(AnalysisSample{c.l, c.labA, c.labB, c.r, c.g, c.b, 1});
    }
    std::vector<VectorTriangle> triangles;
    Check(BuildVectorGeometry(cells, 16, 16, 1, 16, 16, 0, 0, 1, 0, 0, 0, 0, 0, 1, closePalette, triangles),
        "detail geometry succeeds");
    Check(triangles.size() == 512, "all 256 single-pixel islands survive size=1 tolerance=0");

    std::vector<PointF> contour;
    for (int i = 0; i < 600; ++i) {
        const float angle = i * 6.28318530718f / 600;
        const float radius = i % 2 ? 100.0f : 110.0f;
        contour.push_back(PointF{std::cos(angle) * radius, std::sin(angle) * radius});
    }
    Check(SimplifyPolygon(contour, 0, static_cast<int>(contour.size())).size() == 600,
        "zero simplify preserves non-collinear contour vertices");
    std::puts("PASS: 128 colors, 256 one-pixel regions, 600 contour vertices retained.");
    return 0;
}
