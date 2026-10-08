#include "../Abstraction.cu"
#include <cstdio>
#include <cstdlib>

static void Check(bool ok, const char* message)
{
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", message); std::exit(1); }
}

static bool Covered(float x, float y, const std::vector<VectorTriangle>& triangles)
{
    for (const auto& t : triangles) {
        const PointF p{x,y}, a{t.x0,t.y0}, b{t.x1,t.y1}, c{t.x2,t.y2};
        const float u=Cross(a,b,p), v=Cross(b,c,p), w=Cross(c,a,p);
        if (!((u < -1e-5f || v < -1e-5f || w < -1e-5f) &&
              (u > 1e-5f || v > 1e-5f || w > 1e-5f))) return true;
    }
    return false;
}

static void CheckContour(const std::vector<PointF>& polygon)
{
    const PaletteEntry color{50,0,0,1,0,0};
    std::vector<VectorTriangle> triangles;
    const bool success = TriangulatePolygon(polygon, color, 1, 0, 1, triangles);
    Check(success, "a repeated boundary vertex must not delete an entire color region");
    // Compare tessellation to independent even-odd contour membership, including gaps.
    for (int y = -4; y < 40; ++y) for (int x = -4; x < 40; ++x) {
        const float px=x*.25f+.073f, py=y*.25f+.117f;
        bool inside=false;
        for (size_t i=0,j=polygon.size()-1; i<polygon.size(); j=i++) {
            const auto a=polygon[i], b=polygon[j];
            if ((a.y>py)!=(b.y>py) && px<(b.x-a.x)*(py-a.y)/(b.y-a.y)+a.x) inside=!inside;
        }
        if (Covered(px,py,triangles)!=inside) std::fprintf(stderr,"vertices=%zu x=%f y=%f expected=%d\n",polygon.size(),px,py,inside);
        Check(Covered(px,py,triangles)==inside, "vector fill must preserve lobes and true gaps");
    }
}

int main()
{
    CheckContour({{0,0},{4,0},{4,4},{8,4},{8,8},{4,8},{4,4},{0,4}});
    CheckContour({{0,0},{8,8},{0,8},{8,0}}); // crossing introduced by simplification/jitter
    CheckContour({{0,0},{8,0},{8,8},{0,8},{0,0},{2,2},{2,6},{6,6},{6,2},{2,2}}); // hole with retraced bridge
    const std::vector<PaletteEntry> palette{{30,0,0,.3f,.3f,.3f},{80,0,0,.8f,.8f,.8f}};
    for (unsigned int seed=0; seed<6; ++seed) {
        std::vector<AnalysisSample> cells;
        for (int i=0; i<32*32; ++i) {
            const auto& c=palette[Hash32(i+seed*4096)%2];
            cells.push_back(AnalysisSample{c.l,0,0,c.r,c.g,c.b,1});
        }
        std::vector<VectorTriangle> triangles;
        Check(BuildVectorGeometry(cells,32,32,1,32,32,0,0,1,0,0,0,0,0,1,palette,triangles), "complex connected-region extraction");
        for (int y=0; y<32; ++y) for (int x=0; x<32; ++x)
            Check(Covered(x+.5f,y+.5f,triangles), "opaque source must not lose entire regions at Simplify zero");
    }
    std::puts("PASS: touching/crossing/holed contours and six complex opaque region maps retain vector coverage.");
    return 0;
}
