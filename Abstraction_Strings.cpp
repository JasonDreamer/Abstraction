#include "Abstraction.h"

typedef struct {
    A_u_long index;
    A_char str[256];
} TableString;

TableString g_strs[StrID_NUMTYPES] = {
    StrID_NONE, "",
    StrID_Name, "Abstraction",
    StrID_Description, "Vectorized connected color regions.",
    StrID_RegionSize_Param_Name, "Region Size",
    StrID_ColorTolerance_Param_Name, "Color Tolerance",
    StrID_PaletteColors_Param_Name, "Palette Colors",
    StrID_Simplify_Param_Name, "Simplify",
    StrID_ShapeJitter_Param_Name, "Shape Jitter",
    StrID_Explode_Param_Name, "Explode",
    StrID_Seed_Param_Name, "Seed",
    StrID_Mix_Param_Name, "Mix",
    StrID_Relax_Param_Name, "Relax"
};

char* GetStringPtr(int strNum)
{
    return g_strs[strNum].str;
}
