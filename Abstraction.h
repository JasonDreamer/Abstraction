#pragma once

#ifndef ABSTRACTION_H
#define ABSTRACTION_H

typedef unsigned char u_char;
typedef unsigned short u_short;
typedef unsigned short u_int16;
typedef unsigned long u_long;
typedef short int int16;

#define PF_TABLE_BITS 12
#define PF_TABLE_SZ_16 4096
#define PF_DEEP_COLOR_AWARE 1

#include "AEConfig.h"

#ifdef AE_OS_WIN
#ifndef NOMINMAX
#define NOMINMAX
#endif
typedef unsigned short PixelType;
#include <Windows.h>
#endif

#include "entry.h"
#include "AE_Effect.h"
#include "AE_EffectCB.h"
#include "AE_Macros.h"
#include "Param_Utils.h"
#include "AE_EffectCBSuites.h"
#include "String_Utils.h"
#include "AE_GeneralPlug.h"
#include "AEFX_ChannelDepthTpl.h"
#include "AEGP_SuiteHandler.h"

#include "Abstraction_Strings.h"

#define MAJOR_VERSION 1
#define MINOR_VERSION 7
#define BUG_VERSION 0
#define STAGE_VERSION PF_Stage_DEVELOP
#define BUILD_VERSION 1

#define ABSTRACTION_REGION_SIZE_MIN 1.0
#define ABSTRACTION_REGION_SIZE_SLIDER_MAX 160.0
#define ABSTRACTION_REGION_SIZE_VALID_MAX 256.0
#define ABSTRACTION_REGION_SIZE_DFLT 8.0

#define ABSTRACTION_PERCENT_MIN 0.0
#define ABSTRACTION_PERCENT_MAX 100.0
#define ABSTRACTION_COLOR_TOLERANCE_DFLT 0.0
#define ABSTRACTION_SIMPLIFY_DFLT 10.0
#define ABSTRACTION_SHAPE_JITTER_DFLT 0.0
#define ABSTRACTION_MIX_DFLT 100.0
#define ABSTRACTION_RESERVED_8_DFLT 0.0

#define ABSTRACTION_PALETTE_MIN 2.0
#define ABSTRACTION_PALETTE_MAX 256.0
#define ABSTRACTION_PALETTE_DFLT 64.0

#define ABSTRACTION_EXPLODE_MIN 0.0
#define ABSTRACTION_EXPLODE_MAX 200.0
#define ABSTRACTION_EXPLODE_DFLT 0.0

#define ABSTRACTION_RELAX_MIN 0.0
#define ABSTRACTION_RELAX_MAX 200.0
#define ABSTRACTION_RELAX_DFLT 0.0

#define ABSTRACTION_SEED_MIN 0.0
#define ABSTRACTION_SEED_MAX 10000.0
#define ABSTRACTION_SEED_DFLT 1.0

enum {
    ABSTRACTION_INPUT = 0,
    ABSTRACTION_REGION_SIZE,
    ABSTRACTION_COLOR_TOLERANCE,
    ABSTRACTION_PALETTE_COLORS,
    ABSTRACTION_SIMPLIFY,
    ABSTRACTION_SHAPE_JITTER,
    ABSTRACTION_EXPLODE,
    ABSTRACTION_SEED,
    ABSTRACTION_RESERVED_8,
    ABSTRACTION_MIX,
    ABSTRACTION_RELAX,
    ABSTRACTION_NUM_PARAMS
};

enum {
    REGION_SIZE_DISK_ID = 1,
    COLOR_TOLERANCE_DISK_ID,
    PALETTE_COLORS_DISK_ID,
    SIMPLIFY_DISK_ID,
    SHAPE_JITTER_DISK_ID,
    EXPLODE_DISK_ID,
    SEED_DISK_ID,
    RESERVED_8_DISK_ID,
    MIX_DISK_ID,
    RELAX_DISK_ID
};

extern "C" {

DllExport
PF_Err EffectMain(
    PF_Cmd cmd,
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_ParamDef* params[],
    PF_LayerDef* output,
    void* extra);

}

#endif
