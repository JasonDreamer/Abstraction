#include "Abstraction.h"

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
    float relax);

namespace {

struct RenderParams {
    int regionSize;
    float colorTolerance;
    int paletteColors;
    float simplify;
    float shapeJitter;
    float explode;
    int seed;
    float mix;
    float relax;
};

static PF_Err GetPixelFormat(PF_InData* in_data, PF_EffectWorld* world, PF_PixelFormat* format)
{
    const void* acquiredSuite = nullptr;
    const SPErr acquireErr = in_data->pica_basicP->AcquireSuite(
        kPFWorldSuite,
        kPFWorldSuiteVersion2,
        &acquiredSuite);
    if (acquireErr || !acquiredSuite) return PF_Err_BAD_CALLBACK_PARAM;

    PF_WorldSuite2* worldSuite = const_cast<PF_WorldSuite2*>(
        static_cast<const PF_WorldSuite2*>(acquiredSuite));
    PF_Err err = worldSuite->PF_GetPixelFormat(world, format);
    const SPErr releaseErr = in_data->pica_basicP->ReleaseSuite(kPFWorldSuite, kPFWorldSuiteVersion2);
    return err ? err : (releaseErr ? PF_Err_BAD_CALLBACK_PARAM : PF_Err_NONE);
}

static int PixelFormatToDepth(PF_PixelFormat format)
{
    if (format == PF_PixelFormat_ARGB32) return 8;
    if (format == PF_PixelFormat_ARGB64) return 16;
    if (format == PF_PixelFormat_ARGB128) return 32;
    return 0;
}

static PF_Err CopyAlignedWorld(
    PF_InData* in_data,
    PF_EffectWorld* input,
    PF_EffectWorld* output)
{
    PF_Err err = PF_FILL(nullptr, nullptr, output);
    if (err) return err;

    const A_long left = input->origin_x > output->origin_x ? input->origin_x : output->origin_x;
    const A_long top = input->origin_y > output->origin_y ? input->origin_y : output->origin_y;
    const A_long inputRight = input->origin_x + input->width;
    const A_long outputRight = output->origin_x + output->width;
    const A_long inputBottom = input->origin_y + input->height;
    const A_long outputBottom = output->origin_y + output->height;
    const A_long right = inputRight < outputRight ? inputRight : outputRight;
    const A_long bottom = inputBottom < outputBottom ? inputBottom : outputBottom;
    if (left >= right || top >= bottom) return PF_Err_NONE;

    PF_Rect sourceRect = {
        left - input->origin_x,
        top - input->origin_y,
        right - input->origin_x,
        bottom - input->origin_y};
    PF_Rect destinationRect = {
        left - output->origin_x,
        top - output->origin_y,
        right - output->origin_x,
        bottom - output->origin_y};
    return PF_COPY(input, output, &sourceRect, &destinationRect);
}

static RenderParams ParamsFromLegacy(PF_ParamDef* params[])
{
    RenderParams result{};
    result.regionSize = static_cast<int>(params[ABSTRACTION_REGION_SIZE]->u.fs_d.value);
    result.colorTolerance = static_cast<float>(params[ABSTRACTION_COLOR_TOLERANCE]->u.fs_d.value / 100.0);
    result.paletteColors = static_cast<int>(params[ABSTRACTION_PALETTE_COLORS]->u.fs_d.value);
    result.simplify = static_cast<float>(params[ABSTRACTION_SIMPLIFY]->u.fs_d.value / 100.0);
    result.shapeJitter = static_cast<float>(params[ABSTRACTION_SHAPE_JITTER]->u.fs_d.value / 100.0);
    result.explode = static_cast<float>(params[ABSTRACTION_EXPLODE]->u.fs_d.value / 100.0);
    result.seed = static_cast<int>(params[ABSTRACTION_SEED]->u.fs_d.value);
    result.mix = static_cast<float>(params[ABSTRACTION_MIX]->u.fs_d.value / 100.0);
    result.relax = static_cast<float>(params[ABSTRACTION_RELAX]->u.fs_d.value / 100.0);
    return result;
}

static RenderParams ParamsFromChecked(const PF_ParamDef checked[])
{
    RenderParams result{};
    result.regionSize = static_cast<int>(checked[ABSTRACTION_REGION_SIZE].u.fs_d.value);
    result.colorTolerance = static_cast<float>(checked[ABSTRACTION_COLOR_TOLERANCE].u.fs_d.value / 100.0);
    result.paletteColors = static_cast<int>(checked[ABSTRACTION_PALETTE_COLORS].u.fs_d.value);
    result.simplify = static_cast<float>(checked[ABSTRACTION_SIMPLIFY].u.fs_d.value / 100.0);
    result.shapeJitter = static_cast<float>(checked[ABSTRACTION_SHAPE_JITTER].u.fs_d.value / 100.0);
    result.explode = static_cast<float>(checked[ABSTRACTION_EXPLODE].u.fs_d.value / 100.0);
    result.seed = static_cast<int>(checked[ABSTRACTION_SEED].u.fs_d.value);
    result.mix = static_cast<float>(checked[ABSTRACTION_MIX].u.fs_d.value / 100.0);
    result.relax = static_cast<float>(checked[ABSTRACTION_RELAX].u.fs_d.value / 100.0);
    return result;
}

static PF_Err ProcessCUDA(
    PF_InData* in_data,
    PF_EffectWorld* input,
    PF_EffectWorld* output,
    const RenderParams& params)
{
    if (!input || !output || !input->data || !output->data) return PF_Err_BAD_CALLBACK_PARAM;
    if (params.mix <= 0.0f) return CopyAlignedWorld(in_data, input, output);

    PF_PixelFormat format = PF_PixelFormat_INVALID;
    PF_Err err = GetPixelFormat(in_data, output, &format);
    if (err) return err;
    const int depth = PixelFormatToDepth(format);
    if (!depth) return PF_Err_BAD_CALLBACK_PARAM;

    const bool success = LaunchAbstractionCUDA(
        input->data,
        output->data,
        input->width,
        input->height,
        output->width,
        output->height,
        input->rowbytes,
        output->rowbytes,
        input->origin_x,
        input->origin_y,
        output->origin_x,
        output->origin_y,
        depth,
        params.regionSize,
        params.colorTolerance,
        params.paletteColors,
        params.simplify,
        params.shapeJitter,
        params.explode,
        params.seed,
        params.mix,
        params.relax);

    return success ? PF_Err_NONE : CopyAlignedWorld(in_data, input, output);
}

static PF_Err About(PF_InData* in_data, PF_OutData* out_data)
{
    AEGP_SuiteHandler suites(in_data->pica_basicP);
    suites.ANSICallbacksSuite1()->sprintf(
        out_data->return_msg,
        "%s v%d.%d\r%s",
        STR(StrID_Name),
        MAJOR_VERSION,
        MINOR_VERSION,
        STR(StrID_Description));
    return PF_Err_NONE;
}

static PF_Err GlobalSetup(PF_OutData* out_data)
{
    out_data->my_version = PF_VERSION(
        MAJOR_VERSION,
        MINOR_VERSION,
        BUG_VERSION,
        STAGE_VERSION,
        BUILD_VERSION);
    out_data->out_flags = PF_OutFlag_DEEP_COLOR_AWARE;
    out_data->out_flags2 = PF_OutFlag2_FLOAT_COLOR_AWARE | PF_OutFlag2_SUPPORTS_SMART_RENDER;
    return PF_Err_NONE;
}

static PF_Err ParamsSetup(PF_InData* in_data, PF_OutData* out_data)
{
    PF_Err err = PF_Err_NONE;
    PF_ParamDef def;

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX(
        STR(StrID_RegionSize_Param_Name),
        ABSTRACTION_REGION_SIZE_MIN,
        ABSTRACTION_REGION_SIZE_VALID_MAX,
        ABSTRACTION_REGION_SIZE_MIN,
        ABSTRACTION_REGION_SIZE_SLIDER_MAX,
        ABSTRACTION_REGION_SIZE_DFLT,
        PF_Precision_INTEGER, 0, 0, REGION_SIZE_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX(
        STR(StrID_ColorTolerance_Param_Name),
        ABSTRACTION_PERCENT_MIN,
        ABSTRACTION_PERCENT_MAX,
        ABSTRACTION_PERCENT_MIN,
        ABSTRACTION_PERCENT_MAX,
        ABSTRACTION_COLOR_TOLERANCE_DFLT,
        PF_Precision_TENTHS, 0, 0, COLOR_TOLERANCE_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX(
        STR(StrID_PaletteColors_Param_Name),
        ABSTRACTION_PALETTE_MIN,
        ABSTRACTION_PALETTE_MAX,
        ABSTRACTION_PALETTE_MIN,
        ABSTRACTION_PALETTE_MAX,
        ABSTRACTION_PALETTE_DFLT,
        PF_Precision_INTEGER, 0, 0, PALETTE_COLORS_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX(
        STR(StrID_Simplify_Param_Name),
        ABSTRACTION_PERCENT_MIN,
        ABSTRACTION_PERCENT_MAX,
        ABSTRACTION_PERCENT_MIN,
        ABSTRACTION_PERCENT_MAX,
        ABSTRACTION_SIMPLIFY_DFLT,
        PF_Precision_TENTHS, 0, 0, SIMPLIFY_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX(
        STR(StrID_ShapeJitter_Param_Name),
        ABSTRACTION_PERCENT_MIN,
        ABSTRACTION_PERCENT_MAX,
        ABSTRACTION_PERCENT_MIN,
        ABSTRACTION_PERCENT_MAX,
        ABSTRACTION_SHAPE_JITTER_DFLT,
        PF_Precision_TENTHS, 0, 0, SHAPE_JITTER_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX(
        STR(StrID_Explode_Param_Name),
        ABSTRACTION_EXPLODE_MIN,
        ABSTRACTION_EXPLODE_MAX,
        ABSTRACTION_EXPLODE_MIN,
        ABSTRACTION_EXPLODE_MAX,
        ABSTRACTION_EXPLODE_DFLT,
        PF_Precision_TENTHS, 0, 0, EXPLODE_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX(
        STR(StrID_Seed_Param_Name),
        ABSTRACTION_SEED_MIN,
        ABSTRACTION_SEED_MAX,
        ABSTRACTION_SEED_MIN,
        ABSTRACTION_SEED_MAX,
        ABSTRACTION_SEED_DFLT,
        PF_Precision_INTEGER, 0, 0, SEED_DISK_ID);

    AEFX_CLR_STRUCT(def);
    def.ui_flags = PF_PUI_INVISIBLE;
    PF_ADD_FLOAT_SLIDER(
        "",
        ABSTRACTION_PERCENT_MIN,
        ABSTRACTION_PERCENT_MAX,
        ABSTRACTION_PERCENT_MIN,
        ABSTRACTION_PERCENT_MAX,
        AEFX_DEFAULT_CURVE_TOLERANCE,
        ABSTRACTION_RESERVED_8_DFLT,
        PF_Precision_TENTHS,
        0,
        false,
        RESERVED_8_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX(
        STR(StrID_Mix_Param_Name),
        ABSTRACTION_PERCENT_MIN,
        ABSTRACTION_PERCENT_MAX,
        ABSTRACTION_PERCENT_MIN,
        ABSTRACTION_PERCENT_MAX,
        ABSTRACTION_MIX_DFLT,
        PF_Precision_TENTHS, 0, 0, MIX_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX(
        STR(StrID_Relax_Param_Name),
        ABSTRACTION_RELAX_MIN,
        ABSTRACTION_RELAX_MAX,
        ABSTRACTION_RELAX_MIN,
        ABSTRACTION_RELAX_MAX,
        ABSTRACTION_RELAX_DFLT,
        PF_Precision_TENTHS, 0, 0, RELAX_DISK_ID);

    out_data->num_params = ABSTRACTION_NUM_PARAMS;
    return err;
}

static PF_Err PreRender(PF_InData* in_data, PF_PreRenderExtra* extra)
{
    PF_Err err = PF_Err_NONE;
    PF_RenderRequest request = extra->input->output_request;
    PF_CheckoutResult inputResult;
    AEFX_CLR_STRUCT(inputResult);
    request.preserve_rgb_of_zero_alpha = TRUE;
    request.rect.left = 0;
    request.rect.top = 0;
    request.rect.right = in_data->width;
    request.rect.bottom = in_data->height;
    ERR(extra->cb->checkout_layer(
        in_data->effect_ref,
        ABSTRACTION_INPUT,
        ABSTRACTION_INPUT,
        &request,
        in_data->current_time,
        in_data->time_step,
        in_data->time_scale,
        &inputResult));
    if (!err) {
        const PF_LRect& requested = extra->input->output_request.rect;
        extra->output->result_rect.left = inputResult.max_result_rect.left > requested.left
            ? inputResult.max_result_rect.left : requested.left;
        extra->output->result_rect.top = inputResult.max_result_rect.top > requested.top
            ? inputResult.max_result_rect.top : requested.top;
        extra->output->result_rect.right = inputResult.max_result_rect.right < requested.right
            ? inputResult.max_result_rect.right : requested.right;
        extra->output->result_rect.bottom = inputResult.max_result_rect.bottom < requested.bottom
            ? inputResult.max_result_rect.bottom : requested.bottom;
        if (extra->output->result_rect.right <= extra->output->result_rect.left ||
            extra->output->result_rect.bottom <= extra->output->result_rect.top) {
            AEFX_CLR_STRUCT(extra->output->result_rect);
        }
        extra->output->max_result_rect = inputResult.max_result_rect;
    }
    return err;
}

static PF_Err SmartRender(PF_InData* in_data, PF_SmartRenderExtra* extra)
{
    PF_Err err = PF_Err_NONE;
    PF_Err checkinErr = PF_Err_NONE;
    PF_EffectWorld* input = nullptr;
    PF_EffectWorld* output = nullptr;
    PF_ParamDef checked[ABSTRACTION_NUM_PARAMS];
    bool checkedOut[ABSTRACTION_NUM_PARAMS] = {};

    for (int i = ABSTRACTION_REGION_SIZE; i < ABSTRACTION_NUM_PARAMS && !err; ++i) {
        AEFX_CLR_STRUCT(checked[i]);
        err = PF_CHECKOUT_PARAM(
            in_data,
            i,
            in_data->current_time,
            in_data->time_step,
            in_data->time_scale,
            &checked[i]);
        checkedOut[i] = err == PF_Err_NONE;
    }

    if (!err) err = extra->cb->checkout_layer_pixels(in_data->effect_ref, ABSTRACTION_INPUT, &input);
    if (!err) err = extra->cb->checkout_output(in_data->effect_ref, &output);
    if (!err && input && output) err = ProcessCUDA(in_data, input, output, ParamsFromChecked(checked));

    if (input) {
        checkinErr = extra->cb->checkin_layer_pixels(in_data->effect_ref, ABSTRACTION_INPUT);
        if (!err) err = checkinErr;
    }
    for (int i = ABSTRACTION_REGION_SIZE; i < ABSTRACTION_NUM_PARAMS; ++i) {
        if (checkedOut[i]) {
            checkinErr = PF_CHECKIN_PARAM(in_data, &checked[i]);
            if (!err) err = checkinErr;
        }
    }
    return err;
}

static PF_Err Render(PF_InData* in_data, PF_ParamDef* params[], PF_LayerDef* output)
{
    return ProcessCUDA(in_data, &params[ABSTRACTION_INPUT]->u.ld, output, ParamsFromLegacy(params));
}

} // namespace

extern "C" DllExport
PF_Err PluginDataEntryFunction2(
    PF_PluginDataPtr inPtr,
    PF_PluginDataCB2 inPluginDataCallBackPtr,
    SPBasicSuite* inSPBasicSuitePtr,
    const char* inHostName,
    const char* inHostVersion)
{
    PF_Err result = PF_Err_INVALID_CALLBACK;
    result = PF_REGISTER_EFFECT_EXT2(
        inPtr,
        inPluginDataCallBackPtr,
        "Abstraction",
        "mknb:Abstraction",
        "mknb_Plugin",
        AE_RESERVED_INFO,
        "EffectMain",
        "https://www.adobe.com");
    return result;
}

PF_Err EffectMain(
    PF_Cmd cmd,
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_ParamDef* params[],
    PF_LayerDef* output,
    void* extra)
{
    PF_Err err = PF_Err_NONE;
    try {
        switch (cmd) {
            case PF_Cmd_ABOUT:
                err = About(in_data, out_data);
                break;
            case PF_Cmd_GLOBAL_SETUP:
                err = GlobalSetup(out_data);
                break;
            case PF_Cmd_PARAMS_SETUP:
                err = ParamsSetup(in_data, out_data);
                break;
            case PF_Cmd_SMART_PRE_RENDER:
                err = PreRender(in_data, reinterpret_cast<PF_PreRenderExtra*>(extra));
                break;
            case PF_Cmd_SMART_RENDER:
                err = SmartRender(in_data, reinterpret_cast<PF_SmartRenderExtra*>(extra));
                break;
            case PF_Cmd_RENDER:
                err = Render(in_data, params, output);
                break;
            default:
                break;
        }
    } catch (PF_Err& thrownErr) {
        err = thrownErr;
    }
    return err;
}
