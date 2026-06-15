#import "VideoDecoderRenderer.h"
#import "StreamView.h"
#include <libavcodec/avcodec.h>
#include <libavcodec/cbs.h>
#include <libavcodec/cbs_av1.h>
#include <libavformat/avio.h>
#include <libavutil/mem.h>
#import <VideoToolbox/VideoToolbox.h>
#import <Vision/Vision.h>
#import <CoreML/CoreML.h>
#import <CoreImage/CoreImage.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <stdatomic.h> // 引入原子锁机制

// ==========================================================
// ⚙️ 炼丹师专属调参区
// ==========================================================
// 降低阈值以提高灵敏度（原 0.45 -> 现 0.28）
#define AI_CONFIDENCE_THRESHOLD 0.28f
// 瞄准点下压比例（0.20 = 框的中心往下 20%，瞄准胸口）
#define AI_AIM_OFFSET 0.20f

// ==========================================================
// 📡 [M2 ANE] 异步纯旁路 AI 引擎 (满血防卡死版)
// ==========================================================
static int m2_udp_sock = -1;
static struct sockaddr_in m2_pc_addr;
static struct sockaddr_in m2_hud_addr;

static void init_logger_once(void) {
    if (m2_udp_sock != -1) return;
    m2_udp_sock = socket(AF_INET, SOCK_DGRAM, 0);
    fcntl(m2_udp_sock, F_SETFL, O_NONBLOCK);
    m2_pc_addr.sin_family = AF_INET;
    m2_pc_addr.sin_port = htons(9999);
    // 🚨🚨🚨 确保这是你的 PC IP
    inet_pton(AF_INET, "10.0.0.1", &m2_pc_addr.sin_addr);
    m2_hud_addr = m2_pc_addr;
    m2_hud_addr.sin_port = htons(9998);
}

static void M2_LOG(const char *format, ...) {
    init_logger_once();
    char buffer[512];
    va_list args;
    va_start(args, format);
    vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);
    sendto(m2_udp_sock, buffer, strlen(buffer), 0, (struct sockaddr *)&m2_pc_addr, sizeof(m2_pc_addr));
}

static VNCoreMLModel *m2_ai_model = nil;
static VNCoreMLRequest *m2_ai_request = nil;
static dispatch_queue_t m2_queue = nil;
static CIContext *m2_ci_context = nil;
static CFAbsoluteTime m2_last_hud_time = 0.0;
static CFAbsoluteTime m2_attachment_scan_until = 0.0;
static NSString *m2_ai_debug_text = @"AI waiting";

typedef struct {
    CGFloat x;
    CGFloat y;
    CGFloat w;
    CGFloat h;
} M2HudRect;

typedef struct {
    CGFloat r;
    CGFloat g;
    CGFloat b;
    CGFloat luma;
} M2HudColor;

static const CGFloat M2_HUD_BASE_W = 1280.0;
static const CGFloat M2_HUD_BASE_H = 720.0;
static const M2HudRect M2_SLOT1_BRIGHT = {1103, 693, 10, 9};
static const M2HudRect M2_SLOT2_BRIGHT = {1131, 695, 12, 8};
static const M2HudRect M2_BACKPACK_MARKER = {186, 558, 78, 18};
static const M2HudRect M2_BACKPACK_LEFT_NAME = {438, 228, 164, 18};
static const M2HudRect M2_BACKPACK_RIGHT_NAME = {804, 228, 103, 16};
static const M2HudRect M2_BACKPACK_LEFT_BARREL = {441, 304, 40, 3};
static const M2HudRect M2_BACKPACK_LEFT_SCOPE = {545, 304, 42, 3};
static const M2HudRect M2_BACKPACK_RIGHT_BARREL = {807, 304, 41, 3};
static const M2HudRect M2_BACKPACK_RIGHT_SCOPE = {911, 304, 41, 2};

@interface VideoDecoderRenderer ()
- (void)m2UpdateHudOverlayWithText:(NSString *)text activeSlot:(NSInteger)activeSlot attachmentScan:(BOOL)attachmentScan;
@end

void M2NotifyApexMenuButton(BOOL pressed) {
    (void)pressed;
}

// 🛡️ 核心修复 1：原子锁，防止 4K 120FPS 撑爆显存
static atomic_bool ai_is_busy = false;

static NSString *m2_safe_string(NSString *value) {
    return value ?: @"None";
}

static NSString *m2_json_escape(NSString *value) {
    NSMutableString *s = [NSMutableString stringWithString:m2_safe_string(value)];
    [s replaceOccurrencesOfString:@"\\" withString:@"\\\\" options:0 range:NSMakeRange(0, s.length)];
    [s replaceOccurrencesOfString:@"\"" withString:@"\\\"" options:0 range:NSMakeRange(0, s.length)];
    return s;
}

static CGRect m2_scaled_rect_for_buffer(M2HudRect r, CVImageBufferRef pix) {
    CGFloat pw = (CGFloat)CVPixelBufferGetWidth(pix);
    CGFloat ph = (CGFloat)CVPixelBufferGetHeight(pix);
    CGFloat sx = pw / M2_HUD_BASE_W;
    CGFloat sy = ph / M2_HUD_BASE_H;
    return CGRectMake(r.x * sx, ph - ((r.y + r.h) * sy), r.w * sx, r.h * sy);
}

static CGImageRef m2_create_crop_image(CVImageBufferRef pix, M2HudRect r, CGFloat upscale) {
    if (!pix) return nil;
    if (!m2_ci_context) m2_ci_context = [CIContext contextWithOptions:nil];
    CIImage *image = [CIImage imageWithCVPixelBuffer:pix];
    CGRect cropRect = m2_scaled_rect_for_buffer(r, pix);
    CIImage *cropped = [[image imageByCroppingToRect:cropRect] imageByApplyingTransform:CGAffineTransformMakeTranslation(-cropRect.origin.x, -cropRect.origin.y)];
    if (upscale > 1.0) {
        cropped = [cropped imageByApplyingTransform:CGAffineTransformMakeScale(upscale, upscale)];
    }
    return [m2_ci_context createCGImage:cropped fromRect:cropped.extent];
}

static M2HudColor m2_sample_color(CGImageRef image) {
    M2HudColor out = {0, 0, 0, 0};
    if (!image) return out;
    size_t width = CGImageGetWidth(image);
    size_t height = CGImageGetHeight(image);
    if (width == 0 || height == 0) return out;

    size_t bytesPerRow = width * 4;
    unsigned char *data = calloc(height, bytesPerRow);
    if (!data) return out;
    CGColorSpaceRef cs = CGColorSpaceCreateDeviceRGB();
    CGContextRef ctx = CGBitmapContextCreate(data, width, height, 8, bytesPerRow, cs, kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big);
    CGColorSpaceRelease(cs);
    if (!ctx) {
        free(data);
        return out;
    }

    CGContextDrawImage(ctx, CGRectMake(0, 0, width, height), image);
    double r = 0, g = 0, b = 0;
    size_t count = width * height;
    for (size_t i = 0; i < count; i++) {
        unsigned char *p = data + i * 4;
        r += p[0];
        g += p[1];
        b += p[2];
    }
    CGContextRelease(ctx);
    free(data);

    out.r = r / count;
    out.g = g / count;
    out.b = b / count;
    out.luma = out.r * 0.299 + out.g * 0.587 + out.b * 0.114;
    return out;
}

static CGFloat m2_color_distance(M2HudColor c, CGFloat r, CGFloat g, CGFloat b) {
    CGFloat dr = c.r - r, dg = c.g - g, db = c.b - b;
    return sqrt(dr * dr + dg * dg + db * db);
}

static NSString *m2_classify_equipment_color(M2HudColor c) {
    CGFloat dWhite = MIN(MIN(m2_color_distance(c, 114, 109, 104), m2_color_distance(c, 110, 107, 102)),
                         m2_color_distance(c, 181, 190, 193));
    CGFloat dBlue = MIN(MIN(m2_color_distance(c, 79, 104, 139), m2_color_distance(c, 66, 100, 141)),
                        MIN(MIN(m2_color_distance(c, 25, 84, 136), m2_color_distance(c, 22, 103, 163)),
                            MIN(MIN(m2_color_distance(c, 24, 110, 170), m2_color_distance(c, 34, 145, 206)),
                                MIN(m2_color_distance(c, 45, 171, 233), m2_color_distance(c, 47, 173, 237)))));
    CGFloat dPurple = MIN(MIN(MIN(m2_color_distance(c, 126, 104, 145), m2_color_distance(c, 129, 109, 148)),
                              MIN(m2_color_distance(c, 74, 43, 100), m2_color_distance(c, 107, 80, 127))),
                          MIN(m2_color_distance(c, 195, 116, 240), m2_color_distance(c, 171, 102, 215)));
    CGFloat dNone = MIN(m2_color_distance(c, 36, 38, 37), m2_color_distance(c, 34, 37, 36));
    if (dNone <= 30.0 || c.luma < 45.0) return @"None";
    CGFloat best = MIN(dWhite, MIN(dBlue, dPurple));
    if (best > 46.0) return @"None";
    if (dPurple <= dBlue && dPurple <= dWhite) return @"Purple";
    if (dBlue <= dWhite) return @"Blue";
    return @"White";
}

static NSString *m2_scope_for_color(NSString *color) {
    if ([color isEqualToString:@"Blue"] || [color isEqualToString:@"Purple"]) return @"S2x";
    if ([color isEqualToString:@"White"]) return @"S1x";
    return @"S1x";
}

static NSString *m2_classify_havoc_paintball(M2HudColor c) {
    CGFloat dActive = m2_color_distance(c, 157, 137, 104);
    CGFloat dInactive = m2_color_distance(c, 141, 141, 132);
    if (dActive < dInactive || (c.r - c.b >= 25.0 && c.r - c.g >= 8.0)) return @"Paintball";
    return @"None";
}

static NSString *m2_ocr_text(CGImageRef image) {
    if (!image) return @"";
    __block NSString *result = @"";
    if (@available(iOS 13.0, tvOS 13.0, *)) {
        VNRecognizeTextRequest *request = [[VNRecognizeTextRequest alloc] initWithCompletionHandler:^(VNRequest *req, NSError *err) {
            if (err) return;
            NSMutableArray<NSString *> *parts = [NSMutableArray array];
            for (VNRecognizedTextObservation *obs in req.results) {
                VNRecognizedText *top = [[obs topCandidates:1] firstObject];
                if (top.string.length > 0) [parts addObject:top.string];
            }
            result = [[parts componentsJoinedByString:@" "] copy];
        }];
        request.recognitionLevel = VNRequestTextRecognitionLevelAccurate;
        request.usesLanguageCorrection = NO;
        request.recognitionLanguages = @[@"zh-Hans", @"en-US"];
        request.minimumTextHeight = 0.08;
        VNImageRequestHandler *handler = [[VNImageRequestHandler alloc] initWithCGImage:image options:@{}];
        [handler performRequests:@[request] error:nil];
    }
    return [result stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
}

static NSString *m2_normalize_weapon(NSString *raw) {
    NSString *s = [[raw lowercaseString] stringByReplacingOccurrencesOfString:@" " withString:@""];
    s = [s stringByReplacingOccurrencesOfString:@"-" withString:@""];
    if (s.length == 0) return @"None";
    if ([s containsString:@"3"] && [s containsString:@"0"]) return @"R301";
    if ([s containsString:@"45"]) return @"RE45";
    if ([s containsString:@"c"] && [s containsString:@"a"]) return @"CAR";
    if ([s containsString:@"仇"]) return @"Nemesis";
    if ([s containsString:@"哈"] && [s containsString:@"克"]) return @"Havoc";
    if ([s containsString:@"姆"]) return @"Flatline";
    if ([s containsString:@"9"] && [s containsString:@"r"]) return @"R99";
    if ([s containsString:@"nemesis"] || [s containsString:@"复仇"]) return @"Nemesis";
    if ([s containsString:@"havoc"] || [s containsString:@"哈沃"]) return @"Havoc";
    if ([s containsString:@"r301"] || [s containsString:@"301"]) return @"R301";
    if ([s containsString:@"hemlok"] || [s containsString:@"赫姆"] || [s containsString:@"平行"] || [s containsString:@"flatline"]) return @"Flatline";
    if ([s containsString:@"r99"] || [s containsString:@"r-99"]) return @"R99";
    if ([s isEqualToString:@"car"] || [s containsString:@"c.a.r"] || [s containsString:@"car冲锋"]) return @"CAR";
    if ([s containsString:@"alternator"] || [s containsString:@"转换"]) return @"Alternator";
    if ([s containsString:@"prowler"] || [s containsString:@"猎兽"]) return @"Prowler";
    if ([s containsString:@"re45"]) return @"RE45";
    if ([s containsString:@"spitfire"] || [s containsString:@"喷火"]) return @"Spitfire";
    if ([s containsString:@"lstar"] || [s containsString:@"l-star"]) return @"LStar";
    if ([s containsString:@"devotion"] || [s containsString:@"专注"]) return @"Devotion";
    return @"None";
}

static BOOL m2_is_locked_1x_weapon(NSString *weapon) {
    return [@[@"R99", @"CAR", @"Alternator", @"Prowler", @"RE45"] containsObject:weapon];
}

static BOOL m2_uses_rifle_barrel(NSString *weapon) {
    return [@[@"Nemesis", @"R301"] containsObject:weapon];
}

static BOOL m2_uses_rifle_scope(NSString *weapon) {
    return [@[@"Nemesis", @"R301", @"Flatline"] containsObject:weapon];
}

static NSDictionary *m2_empty_slot(NSString *weapon) {
    return @{@"weapon": m2_safe_string(weapon),
             @"scope": @"Keep",
             @"scopeColor": @"Skipped",
             @"barrel": @"Keep",
             @"barrelColor": @"Skipped"};
}

static BOOL m2_backpack_marker_detected(CVImageBufferRef pix, NSString **rawTextOut) {
    CGImageRef markerImage = m2_create_crop_image(pix, M2_BACKPACK_MARKER, 4.0);
    NSString *raw = m2_ocr_text(markerImage);
    if (markerImage) CGImageRelease(markerImage);
    if (rawTextOut) *rawTextOut = raw ?: @"";

    NSString *s = [[raw lowercaseString] stringByReplacingOccurrencesOfString:@" " withString:@""];
    return [s containsString:@"平"] || [s containsString:@"ha"];
}

static M2HudRect m2_backpack_scope_rect(NSInteger active) {
    return active == 2 ? M2_BACKPACK_RIGHT_SCOPE : M2_BACKPACK_LEFT_SCOPE;
}

static M2HudRect m2_backpack_barrel_rect(NSInteger active) {
    return active == 2 ? M2_BACKPACK_RIGHT_BARREL : M2_BACKPACK_LEFT_BARREL;
}

static NSString *m2_backpack_weapon(CVImageBufferRef pix, NSInteger slot, NSString *fallback) {
    M2HudRect nameRect = slot == 2 ? M2_BACKPACK_RIGHT_NAME : M2_BACKPACK_LEFT_NAME;
    CGImageRef nameImage = m2_create_crop_image(pix, nameRect, 4.0);
    NSString *weapon = m2_normalize_weapon(m2_ocr_text(nameImage));
    if (nameImage) CGImageRelease(nameImage);
    return [weapon isEqualToString:@"None"] ? fallback : weapon;
}

static NSDictionary *m2_detect_slot(CVImageBufferRef pix, NSString *weapon, NSInteger slot) {
    if (!weapon || [weapon isEqualToString:@"None"] || [weapon isEqualToString:@"Keep"]) return m2_empty_slot(weapon ?: @"Keep");

    NSString *scope = @"S1x";
    NSString *scopeColor = @"None";
    NSString *barrel = @"None";
    NSString *barrelColor = @"None";

    if (m2_is_locked_1x_weapon(weapon)) {
        scopeColor = @"Locked";
        if ([weapon isEqualToString:@"Alternator"] || [weapon isEqualToString:@"Prowler"]) {
            barrel = @"Paintball";
            barrelColor = @"Locked";
        }
    } else if ([weapon isEqualToString:@"Havoc"]) {
        CGImageRef scopeImage = m2_create_crop_image(pix, m2_backpack_scope_rect(slot), 1.0);
        scopeColor = m2_classify_equipment_color(m2_sample_color(scopeImage));
        if (scopeImage) CGImageRelease(scopeImage);
        scope = m2_scope_for_color(scopeColor);

        CGImageRef paintImage = m2_create_crop_image(pix, m2_backpack_barrel_rect(slot), 1.0);
        barrel = m2_classify_havoc_paintball(m2_sample_color(paintImage));
        if (paintImage) CGImageRelease(paintImage);
        barrelColor = [barrel isEqualToString:@"Paintball"] ? @"Active" : @"None";
    } else if (m2_uses_rifle_scope(weapon)) {
        CGImageRef scopeImage = m2_create_crop_image(pix, m2_backpack_scope_rect(slot), 1.0);
        scopeColor = m2_classify_equipment_color(m2_sample_color(scopeImage));
        if (scopeImage) CGImageRelease(scopeImage);
        scope = m2_scope_for_color(scopeColor);

        if (m2_uses_rifle_barrel(weapon)) {
            CGImageRef barrelImage = m2_create_crop_image(pix, m2_backpack_barrel_rect(slot), 1.0);
            barrelColor = m2_classify_equipment_color(m2_sample_color(barrelImage));
            if (barrelImage) CGImageRelease(barrelImage);
            barrel = barrelColor;
        }
    }

    return @{@"weapon": weapon,
             @"scope": scope,
             @"scopeColor": scopeColor,
             @"barrel": barrel,
             @"barrelColor": barrelColor};
}

static NSString *m2_slot_json(NSDictionary *slot) {
    return [NSString stringWithFormat:@"{\"weapon\":\"%@\",\"scope\":\"%@\",\"scopeColor\":\"%@\",\"barrel\":\"%@\",\"barrelColor\":\"%@\"}",
            m2_json_escape(slot[@"weapon"]), m2_json_escape(slot[@"scope"]), m2_json_escape(slot[@"scopeColor"]),
            m2_json_escape(slot[@"barrel"]), m2_json_escape(slot[@"barrelColor"])];
}

static void m2_send_hud_json(NSDictionary *slot, NSDictionary *slot1, NSDictionary *slot2, NSInteger active, CGFloat l1, CGFloat l2, BOOL attachmentScan) {
    NSString *json = nil;
    if (attachmentScan) {
        json = [NSString stringWithFormat:
            @"{\"type\":\"apex_hud\",\"mode\":\"menu\",\"active\":%ld,\"attachmentsValid\":true,\"slot1\":%@,\"slot2\":%@,\"luma1\":%.1f,\"luma2\":%.1f}",
            (long)active, m2_slot_json(slot1), m2_slot_json(slot2), l1, l2];
    } else {
        json = [NSString stringWithFormat:
            @"{\"type\":\"apex_hud\",\"mode\":\"main\",\"active\":%ld,\"attachmentsValid\":false,\"slot\":%@,\"luma1\":%.1f,\"luma2\":%.1f}",
            (long)active, m2_slot_json(slot), l1, l2];
    }
    const char *payload = [json UTF8String];
    sendto(m2_udp_sock, payload, (int)strlen(payload), 0, (struct sockaddr *)&m2_hud_addr, sizeof(m2_hud_addr));
}

static void m2_run_hud(CVImageBufferRef pix, id renderer) {
    CFAbsoluteTime now = CFAbsoluteTimeGetCurrent();
    if (now - m2_last_hud_time < 0.20) return;
    m2_last_hud_time = now;

    CGImageRef bright1Image = m2_create_crop_image(pix, M2_SLOT1_BRIGHT, 1.0);
    CGImageRef bright2Image = m2_create_crop_image(pix, M2_SLOT2_BRIGHT, 1.0);
    M2HudColor bright1 = m2_sample_color(bright1Image);
    M2HudColor bright2 = m2_sample_color(bright2Image);
    if (bright1Image) CGImageRelease(bright1Image);
    if (bright2Image) CGImageRelease(bright2Image);

    NSInteger active = bright1.luma > bright2.luma ? 1 : 2;
    NSString *markerText = @"";
    if (m2_backpack_marker_detected(pix, &markerText)) {
        m2_attachment_scan_until = now + 0.3;
    }
    BOOL attachmentScan = now <= m2_attachment_scan_until;
    NSDictionary *slot = nil;
    NSDictionary *slot1 = nil;
    NSDictionary *slot2 = nil;
    if (attachmentScan) {
        NSString *backpackWeapon1 = m2_backpack_weapon(pix, 1, @"Keep");
        NSString *backpackWeapon2 = m2_backpack_weapon(pix, 2, @"Keep");
        slot1 = m2_detect_slot(pix, backpackWeapon1, 1);
        slot2 = m2_detect_slot(pix, backpackWeapon2, 2);
        slot = active == 1 ? slot1 : slot2;
    } else {
        slot = m2_empty_slot(@"Keep");
    }
    m2_send_hud_json(slot, slot1, slot2, active, bright1.luma, bright2.luma, attachmentScan);

    NSString *regionText = attachmentScan
        ? @"pos bag trig(186,558) L name(438,228) brl(441,304) scp(545,304) | R name(804,228) brl(807,304) scp(911,304)"
        : @"pos main trig(186,558) b1(1103,693) b2(1131,695)";
    NSString *resultText = nil;
    if (attachmentScan) {
        resultText = [NSString stringWithFormat:@"slot1 %@ %@/%@/%@  slot2 %@ %@/%@/%@",
                      slot1[@"weapon"], slot1[@"scope"], slot1[@"scopeColor"], slot1[@"barrelColor"],
                      slot2[@"weapon"], slot2[@"scope"], slot2[@"scopeColor"], slot2[@"barrelColor"]];
    } else {
        resultText = [NSString stringWithFormat:@"active=%ld L=%.0f/%.0f marker='%@' %@ %@/%@/%@",
                      (long)active, bright1.luma, bright2.luma,
                      markerText,
                      slot[@"weapon"], slot[@"scope"], slot[@"scopeColor"], slot[@"barrelColor"]];
    }
    NSString *text = [NSString stringWithFormat:@"HUD %@ %@\n%@\n%@",
                      attachmentScan ? @"MENU" : @"MAIN",
                      regionText, resultText, m2_ai_debug_text ?: @"AI waiting"];
    dispatch_async(dispatch_get_main_queue(), ^{
        if ([renderer respondsToSelector:@selector(m2UpdateHudOverlayWithText:activeSlot:attachmentScan:)]) {
            [renderer m2UpdateHudOverlayWithText:text activeSlot:active attachmentScan:attachmentScan];
        }
    });
}

static void m2_init_plugin(void) {
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        M2_LOG("[AI] 神经插件装载中...");
        m2_queue = dispatch_queue_create("com.m2.ai", DISPATCH_QUEUE_SERIAL);
        dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
            NSURL *url = [[NSBundle mainBundle] URLForResource:@"best" withExtension:@"mlmodelc"];
            if (!url) { M2_LOG("[AI] ❌ 模型文件丢失！"); return; }
            MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
            config.computeUnits = MLComputeUnitsAll;
            NSError *err = nil;
            MLModel *ml = [MLModel modelWithContentsOfURL:url configuration:config error:&err];
            if (ml) {
                m2_ai_model = [VNCoreMLModel modelForMLModel:ml error:nil];
                m2_ai_request = [[VNCoreMLRequest alloc] initWithModel:m2_ai_model];
                // 🛡️ 核心修复 2：使用 ScaleFit (等比缩放并留黑边)，完美保留游戏真实长宽比例！
                m2_ai_request.imageCropAndScaleOption = VNImageCropAndScaleOptionScaleFit;
                M2_LOG("[AI] ✅ 神经引擎点火成功！");
            }
        });
    });
}

static void m2_run_ai(CVImageBufferRef pix, id renderer) {
    if (!pix) return;
    if (!m2_ai_request) {
        m2_run_hud(pix, renderer);
        return;
    }

    // 如果 AI 还没处理完上一帧，直接丢弃新画面，保护系统不卡死！
    if (atomic_exchange(&ai_is_busy, true)) return;

    CFRetain(pix);
    dispatch_async(m2_queue, ^{
        @autoreleasepool {
            VNImageRequestHandler *h = [[VNImageRequestHandler alloc] initWithCVPixelBuffer:pix options:@{}];
            if ([h performRequests:@[m2_ai_request] error:nil]) {
                int w = (int)CVPixelBufferGetWidth(pix), h_px = (int)CVPixelBufferGetHeight(pix);
                int best_x = -1, best_y = -1; float min_d = 1e10;
                float best_conf = 0.0f;
                NSString *best_label = @"unknown";

                for (VNRecognizedObjectObservation *o in m2_ai_request.results) {
                    // 读取顶部宏定义的阈值
                    if (o.confidence > AI_CONFIDENCE_THRESHOLD) {
                        CGRect b = o.boundingBox;
                        // Vision 框架会自动帮我们把坐标还原回原图 (1080P/4K) 的比例
                        int tx = (b.origin.x + b.size.width/2.0)*w;
                        int ty = (1.0-b.origin.y-b.size.height*(1.0-AI_AIM_OFFSET))*h_px;

                        float d = pow(tx-w/2.0, 2) + pow(ty-h_px/2.0, 2);
                        if (d < min_d) {
                            min_d = d;
                            best_x = tx;
                            best_y = ty;
                            best_conf = o.confidence;
                            VNClassificationObservation *label = o.labels.firstObject;
                            best_label = label.identifier ?: @"unknown";
                        }
                    }
                }
                if (best_x != -1) {
                    float dx = (float)(best_x - w / 2);
                    float dy = (float)(best_y - h_px / 2);
                    m2_ai_debug_text = [NSString stringWithFormat:@"AI %@ %.2f pos=%d,%d dx=%.0f dy=%.0f", best_label, best_conf, best_x, best_y, dx, dy];
                    char m[64]; snprintf(m, 64, "{\"f\":1,\"dx\":%.1f,\"dy\":%.1f}", dx, dy);
                    sendto(m2_udp_sock, m, (int)strlen(m), 0, (struct sockaddr *)&m2_pc_addr, sizeof(m2_pc_addr));
                } else {
                    m2_ai_debug_text = @"AI none";
                    sendto(m2_udp_sock, "{\"f\":0}", 7, 0, (struct sockaddr *)&m2_pc_addr, sizeof(m2_pc_addr));
                }
            }
            m2_run_hud(pix, renderer);
            CFRelease(pix);
        }
        // AI 任务完成，解锁，允许接收下一帧
        atomic_store(&ai_is_busy, false);
    });
}

static void m2_decomp_callback(void *refCon, void *sfRefCon, OSStatus status, VTDecodeInfoFlags info, CVImageBufferRef img, CMTime pts, CMTime dur) {
    if (status == noErr && img) {
        VideoDecoderRenderer *renderer = (__bridge VideoDecoderRenderer *)refCon;
        m2_run_ai(img, renderer);
    }
}

// ==========================================================
// 🏗️ Moonlight 安全渲染核心 (官方源码保持不变)
// ==========================================================
extern int ff_isom_write_av1c(AVIOContext *pb, const uint8_t *buf, int size, int write_seq_header);

@implementation VideoDecoderRenderer {
    StreamView* _view;
    id<ConnectionCallbacks> _callbacks;
    float _streamAspectRatio;
    AVSampleBufferDisplayLayer* displayLayer;
    int videoFormat, frameRate;
    NSMutableArray *parameterSetBuffers;
    CMVideoFormatDescriptionRef formatDesc;
    CADisplayLink* _displayLink;
    BOOL framePacing;
    VTDecompressionSessionRef _session;
    CALayer *_m2HudOverlayLayer;
    CATextLayer *_m2HudTextLayer;
    NSMutableArray<CALayer *> *_m2HudRegionLayers;
}

- (void)reinitializeDisplayLayer {
    if (displayLayer) [displayLayer removeFromSuperlayer];
    displayLayer = [[AVSampleBufferDisplayLayer alloc] init];
    displayLayer.backgroundColor = [UIColor blackColor].CGColor;
    displayLayer.videoGravity = AVLayerVideoGravityResize;

    float vW = _view.bounds.size.width, vH = _view.bounds.size.height;
    if (vW > vH * _streamAspectRatio) vW = vH * _streamAspectRatio; else vH = vW / _streamAspectRatio;
    displayLayer.position = CGPointMake(CGRectGetMidX(_view.bounds), CGRectGetMidY(_view.bounds));
    displayLayer.bounds = CGRectMake(0, 0, vW, vH);
    [_view.layer addSublayer:displayLayer];
    [self m2UpdateHudOverlayWithText:@"HUD waiting" activeSlot:1 attachmentScan:NO];

    if (_session) { VTDecompressionSessionInvalidate(_session); CFRelease(_session); _session = NULL; }
    if (formatDesc) { CFRelease(formatDesc); formatDesc = nil; }
}

- (CGRect)m2RectForHudRect:(M2HudRect)rect {
    if (!_m2HudOverlayLayer) return CGRectZero;
    CGFloat sx = _m2HudOverlayLayer.bounds.size.width / M2_HUD_BASE_W;
    CGFloat sy = _m2HudOverlayLayer.bounds.size.height / M2_HUD_BASE_H;
    return CGRectMake(rect.x * sx, rect.y * sy, MAX(1.0, rect.w * sx), MAX(1.0, rect.h * sy));
}

- (void)m2EnsureHudOverlay {
    if (!_view || !displayLayer) return;
    if (!_m2HudOverlayLayer) {
        _m2HudOverlayLayer = [CALayer layer];
        _m2HudOverlayLayer.masksToBounds = YES;
        _m2HudRegionLayers = [NSMutableArray array];
        for (int i = 0; i < 8; i++) {
            CALayer *layer = [CALayer layer];
            layer.borderWidth = 1.5;
            layer.backgroundColor = [UIColor clearColor].CGColor;
            [_m2HudOverlayLayer addSublayer:layer];
            [_m2HudRegionLayers addObject:layer];
        }
        _m2HudTextLayer = [CATextLayer layer];
        _m2HudTextLayer.contentsScale = [UIScreen mainScreen].scale;
        _m2HudTextLayer.fontSize = 12.0;
        _m2HudTextLayer.alignmentMode = kCAAlignmentLeft;
        _m2HudTextLayer.wrapped = YES;
        _m2HudTextLayer.foregroundColor = [UIColor colorWithRed:1.0 green:0.95 blue:0.35 alpha:1.0].CGColor;
        _m2HudTextLayer.backgroundColor = [UIColor colorWithWhite:0.0 alpha:0.55].CGColor;
        [_m2HudOverlayLayer addSublayer:_m2HudTextLayer];
    }
    if (_m2HudOverlayLayer.superlayer != _view.layer) {
        [_view.layer addSublayer:_m2HudOverlayLayer];
    }
    _m2HudOverlayLayer.position = displayLayer.position;
    _m2HudOverlayLayer.bounds = displayLayer.bounds;
    _m2HudOverlayLayer.zPosition = 1000;
}

- (void)m2SetHudRegion:(NSUInteger)index rect:(M2HudRect)rect color:(UIColor *)color {
    if (index >= _m2HudRegionLayers.count) return;
    CALayer *layer = _m2HudRegionLayers[index];
    layer.frame = [self m2RectForHudRect:rect];
    layer.borderColor = color.CGColor;
    layer.hidden = NO;
}

- (void)m2UpdateHudOverlayWithText:(NSString *)text activeSlot:(NSInteger)activeSlot attachmentScan:(BOOL)attachmentScan {
    [self m2EnsureHudOverlay];
    if (!_m2HudOverlayLayer) return;

    UIColor *triggerColor = [UIColor colorWithRed:0.95 green:0.95 blue:0.15 alpha:1.0];
    UIColor *brightColor = [UIColor colorWithRed:1.0 green:0.85 blue:0.15 alpha:1.0];
    UIColor *activeAttachColor = [UIColor colorWithRed:0.75 green:0.45 blue:1.0 alpha:1.0];
    UIColor *inactiveAttachColor = [UIColor colorWithRed:0.45 green:0.45 blue:0.45 alpha:0.85];

    [self m2SetHudRegion:0 rect:M2_BACKPACK_MARKER color:triggerColor];
    if (attachmentScan) {
        UIColor *leftAttachColor = activeSlot == 1 ? activeAttachColor : inactiveAttachColor;
        UIColor *rightAttachColor = activeSlot == 2 ? activeAttachColor : inactiveAttachColor;
        [self m2SetHudRegion:1 rect:M2_BACKPACK_LEFT_NAME color:leftAttachColor];
        [self m2SetHudRegion:2 rect:M2_BACKPACK_RIGHT_NAME color:rightAttachColor];
        [self m2SetHudRegion:3 rect:M2_BACKPACK_LEFT_BARREL color:leftAttachColor];
        [self m2SetHudRegion:4 rect:M2_BACKPACK_LEFT_SCOPE color:leftAttachColor];
        [self m2SetHudRegion:5 rect:M2_BACKPACK_RIGHT_BARREL color:rightAttachColor];
        [self m2SetHudRegion:6 rect:M2_BACKPACK_RIGHT_SCOPE color:rightAttachColor];
        _m2HudRegionLayers[7].hidden = YES;
    } else {
        [self m2SetHudRegion:1 rect:M2_SLOT1_BRIGHT color:brightColor];
        [self m2SetHudRegion:2 rect:M2_SLOT2_BRIGHT color:brightColor];
        for (NSUInteger i = 3; i < _m2HudRegionLayers.count; i++) {
            _m2HudRegionLayers[i].hidden = YES;
        }
    }

    CGFloat width = _m2HudOverlayLayer.bounds.size.width;
    _m2HudTextLayer.frame = CGRectMake(8, 8, MAX(200, width - 16), 64);
    _m2HudTextLayer.string = text ?: @"HUD waiting";
    _m2HudOverlayLayer.hidden = NO;
}

- (id)initWithView:(StreamView*)view callbacks:(id<ConnectionCallbacks>)callbacks streamAspectRatio:(float)aspectRatio useFramePacing:(BOOL)useFramePacing {
    self = [super init];
    _view = view; _callbacks = callbacks; _streamAspectRatio = aspectRatio; framePacing = useFramePacing;
    parameterSetBuffers = [NSMutableArray new];
    m2_init_plugin();
    [self reinitializeDisplayLayer];
    return self;
}

- (void)setupWithVideoFormat:(int)vf width:(int)vw height:(int)vh frameRate:(int)fr {
    self->videoFormat = vf; self->frameRate = fr;
}

- (void)start {
    _displayLink = [CADisplayLink displayLinkWithTarget:self selector:@selector(displayLinkCallback:)];
    if (@available(iOS 15.0, tvOS 15.0, *)) { _displayLink.preferredFrameRateRange = CAFrameRateRangeMake(self->frameRate, self->frameRate, self->frameRate); }
    else { _displayLink.preferredFramesPerSecond = self->frameRate; }
    [_displayLink addToRunLoop:[NSRunLoop mainRunLoop] forMode:NSDefaultRunLoopMode];
}

int DrSubmitDecodeUnit(PDECODE_UNIT du);
- (void)displayLinkCallback:(CADisplayLink *)sender {
    VIDEO_FRAME_HANDLE h; PDECODE_UNIT du;
    while (LiPollNextVideoFrame(&h, &du)) { LiCompleteVideoFrame(h, DrSubmitDecodeUnit(du)); }
}

- (void)stop {
    [_displayLink invalidate];
    _m2HudOverlayLayer.hidden = YES;
    if (_session) { VTDecompressionSessionInvalidate(_session); CFRelease(_session); _session = NULL; }
}

- (void)updateAnnexBBufferForRange:(CMBlockBufferRef)frameBuffer dataBlock:(CMBlockBufferRef)dataBlock offset:(int)offset length:(int)nalLength data:(unsigned char *)data {
    if (nalLength < 4) return;
    int startLen = 3;
    if (nalLength >= 4 && data[offset] == 0 && data[offset+1] == 0 && data[offset+2] == 0 && data[offset+3] == 1) {
        startLen = 4;
    } else if (nalLength >= 3 && data[offset] == 0 && data[offset+1] == 0 && data[offset+2] == 1) {
        startLen = 3;
    } else return;

    OSStatus status;
    size_t oldOffset = CMBlockBufferGetDataLength(frameBuffer);
    status = CMBlockBufferAppendMemoryBlock(frameBuffer, NULL, 4, kCFAllocatorDefault, NULL, 0, 4, 0);
    if (status != noErr) return;

    const int dataLength = nalLength - startLen;
    const uint8_t lengthBytes[] = {(uint8_t)(dataLength >> 24), (uint8_t)(dataLength >> 16), (uint8_t)(dataLength >> 8), (uint8_t)dataLength};
    status = CMBlockBufferReplaceDataBytes(lengthBytes, frameBuffer, oldOffset, 4);
    if (status != noErr) return;

    CMBlockBufferAppendBufferReference(frameBuffer, dataBlock, offset + startLen, dataLength, 0);
}

- (int)submitDecodeBuffer:(unsigned char *)data length:(int)length bufferType:(int)bufferType decodeUnit:(PDECODE_UNIT)du {
    if (du->frameType == FRAME_TYPE_IDR) {
        if (bufferType != BUFFER_TYPE_PICDATA) {
            if (bufferType == BUFFER_TYPE_VPS || bufferType == BUFFER_TYPE_SPS || bufferType == BUFFER_TYPE_PPS) {
                int startLen = 3;
                if (length >= 4 && data[0] == 0 && data[1] == 0 && data[2] == 0 && data[3] == 1) startLen = 4;
                [parameterSetBuffers addObject:[NSData dataWithBytes:&data[startLen] length:length - startLen]];
            }
            return DR_OK;
        }

        if (formatDesc != NULL) { CFRelease(formatDesc); formatDesc = NULL; }
        if (_session != NULL) { VTDecompressionSessionInvalidate(_session); CFRelease(_session); _session = NULL; }

        size_t pc = [parameterSetBuffers count];
        if (pc == 0) { free(data); return DR_NEED_IDR; }

        const uint8_t* pps[pc]; size_t pss[pc];
        for (int i = 0; i < pc; i++) { NSData* p = parameterSetBuffers[i]; pps[i] = p.bytes; pss[i] = p.length; }

        if (videoFormat & VIDEO_FORMAT_MASK_H264) {
            CMVideoFormatDescriptionCreateFromH264ParameterSets(NULL, pc, pps, pss, 4, &formatDesc);
        } else if (videoFormat & VIDEO_FORMAT_MASK_H265) {
            CMVideoFormatDescriptionCreateFromHEVCParameterSets(NULL, pc, pps, pss, 4, NULL, &formatDesc);
        }
        [parameterSetBuffers removeAllObjects];

        if (formatDesc) {
            VTDecompressionOutputCallbackRecord cb = {m2_decomp_callback, (__bridge void *)self};
            NSDictionary *attr = @{(id)kCVPixelBufferPixelFormatTypeKey: @(kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange)};
            VTDecompressionSessionCreate(NULL, formatDesc, NULL, (__bridge CFDictionaryRef)attr, &cb, &_session);
        }
    }

    if (!formatDesc) { free(data); return DR_NEED_IDR; }

    CMBlockBufferRef fbb, dbb;
    CMBlockBufferCreateWithMemoryBlock(NULL, data, length, kCFAllocatorDefault, NULL, 0, length, 0, &dbb);
    CMBlockBufferCreateEmpty(NULL, 0, 0, &fbb);

    int last = -1;
    for (int i = 0; i < length - 2; i++) {
        if (data[i] == 0 && data[i+1] == 0 && data[i+2] == 1) {
            int start_idx = i;
            if (i > 0 && data[i-1] == 0) start_idx = i - 1;
            if (last != -1) [self updateAnnexBBufferForRange:fbb dataBlock:dbb offset:last length:start_idx - last data:data];
            last = start_idx;
            i += 2;
        }
    }
    if (last != -1) {
        [self updateAnnexBBufferForRange:fbb dataBlock:dbb offset:last length:length - last data:data];
    } else {
        CMBlockBufferAppendBufferReference(fbb, dbb, 0, length, 0);
    }

    CMSampleBufferRef sb;
    CMSampleTimingInfo ti = {kCMTimeInvalid, CMTimeMake(du->presentationTimeMs, 1000), kCMTimeInvalid};
    OSStatus sbStatus = CMSampleBufferCreateReady(NULL, fbb, formatDesc, 1, 1, &ti, 0, NULL, &sb);

    if (sbStatus == noErr) {
        [displayLayer enqueueSampleBuffer:sb];
        if (_session) { VTDecompressionSessionDecodeFrame(_session, sb, kVTDecodeFrame_EnableAsynchronousDecompression, NULL, NULL); }
    } else {
        CFRelease(dbb); CFRelease(fbb); free(data); return DR_NEED_IDR;
    }

    if (du->frameType == FRAME_TYPE_IDR) {
        displayLayer.hidden = NO;
        [_callbacks videoContentShown];
    }

    CFRelease(dbb); CFRelease(fbb); if (sb) CFRelease(sb);
    return DR_OK;
}

- (void)setHdrMode:(BOOL)enabled {}
@end
