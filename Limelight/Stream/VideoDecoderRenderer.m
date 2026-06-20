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
#include <netdb.h>
#include <fcntl.h>
#include <math.h>
#include <stdatomic.h> // 引入原子锁机制

// ==========================================================
// ⚙️ 炼丹师专属调参区
// ==========================================================
// 降低阈值以提高灵敏度（原 0.45 -> 现 0.28）
#define AI_LOCKED_CONFIDENCE_THRESHOLD 0.22f
#define AI_NEW_CONFIDENCE_THRESHOLD 0.28f
#define AI_IMMEDIATE_CONFIDENCE_THRESHOLD 0.45f
// 瞄准点下压比例（0.20 = 框的中心往下 20%，瞄准胸口）
#define AI_AIM_OFFSET 0.20f
// 调试范围框保留显示，但框线外扩绘制，不压住实际取样/OCR 像素。
#define M2_SHOW_HUD_REGION_BOXES 1
#define M2_HUD_REGION_BOX_PADDING 4.0

// ==========================================================
// 📡 [M2 ANE] 异步纯旁路 AI 引擎 (满血防卡死版)
// ==========================================================
static int m2_udp_sock = -1;
static struct sockaddr_in m2_pc_addr;
static struct sockaddr_in m2_hud_addr;
static struct sockaddr_in m2_host_ai_addr;
static struct sockaddr_in m2_host_hud_addr;
static atomic_bool m2_has_host_addr = false;

static void init_logger_once(void) {
    if (m2_udp_sock != -1) return;
    m2_udp_sock = socket(AF_INET, SOCK_DGRAM, 0);
    int broadcast = 1;
    setsockopt(m2_udp_sock, SOL_SOCKET, SO_BROADCAST, &broadcast, sizeof(broadcast));
    fcntl(m2_udp_sock, F_SETFL, O_NONBLOCK);
    m2_pc_addr.sin_family = AF_INET;
    m2_pc_addr.sin_port = htons(9999);
    // Broadcast avoids hard-coding the PC IP. The Windows receiver listens on
    // 9999 and forwards AI packets to DS4W's internal 10000 port.
    inet_pton(AF_INET, "255.255.255.255", &m2_pc_addr.sin_addr);
    m2_hud_addr = m2_pc_addr;
    m2_hud_addr.sin_port = htons(9998);
}

static BOOL M2ResolveApexHost(NSString *host, struct in_addr *outAddr) {
    if (!host || host.length == 0 || !outAddr) {
        return NO;
    }

    NSString *name = [host stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
    if (name.length == 0) {
        return NO;
    }

    NSRange colon = [name rangeOfString:@":"];
    if (colon.location != NSNotFound) {
        name = [name substringToIndex:colon.location];
    }

    if (inet_pton(AF_INET, [name UTF8String], outAddr) == 1) {
        return YES;
    }

    struct addrinfo hints;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_DGRAM;

    struct addrinfo *result = NULL;
    int rc = getaddrinfo([name UTF8String], NULL, &hints, &result);
    if (rc != 0 || !result) {
        return NO;
    }

    BOOL resolved = NO;
    for (struct addrinfo *item = result; item; item = item->ai_next) {
        if (item->ai_family == AF_INET && item->ai_addrlen >= sizeof(struct sockaddr_in)) {
            struct sockaddr_in *addr = (struct sockaddr_in *)item->ai_addr;
            *outAddr = addr->sin_addr;
            resolved = YES;
            break;
        }
    }
    freeaddrinfo(result);
    return resolved;
}

static void M2SetApexHost(NSString *host) {
    if (!host || host.length == 0) {
        atomic_store(&m2_has_host_addr, false);
        return;
    }

    struct sockaddr_in aiAddr;
    memset(&aiAddr, 0, sizeof(aiAddr));
    aiAddr.sin_family = AF_INET;
    aiAddr.sin_port = htons(9999);
    if (M2ResolveApexHost(host, &aiAddr.sin_addr)) {
        m2_host_ai_addr = aiAddr;
        m2_host_hud_addr = aiAddr;
        m2_host_hud_addr.sin_port = htons(9998);
        atomic_store(&m2_has_host_addr, true);
    } else {
        atomic_store(&m2_has_host_addr, false);
    }
}

static void m2_send_ai_payload(const char *payload, int len) {
    init_logger_once();
    if (atomic_load(&m2_has_host_addr)) {
        sendto(m2_udp_sock, payload, len, 0, (struct sockaddr *)&m2_host_ai_addr, sizeof(m2_host_ai_addr));
    }
    sendto(m2_udp_sock, payload, len, 0, (struct sockaddr *)&m2_pc_addr, sizeof(m2_pc_addr));
}

static void m2_send_hud_payload(const char *payload, int len) {
    init_logger_once();
    if (atomic_load(&m2_has_host_addr)) {
        sendto(m2_udp_sock, payload, len, 0, (struct sockaddr *)&m2_host_hud_addr, sizeof(m2_host_hud_addr));
    }
    sendto(m2_udp_sock, payload, len, 0, (struct sockaddr *)&m2_hud_addr, sizeof(m2_hud_addr));
}

static void M2_LOG(const char *format, ...) {
    init_logger_once();
    char buffer[512];
    va_list args;
    va_start(args, format);
    vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);
    m2_send_ai_payload(buffer, (int)strlen(buffer));
}

static VNCoreMLModel *m2_ai_model = nil;
static VNCoreMLRequest *m2_ai_request = nil;
static dispatch_queue_t m2_queue = nil;
static dispatch_queue_t m2_hud_queue = nil;
static CIContext *m2_ci_context = nil;
static CFAbsoluteTime m2_last_hud_time = 0.0;
static CFAbsoluteTime m2_attachment_scan_until = 0.0;
static CFAbsoluteTime m2_last_ai_sent_time = 0.0;
static const CGFloat M2_AI_CROP_BASE_SIZE = 640.0;
static const int M2_AI_FULL_FRAME_REACQUIRE_MISSES = 4;
static const int M2_AI_FULL_FRAME_REACQUIRE_PERIOD = 6;
static const int M2_AI_LOCK_HOLD_MISSES = 3;
static const CFTimeInterval M2_HUD_MAIN_INTERVAL = 0.10;
static const CFTimeInterval M2_HUD_ATTACHMENT_INTERVAL = 1.0 / 30.0;
static NSString *m2_ai_debug_text = @"AI waiting";
static int m2_ai_miss_streak = 0;
static BOOL m2_ai_filter_initialized = NO;
static CGFloat m2_ai_filtered_x = 0.0;
static CGFloat m2_ai_filtered_y = 0.0;
static CGFloat m2_ai_prev_raw_x = 0.0;
static CGFloat m2_ai_prev_raw_y = 0.0;
static CGFloat m2_ai_filtered_vx = 0.0;
static CGFloat m2_ai_filtered_vy = 0.0;
static CFAbsoluteTime m2_ai_filter_last_time = 0.0;
static BOOL m2_ai_track_center_valid = NO;
static CGFloat m2_ai_track_center_x = 0.0;
static CGFloat m2_ai_track_center_y = 0.0;
static BOOL m2_ai_lock_valid = NO;
static CGFloat m2_ai_lock_x = 0.0;
static CGFloat m2_ai_lock_y = 0.0;
static CGFloat m2_ai_lock_bw = 0.0;
static CGFloat m2_ai_lock_bh = 0.0;
static int m2_ai_lock_misses = 0;
static BOOL m2_ai_pending_valid = NO;
static CGFloat m2_ai_pending_x = 0.0;
static CGFloat m2_ai_pending_y = 0.0;
static int m2_ai_pending_hits = 0;

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
static const M2HudRect M2_BACKPACK_RIGHT_SCOPE = {911, 304, 42, 3};
static const M2HudRect M2_BACKPACK_LEFT_HAVOC_SCOPE = {494, 304, 39, 2};
static const M2HudRect M2_BACKPACK_RIGHT_HAVOC_SCOPE = {860, 303, 39, 3};
static const M2HudRect M2_BACKPACK_LEFT_HAVOC_PAINTBALL = {601, 308, 36, 3};
static const M2HudRect M2_BACKPACK_RIGHT_HAVOC_PAINTBALL = {967, 308, 36, 3};
static const M2HudRect M2_BACKPACK_LEFT_DEVOTION_PAINTBALL = {662, 308, 36, 3};
static const M2HudRect M2_BACKPACK_RIGHT_DEVOTION_PAINTBALL = {1028, 308, 36, 3};

@interface VideoDecoderRenderer ()
- (void)m2UpdateHudOverlayWithText:(NSString *)text activeSlot:(NSInteger)activeSlot attachmentScan:(BOOL)attachmentScan;
@end

void M2NotifyApexMenuButton(BOOL pressed) {
    (void)pressed;
}

// 🛡️ 核心修复 1：原子锁，防止 4K 120FPS 撑爆显存
static atomic_bool ai_is_busy = false;
static atomic_bool hud_is_busy = false;

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

static CGRect m2_ai_crop_rect_for_buffer(CVImageBufferRef pix) {
    CGFloat pw = (CGFloat)CVPixelBufferGetWidth(pix);
    CGFloat ph = (CGFloat)CVPixelBufferGetHeight(pix);
    if (pw <= 0.0 || ph <= 0.0) return CGRectZero;

    CGFloat shortSide = MIN(pw, ph);
    CGFloat cropSize = M2_AI_CROP_BASE_SIZE * (shortSide / M2_HUD_BASE_H);
    cropSize = MIN(shortSide, MAX(1.0, cropSize));
    CGFloat x = floor((pw - cropSize) * 0.5);
    CGFloat y = floor((ph - cropSize) * 0.5);
    return CGRectMake(x, y, cropSize, cropSize);
}

static CGRect m2_ai_full_rect_for_buffer(CVImageBufferRef pix) {
    CGFloat pw = (CGFloat)CVPixelBufferGetWidth(pix);
    CGFloat ph = (CGFloat)CVPixelBufferGetHeight(pix);
    if (pw <= 0.0 || ph <= 0.0) return CGRectZero;
    return CGRectMake(0.0, 0.0, pw, ph);
}

static CGRect m2_ai_tracking_crop_rect_for_buffer(CVImageBufferRef pix, BOOL *usingTrack) {
    CGRect centerCrop = m2_ai_crop_rect_for_buffer(pix);
    if (usingTrack) *usingTrack = NO;
    if (CGRectIsEmpty(centerCrop) || !m2_ai_track_center_valid) return centerCrop;

    CGFloat pw = (CGFloat)CVPixelBufferGetWidth(pix);
    CGFloat ph = (CGFloat)CVPixelBufferGetHeight(pix);
    CGFloat cropSize = centerCrop.size.width;
    CGFloat left = MIN(MAX(0.0, m2_ai_track_center_x - cropSize * 0.5), MAX(0.0, pw - cropSize));
    CGFloat top = MIN(MAX(0.0, m2_ai_track_center_y - cropSize * 0.5), MAX(0.0, ph - cropSize));
    CGFloat bottom = ph - (top + cropSize);
    if (usingTrack) *usingTrack = YES;
    return CGRectMake(floor(left), floor(bottom), cropSize, cropSize);
}

static BOOL m2_ai_should_use_full_frame_reacquire(void) {
    if (m2_ai_miss_streak < M2_AI_FULL_FRAME_REACQUIRE_MISSES) return NO;
    int missesSinceFirstReacquire = m2_ai_miss_streak - M2_AI_FULL_FRAME_REACQUIRE_MISSES;
    return (missesSinceFirstReacquire % M2_AI_FULL_FRAME_REACQUIRE_PERIOD) == 0;
}

static CGFloat m2_ai_scaled_gate(CGFloat pixels, int w, int h) {
    CGFloat shortSide = MIN((CGFloat)w, (CGFloat)h);
    return pixels * (shortSide / M2_HUD_BASE_H);
}

static void m2_ai_clear_pending_target(void) {
    m2_ai_pending_valid = NO;
    m2_ai_pending_hits = 0;
}

static BOOL m2_ai_confirm_pending_target(CGFloat x, CGFloat y, float confidence, int w, int h) {
    CGFloat gate = MAX(70.0, m2_ai_scaled_gate(120.0, w, h));
    if (m2_ai_pending_valid && hypot(x - m2_ai_pending_x, y - m2_ai_pending_y) <= gate) {
        m2_ai_pending_hits = MIN(m2_ai_pending_hits + 1, 100);
    }
    else {
        m2_ai_pending_valid = YES;
        m2_ai_pending_hits = 1;
    }

    m2_ai_pending_x = x;
    m2_ai_pending_y = y;
    return confidence >= AI_IMMEDIATE_CONFIDENCE_THRESHOLD || m2_ai_pending_hits >= 2;
}

static CGFloat m2_ai_alpha_for_cutoff(CGFloat cutoff, CGFloat dt) {
    if (dt <= 0.0) return 1.0;
    CGFloat tau = 1.0 / (2.0 * 3.14159265358979323846 * MAX(0.001, cutoff));
    return 1.0 / (1.0 + tau / dt);
}

static CGPoint m2_ai_smooth_point(CGPoint raw, CFAbsoluteTime now, BOOL reset) {
    if (!m2_ai_filter_initialized || reset) {
        m2_ai_filter_initialized = YES;
        m2_ai_filtered_x = raw.x;
        m2_ai_filtered_y = raw.y;
        m2_ai_prev_raw_x = raw.x;
        m2_ai_prev_raw_y = raw.y;
        m2_ai_filtered_vx = 0.0;
        m2_ai_filtered_vy = 0.0;
        m2_ai_filter_last_time = now;
        return raw;
    }

    CGFloat dt = (CGFloat)(now - m2_ai_filter_last_time);
    if (dt < 1.0 / 240.0 || dt > 1.0 / 12.0) {
        dt = 1.0 / 60.0;
    }

    CGFloat vx = (raw.x - m2_ai_prev_raw_x) / dt;
    CGFloat vy = (raw.y - m2_ai_prev_raw_y) / dt;
    CGFloat dAlpha = m2_ai_alpha_for_cutoff(1.0, dt);
    m2_ai_filtered_vx += (vx - m2_ai_filtered_vx) * dAlpha;
    m2_ai_filtered_vy += (vy - m2_ai_filtered_vy) * dAlpha;

    CGFloat speed = hypot(m2_ai_filtered_vx, m2_ai_filtered_vy);
    CGFloat cutoff = 5.0 + 0.018 * speed;
    CGFloat alpha = m2_ai_alpha_for_cutoff(cutoff, dt);

    m2_ai_filtered_x += (raw.x - m2_ai_filtered_x) * alpha;
    m2_ai_filtered_y += (raw.y - m2_ai_filtered_y) * alpha;
    m2_ai_prev_raw_x = raw.x;
    m2_ai_prev_raw_y = raw.y;
    m2_ai_filter_last_time = now;

    return CGPointMake(m2_ai_filtered_x, m2_ai_filtered_y);
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
    CGFloat dGold = MIN(m2_color_distance(c, 255, 243, 101), m2_color_distance(c, 249, 201, 84));
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
    if (dGold <= 42.0 && c.r >= 220.0 && c.g >= 180.0 && c.b <= 140.0) return @"Purple";
    CGFloat best = MIN(dWhite, MIN(dBlue, dPurple));
    if (best > 46.0) return @"None";
    if (dPurple <= dBlue && dPurple <= dWhite) return @"Purple";
    if (dBlue <= dWhite) return @"Blue";
    return @"White";
}

static M2HudRect m2_scope_variant_rect(M2HudRect scopeRect) {
    return (M2HudRect){scopeRect.x + 17, scopeRect.y - 27, 9, 2};
}

static NSString *m2_scope_for_color(CVImageBufferRef pix, NSString **colorRef, M2HudRect scopeRect) {
    NSString *color = *colorRef ?: @"None";
    if ([color isEqualToString:@"White"]) return @"S1x";
    if (![color isEqualToString:@"Blue"] && ![color isEqualToString:@"Purple"]) return @"S1x";

    CGImageRef variantImage = m2_create_crop_image(pix, m2_scope_variant_rect(scopeRect), 1.0);
    M2HudColor variantColor = m2_sample_color(variantImage);
    if (variantImage) CGImageRelease(variantImage);

    BOOL brightVariant = variantColor.luma > 70.0;
    if ([color isEqualToString:@"Purple"]) {
        if (brightVariant) {
            *colorRef = @"PurpleVariable";
            return @"S2x";
        }
        *colorRef = @"Purple3x";
        return @"S3x";
    }

    if (brightVariant) {
        *colorRef = @"BlueVariable";
        return @"S1x";
    }
    *colorRef = @"Blue2x";
    return @"S2x";
}

static NSString *m2_classify_havoc_paintball(M2HudColor c) {
    CGFloat dActive = MIN(m2_color_distance(c, 157, 137, 104), m2_color_distance(c, 250, 201, 59));
    CGFloat dInactive = MIN(m2_color_distance(c, 141, 141, 132), m2_color_distance(c, 191, 212, 213));
    if (dActive < dInactive && (c.r - c.b >= 25.0 || c.r - c.g >= 25.0)) return @"Paintball";
    return @"None";
}

static NSString *m2_classify_devotion_paintball(M2HudColor c) {
    CGFloat dActive = MIN(m2_color_distance(c, 157, 137, 104), m2_color_distance(c, 250, 201, 59));
    CGFloat dInactive = MIN(m2_color_distance(c, 141, 141, 132), m2_color_distance(c, 191, 212, 213));
    if (dActive <= dInactive + 25.0 && c.luma >= 45.0 && (c.r - c.b >= 15.0 || c.r - c.g >= 15.0 || c.g - c.b >= 10.0)) return @"Active";
    return @"Inactive";
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
    if ([s containsString:@"lstar"] || [s containsString:@"l-star"] || [s containsString:@"st"] || [s containsString:@"star"]) return @"LStar";
    if ([s containsString:@"devotion"] || [s containsString:@"专注"]) return @"Devotion";
    return @"None";
}

static BOOL m2_is_locked_1x_weapon(NSString *weapon) {
    return [@[@"R99", @"CAR", @"Alternator", @"Prowler", @"RE45"] containsObject:weapon];
}

static BOOL m2_uses_rifle_barrel(NSString *weapon) {
    return [@[@"Nemesis", @"R301", @"Devotion"] containsObject:weapon];
}

static BOOL m2_uses_rifle_scope(NSString *weapon) {
    return [@[@"Nemesis", @"R301", @"Flatline", @"Devotion"] containsObject:weapon];
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

    NSString *s = [[[raw lowercaseString] stringByReplacingOccurrencesOfString:@" " withString:@""]
                   stringByReplacingOccurrencesOfString:@"９" withString:@"9"];
    return [s containsString:@"平"] || [s containsString:@"ha"] || [s containsString:@"99"];
}

static M2HudRect m2_backpack_scope_rect(NSInteger active) {
    return active == 2 ? M2_BACKPACK_RIGHT_SCOPE : M2_BACKPACK_LEFT_SCOPE;
}

static M2HudRect m2_backpack_barrel_rect(NSInteger active) {
    return active == 2 ? M2_BACKPACK_RIGHT_BARREL : M2_BACKPACK_LEFT_BARREL;
}

static M2HudRect m2_backpack_havoc_scope_rect(NSInteger active) {
    return active == 2 ? M2_BACKPACK_RIGHT_HAVOC_SCOPE : M2_BACKPACK_LEFT_HAVOC_SCOPE;
}

static M2HudRect m2_backpack_havoc_paintball_rect(NSInteger active) {
    return active == 2 ? M2_BACKPACK_RIGHT_HAVOC_PAINTBALL : M2_BACKPACK_LEFT_HAVOC_PAINTBALL;
}

static M2HudRect m2_backpack_devotion_paintball_rect(NSInteger active) {
    return active == 2 ? M2_BACKPACK_RIGHT_DEVOTION_PAINTBALL : M2_BACKPACK_LEFT_DEVOTION_PAINTBALL;
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
    } else if ([weapon isEqualToString:@"LStar"]) {
        M2HudRect scopeRect = m2_backpack_barrel_rect(slot);
        CGImageRef scopeImage = m2_create_crop_image(pix, scopeRect, 1.0);
        scopeColor = m2_classify_equipment_color(m2_sample_color(scopeImage));
        if (scopeImage) CGImageRelease(scopeImage);
        scope = m2_scope_for_color(pix, &scopeColor, scopeRect);
    } else if ([weapon isEqualToString:@"Havoc"]) {
        M2HudRect scopeRect = m2_backpack_havoc_scope_rect(slot);
        CGImageRef scopeImage = m2_create_crop_image(pix, scopeRect, 1.0);
        scopeColor = m2_classify_equipment_color(m2_sample_color(scopeImage));
        if (scopeImage) CGImageRelease(scopeImage);
        scope = m2_scope_for_color(pix, &scopeColor, scopeRect);

        CGImageRef paintImage = m2_create_crop_image(pix, m2_backpack_havoc_paintball_rect(slot), 1.0);
        barrel = m2_classify_havoc_paintball(m2_sample_color(paintImage));
        if (paintImage) CGImageRelease(paintImage);
        barrelColor = [barrel isEqualToString:@"Paintball"] ? @"Active" : @"None";
    } else if (m2_uses_rifle_scope(weapon)) {
        M2HudRect scopeRect = m2_backpack_scope_rect(slot);
        CGImageRef scopeImage = m2_create_crop_image(pix, scopeRect, 1.0);
        scopeColor = m2_classify_equipment_color(m2_sample_color(scopeImage));
        if (scopeImage) CGImageRelease(scopeImage);
        scope = m2_scope_for_color(pix, &scopeColor, scopeRect);

        if (m2_uses_rifle_barrel(weapon)) {
            CGImageRef barrelImage = m2_create_crop_image(pix, m2_backpack_barrel_rect(slot), 1.0);
            barrelColor = m2_classify_equipment_color(m2_sample_color(barrelImage));
            if (barrelImage) CGImageRelease(barrelImage);
            barrel = barrelColor;
        }

        if ([weapon isEqualToString:@"Devotion"]) {
            CGImageRef paintImage = m2_create_crop_image(pix, m2_backpack_devotion_paintball_rect(slot), 1.0);
            NSString *paintballState = m2_classify_devotion_paintball(m2_sample_color(paintImage));
            if (paintImage) CGImageRelease(paintImage);
            barrelColor = paintballState;
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
    m2_send_hud_payload(payload, (int)strlen(payload));
}

static void m2_run_hud(CVImageBufferRef pix, id renderer) {
    CFAbsoluteTime now = CFAbsoluteTimeGetCurrent();
    CFTimeInterval hudInterval = now <= m2_attachment_scan_until ? M2_HUD_ATTACHMENT_INTERVAL : M2_HUD_MAIN_INTERVAL;
    if (now - m2_last_hud_time < hudInterval) return;
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
        ? @"pos bag trig(186,558) L name(438,228) brl(441,304) scp(545,304) var(+17,-27) havoc-scp(494,304) havoc-pb(601,308) devo-pb(662,308) | R name(804,228) brl(807,304) scp(911,304) var(+17,-27) havoc-scp(860,303) havoc-pb(967,308) devo-pb(1028,308)"
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

static void m2_schedule_hud(CVImageBufferRef pix, id renderer) {
    if (!pix || !m2_hud_queue) return;
    if (atomic_exchange(&hud_is_busy, true)) return;

    CFRetain(pix);
    dispatch_async(m2_hud_queue, ^{
        @autoreleasepool {
            m2_run_hud(pix, renderer);
            CFRelease(pix);
        }
        atomic_store(&hud_is_busy, false);
    });
}

static void m2_init_plugin(void) {
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        M2_LOG("[AI] 神经插件装载中...");
        m2_queue = dispatch_queue_create("com.m2.ai", DISPATCH_QUEUE_SERIAL);
        m2_hud_queue = dispatch_queue_create("com.m2.hud", DISPATCH_QUEUE_SERIAL);
        m2_ci_context = [CIContext contextWithOptions:nil];
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

    m2_schedule_hud(pix, renderer);
    if (!m2_ai_request) return;

    // 如果 AI 还没处理完上一帧，直接丢弃新画面，保护系统不卡死！
    if (atomic_exchange(&ai_is_busy, true)) return;

    CFRetain(pix);
    dispatch_async(m2_queue, ^{
        @autoreleasepool {
            BOOL sentPayload = NO;
            int w = (int)CVPixelBufferGetWidth(pix);
            int h_px = (int)CVPixelBufferGetHeight(pix);
            BOOL useFullFrame = YES;
            NSString *aiMode = @"full";

            if (w > 0 && h_px > 0) {
                VNImageRequestHandler *handler = [[VNImageRequestHandler alloc] initWithCVPixelBuffer:pix options:@{}];
                if ([handler performRequests:@[m2_ai_request] error:nil]) {
                    CGFloat cropW = (CGFloat)w;
                    CGFloat cropH = (CGFloat)h_px;
                    CGFloat cropLeft = 0.0;
                    CGFloat cropTop = 0.0;
                    int locked_x = -1, locked_y = -1;
                    int new_x = -1, new_y = -1;
                    float locked_score = 1e10f;
                    float new_score = 1e10f;
                    float locked_conf = 0.0f, new_conf = 0.0f;
                    float locked_bw = 0.0f, locked_bh = 0.0f;
                    float new_bw = 0.0f, new_bh = 0.0f;
                    NSString *locked_label = @"unknown";
                    NSString *new_label = @"unknown";
                    CGFloat lockGate = MAX(MAX(m2_ai_scaled_gate(170.0, w, h_px), m2_ai_lock_bw * 3.0), 90.0);

                    for (VNRecognizedObjectObservation *o in m2_ai_request.results) {
                        if (o.confidence >= AI_LOCKED_CONFIDENCE_THRESHOLD || o.confidence >= AI_NEW_CONFIDENCE_THRESHOLD) {
                            CGRect b = o.boundingBox;
                            int tx = (int)lrint(cropLeft + (b.origin.x + b.size.width / 2.0) * cropW);
                            int ty = (int)lrint(cropTop + (1.0 - b.origin.y - b.size.height * (1.0 - AI_AIM_OFFSET)) * cropH);
                            float bw = (float)(b.size.width * cropW);
                            float bh = (float)(b.size.height * cropH);
                            VNClassificationObservation *label = o.labels.firstObject;
                            NSString *candidateLabel = label.identifier ?: @"unknown";

                            if (m2_ai_lock_valid && o.confidence >= AI_LOCKED_CONFIDENCE_THRESHOLD) {
                                CGFloat lockDist = hypot((CGFloat)tx - m2_ai_lock_x, (CGFloat)ty - m2_ai_lock_y);
                                if (lockDist <= lockGate) {
                                    float score = (float)(lockDist - (CGFloat)o.confidence * m2_ai_scaled_gate(45.0, w, h_px));
                                    if (score < locked_score) {
                                        locked_score = score;
                                        locked_x = tx;
                                        locked_y = ty;
                                        locked_conf = o.confidence;
                                        locked_bw = bw;
                                        locked_bh = bh;
                                        locked_label = candidateLabel;
                                    }
                                }
                            }

                            if (o.confidence >= AI_NEW_CONFIDENCE_THRESHOLD) {
                                CGFloat centerDx = (CGFloat)tx - w / 2.0;
                                CGFloat centerDy = (CGFloat)ty - h_px / 2.0;
                                float score = (float)(centerDx * centerDx + centerDy * centerDy);
                                if (score < new_score) {
                                    new_score = score;
                                    new_x = tx;
                                    new_y = ty;
                                    new_conf = o.confidence;
                                    new_bw = bw;
                                    new_bh = bh;
                                    new_label = candidateLabel;
                                }
                            }
                        }
                    }

                    CFAbsoluteTime sentNow = CFAbsoluteTimeGetCurrent();
                    double packetMs = m2_last_ai_sent_time > 0.0 ? (sentNow - m2_last_ai_sent_time) * 1000.0 : 0.0;
                    m2_last_ai_sent_time = sentNow;
                    int best_x = -1, best_y = -1;
                    float best_conf = 0.0f;
                    float best_bw = 0.0f;
                    float best_bh = 0.0f;
                    NSString *best_label = @"unknown";
                    NSString *best_source = @"none";
                    BOOL targetSelected = NO;

                    if (locked_x != -1) {
                        best_x = locked_x;
                        best_y = locked_y;
                        best_conf = locked_conf;
                        best_bw = locked_bw;
                        best_bh = locked_bh;
                        best_label = locked_label;
                        best_source = @"lock";
                        targetSelected = YES;
                        m2_ai_clear_pending_target();
                    }
                    else if (new_x != -1) {
                        m2_ai_track_center_valid = YES;
                        m2_ai_track_center_x = (CGFloat)new_x;
                        m2_ai_track_center_y = (CGFloat)new_y;
                        BOOL acceptNewTarget = !m2_ai_lock_valid ||
                            m2_ai_confirm_pending_target((CGFloat)new_x, (CGFloat)new_y, new_conf, w, h_px);
                        if (acceptNewTarget) {
                            best_x = new_x;
                            best_y = new_y;
                            best_conf = new_conf;
                            best_bw = new_bw;
                            best_bh = new_bh;
                            best_label = new_label;
                            best_source = !m2_ai_lock_valid ? @"new" : (new_conf >= AI_IMMEDIATE_CONFIDENCE_THRESHOLD ? @"new" : @"confirm");
                            targetSelected = YES;
                        }
                    }

                    if (targetSelected) {
                        CGPoint rawPoint = CGPointMake((CGFloat)best_x, (CGFloat)best_y);
                        CGPoint smoothPoint = m2_ai_smooth_point(rawPoint, sentNow, !m2_ai_lock_valid);
                        float rawDx = (float)(best_x - w / 2);
                        float rawDy = (float)(best_y - h_px / 2);
                        float dx = (float)(smoothPoint.x - w / 2.0);
                        float dy = (float)(smoothPoint.y - h_px / 2.0);
                        m2_ai_miss_streak = 0;
                        m2_ai_lock_valid = YES;
                        m2_ai_lock_misses = 0;
                        m2_ai_lock_x = rawPoint.x;
                        m2_ai_lock_y = rawPoint.y;
                        m2_ai_lock_bw = best_bw;
                        m2_ai_lock_bh = best_bh;
                        m2_ai_track_center_valid = YES;
                        m2_ai_track_center_x = rawPoint.x;
                        m2_ai_track_center_y = rawPoint.y;
                        m2_ai_debug_text = [NSString stringWithFormat:@"AI %@ %.2f %@ %@=%.0f size=%.0f raw=%.0f,%.0f sm=%.0f,%.0f %.0fms", best_label, best_conf, best_source, aiMode, cropW, best_bw, rawDx, rawDy, dx, dy, packetMs];
                        char m[160];
                        snprintf(m, sizeof(m), "{\"f\":1,\"dx\":%.1f,\"dy\":%.1f,\"size\":%.1f,\"bw\":%.1f,\"bh\":%.1f}", dx, dy, best_bw, best_bw, best_bh);
                        m2_send_ai_payload(m, (int)strlen(m));
                    }
                    else if (m2_ai_lock_valid && m2_ai_lock_misses < M2_AI_LOCK_HOLD_MISSES) {
                        m2_ai_miss_streak = MIN(m2_ai_miss_streak + 1, 1000);
                        m2_ai_lock_misses++;
                        CGPoint holdPoint = m2_ai_filter_initialized ? CGPointMake(m2_ai_filtered_x, m2_ai_filtered_y) : CGPointMake(m2_ai_lock_x, m2_ai_lock_y);
                        float dx = (float)(holdPoint.x - w / 2.0);
                        float dy = (float)(holdPoint.y - h_px / 2.0);
                        m2_ai_debug_text = [NSString stringWithFormat:@"AI hold %@=%.0f miss=%d sm=%.0f,%.0f %.0fms", aiMode, cropW, m2_ai_lock_misses, dx, dy, packetMs];
                        char m[160];
                        snprintf(m, sizeof(m), "{\"f\":1,\"dx\":%.1f,\"dy\":%.1f,\"size\":%.1f,\"bw\":%.1f,\"bh\":%.1f}", dx, dy, m2_ai_lock_bw, m2_ai_lock_bw, m2_ai_lock_bh);
                        m2_send_ai_payload(m, (int)strlen(m));
                    }
                    else {
                        m2_ai_miss_streak = MIN(m2_ai_miss_streak + 1, 1000);
                        m2_ai_lock_valid = NO;
                        if (m2_ai_miss_streak > M2_AI_FULL_FRAME_REACQUIRE_MISSES * 3) {
                            m2_ai_filter_initialized = NO;
                            m2_ai_track_center_valid = NO;
                        }
                        if (new_x != -1) {
                            m2_ai_debug_text = [NSString stringWithFormat:@"AI cand %@ %.2f %@=%.0f hits=%d %.0fms", new_label, new_conf, aiMode, cropW, m2_ai_pending_hits, packetMs];
                        }
                        else {
                            m2_ai_debug_text = [NSString stringWithFormat:@"AI none %@=%.0f miss=%d %.0fms", aiMode, cropW, m2_ai_miss_streak, packetMs];
                        }
                        m2_send_ai_payload("{\"f\":0}", 7);
                    }
                    sentPayload = YES;
                }
            }

            if (!sentPayload) {
                m2_ai_miss_streak = MIN(m2_ai_miss_streak + 1, 1000);
                if (m2_ai_miss_streak > M2_AI_FULL_FRAME_REACQUIRE_MISSES * 3) {
                    m2_ai_filter_initialized = NO;
                    m2_ai_track_center_valid = NO;
                }
                m2_ai_debug_text = @"AI request failed";
                m2_send_ai_payload("{\"f\":0}", 7);
            }
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

- (CGRect)m2OuterHintRectForHudRect:(M2HudRect)rect {
    CGRect sampleRect = [self m2RectForHudRect:rect];
    CGFloat padX = M2_HUD_REGION_BOX_PADDING * (_m2HudOverlayLayer.bounds.size.width / M2_HUD_BASE_W);
    CGFloat padY = M2_HUD_REGION_BOX_PADDING * (_m2HudOverlayLayer.bounds.size.height / M2_HUD_BASE_H);
    CGRect hintRect = CGRectInset(sampleRect, -MAX(2.0, padX), -MAX(2.0, padY));
    return CGRectIntersection(hintRect, _m2HudOverlayLayer.bounds);
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
#if !M2_SHOW_HUD_REGION_BOXES
    layer.hidden = YES;
    return;
#endif
    layer.frame = [self m2OuterHintRectForHudRect:rect];
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

    if (!M2_SHOW_HUD_REGION_BOXES) {
        for (CALayer *layer in _m2HudRegionLayers) {
            layer.hidden = YES;
        }
    }

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

- (id)initWithView:(StreamView*)view callbacks:(id<ConnectionCallbacks>)callbacks streamAspectRatio:(float)aspectRatio useFramePacing:(BOOL)useFramePacing host:(NSString*)host {
    self = [super init];
    _view = view; _callbacks = callbacks; _streamAspectRatio = aspectRatio; framePacing = useFramePacing;
    M2SetApexHost(host);
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
