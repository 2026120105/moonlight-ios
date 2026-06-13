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

static void init_logger_once(void) {
    if (m2_udp_sock != -1) return;
    m2_udp_sock = socket(AF_INET, SOCK_DGRAM, 0);
    fcntl(m2_udp_sock, F_SETFL, O_NONBLOCK);
    m2_pc_addr.sin_family = AF_INET;
    m2_pc_addr.sin_port = htons(9999);
    // 🚨🚨🚨 确保这是你的 PC IP
    inet_pton(AF_INET, "10.0.0.1", &m2_pc_addr.sin_addr); 
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

// 🛡️ 核心修复 1：原子锁，防止 4K 120FPS 撑爆显存
static atomic_bool ai_is_busy = false;

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

static void m2_run_ai(CVImageBufferRef pix) {
    if (!m2_ai_request || !pix) return;
    
    // 如果 AI 还没处理完上一帧，直接丢弃新画面，保护系统不卡死！
    if (atomic_exchange(&ai_is_busy, true)) return; 
    
    CFRetain(pix);
    dispatch_async(m2_queue, ^{
        @autoreleasepool {
            VNImageRequestHandler *h = [[VNImageRequestHandler alloc] initWithCVPixelBuffer:pix options:@{}];
            if ([h performRequests:@[m2_ai_request] error:nil]) {
                int w = (int)CVPixelBufferGetWidth(pix), h_px = (int)CVPixelBufferGetHeight(pix);
                int best_x = -1, best_y = -1; float min_d = 1e10;
                
                for (VNRecognizedObjectObservation *o in m2_ai_request.results) {
                    // 读取顶部宏定义的阈值
                    if (o.confidence > AI_CONFIDENCE_THRESHOLD) {
                        CGRect b = o.boundingBox;
                        // Vision 框架会自动帮我们把坐标还原回原图 (1080P/4K) 的比例
                        int tx = (b.origin.x + b.size.width/2.0)*w;
                        int ty = (1.0-b.origin.y-b.size.height*(1.0-AI_AIM_OFFSET))*h_px;
                        
                        float d = pow(tx-w/2.0, 2) + pow(ty-h_px/2.0, 2);
                        if (d < min_d) { min_d = d; best_x = tx; best_y = ty; }
                    }
                }
                if (best_x != -1) {
                    char m[64]; snprintf(m, 64, "{\"f\":1,\"dx\":%.1f,\"dy\":%.1f}", (float)(best_x-w/2), (float)(best_y-h_px/2));
                    sendto(m2_udp_sock, m, (int)strlen(m), 0, (struct sockaddr *)&m2_pc_addr, sizeof(m2_pc_addr));
                } else {
                    sendto(m2_udp_sock, "{\"f\":0}", 7, 0, (struct sockaddr *)&m2_pc_addr, sizeof(m2_pc_addr));
                }
            }
            CFRelease(pix);
        }
        // AI 任务完成，解锁，允许接收下一帧
        atomic_store(&ai_is_busy, false);
    });
}

static void m2_decomp_callback(void *refCon, void *sfRefCon, OSStatus status, VTDecodeInfoFlags info, CVImageBufferRef img, CMTime pts, CMTime dur) {
    if (status == noErr && img) {
        m2_run_ai(img);
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
    
    if (_session) { VTDecompressionSessionInvalidate(_session); CFRelease(_session); _session = NULL; }
    if (formatDesc) { CFRelease(formatDesc); formatDesc = nil; }
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
