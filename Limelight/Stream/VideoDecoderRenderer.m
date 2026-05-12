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

// ==========================================================
// 📡 [M2 ANE] 远程遥测 & 异步安全旁路
// ==========================================================
static int m2_udp_sock = -1;
static struct sockaddr_in m2_pc_addr;

static void init_logger_once(void) {
    if (m2_udp_sock != -1) return;
    m2_udp_sock = socket(AF_INET, SOCK_DGRAM, 0);
    fcntl(m2_udp_sock, F_SETFL, O_NONBLOCK);
    m2_pc_addr.sin_family = AF_INET;
    m2_pc_addr.sin_port = htons(9999);
    
    // 🚨🚨🚨 在这里填入你 PC 的真实 IP！
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

static void m2_init_plugin(void) {
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        M2_LOG("[AI] 神经插件装载中...");
        m2_queue = dispatch_queue_create("com.m2.ai", DISPATCH_QUEUE_SERIAL);
        dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
            NSURL *url = [[NSBundle mainBundle] URLForResource:@"best" withExtension:@"mlmodelc"];
            if (!url) { M2_LOG("[AI] ❌ 模型文件 best.mlmodelc 丢失！"); return; }
            MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
            config.computeUnits = MLComputeUnitsAll;
            NSError *err = nil;
            MLModel *ml = [MLModel modelWithContentsOfURL:url configuration:config error:&err];
            if (ml) {
                m2_ai_model = [VNCoreMLModel modelForMLModel:ml error:nil];
                m2_ai_request = [[VNCoreMLRequest alloc] initWithModel:m2_ai_model];
                m2_ai_request.imageCropAndScaleOption = VNImageCropAndScaleOptionScaleFill;
                M2_LOG("[AI] ✅ 神经引擎点火成功！");
            } else {
                M2_LOG("[AI] ❌ 模型装载失败");
            }
        });
    });
}

static void m2_run_ai(CVImageBufferRef pix) {
    if (!m2_ai_request || !pix) return;
    CFRetain(pix);
    dispatch_async(m2_queue, ^{
        @autoreleasepool {
            VNImageRequestHandler *h = [[VNImageRequestHandler alloc] initWithCVPixelBuffer:pix options:@{}];
            if ([h performRequests:@[m2_ai_request] error:nil]) {
                int w = (int)CVPixelBufferGetWidth(pix), h_px = (int)CVPixelBufferGetHeight(pix);
                int best_x = -1, best_y = -1; float min_d = 1e10;
                for (VNRecognizedObjectObservation *o in m2_ai_request.results) {
                    if (o.confidence > 0.45f) {
                        CGRect b = o.boundingBox;
                        int tx = (b.origin.x + b.size.width/2.0)*w, ty = (1.0-b.origin.y-b.size.height*0.8)*h_px;
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
    });
}

static void m2_decomp_callback(void *refCon, void *sfRefCon, OSStatus status, VTDecodeInfoFlags info, CVImageBufferRef img, CMTime pts, CMTime dur) {
    if (status == noErr && img) {
        m2_run_ai(img);
    }
}

// ==========================================================
// 🏗️ Moonlight 像素级安全渲染核心
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
    int debug_frame_count;
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
    debug_frame_count = 0;
    m2_init_plugin();
    [self reinitializeDisplayLayer];
    M2_LOG("[System] 渲染器初始化完毕");
    return self;
}

- (void)setupWithVideoFormat:(int)vf width:(int)vw height:(int)vh frameRate:(int)fr {
    M2_LOG("[Video] Format %d, %dx%d @ %d fps", vf, vw, vh, fr);
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

// 🛡️ 智能包头探测与转换
- (void)updateAnnexBBufferForRange:(CMBlockBufferRef)frameBuffer dataBlock:(CMBlockBufferRef)dataBlock offset:(int)offset length:(int)nalLength data:(unsigned char *)data {
    if (nalLength < 4) return;
    
    int startCodeLen = 3;
    if (nalLength >= 4 && data[offset] == 0 && data[offset+1] == 0 && data[offset+2] == 0 && data[offset+3] == 1) {
        startCodeLen = 4;
    } else if (nalLength >= 3 && data[offset] == 0 && data[offset+1] == 0 && data[offset+2] == 1) {
        startCodeLen = 3;
    } else {
        return; // 无效包头
    }
    
    OSStatus status;
    size_t oldOffset = CMBlockBufferGetDataLength(frameBuffer);
    status = CMBlockBufferAppendMemoryBlock(frameBuffer, NULL, 4, kCFAllocatorDefault, NULL, 0, 4, 0);
    if (status != noErr) return;
    
    const int dataLength = nalLength - startCodeLen;
    const uint8_t lengthBytes[] = {(uint8_t)(dataLength >> 24), (uint8_t)(dataLength >> 16), (uint8_t)(dataLength >> 8), (uint8_t)dataLength};
    status = CMBlockBufferReplaceDataBytes(lengthBytes, frameBuffer, oldOffset, 4);
    if (status != noErr) return;
    
    CMBlockBufferAppendBufferReference(frameBuffer, dataBlock, offset + startCodeLen, dataLength, 0);
}

- (int)submitDecodeBuffer:(unsigned char *)data length:(int)length bufferType:(int)bt decodeUnit:(PDECODE_UNIT)du {
    if (debug_frame_count < 5) { M2_LOG("[Decode] 收到包长: %d (类型: %d)", length, du->frameType); debug_frame_count++; }

    if (du->frameType == FRAME_TYPE_IDR) {
        if (bt != 4) {
            if (bt >= 1 && bt <= 3) {
                int startLen = 3;
                if (length >= 4 && data[0] == 0 && data[1] == 0 && data[2] == 0 && data[3] == 1) startLen = 4;
                [parameterSetBuffers addObject:[NSData dataWithBytes:&data[startLen] length:length - startLen]];
            }
            return 0; // 头文件成功接收，返回 0
        }
        
        M2_LOG("[Decode] 📦 开始处理 IDR 图像帧");
        if (formatDesc != NULL) { CFRelease(formatDesc); formatDesc = NULL; }
        if (_session != NULL) { VTDecompressionSessionInvalidate(_session); CFRelease(_session); _session = NULL; }
        
        size_t pc = [parameterSetBuffers count];
        if (pc == 0) { 
            M2_LOG("[Decode] ❌ 缺失 SPS/PPS！请求重传..."); 
            free(data); return 1; // 🚨 请求电脑重新发一帧！
        }
        
        const uint8_t* pps[pc]; size_t pss[pc];
        for (int i = 0; i < pc; i++) { NSData* p = parameterSetBuffers[i]; pps[i] = p.bytes; pss[i] = p.length; }
        
        if (videoFormat & 0x01) {
            CMVideoFormatDescriptionCreateFromH264ParameterSets(NULL, pc, pps, pss, 4, &formatDesc);
        } else if (videoFormat & 0x02) {
            CMVideoFormatDescriptionCreateFromHEVCParameterSets(NULL, pc, pps, pss, 4, NULL, &formatDesc);
        }
        [parameterSetBuffers removeAllObjects];
        
        if (formatDesc) {
            VTDecompressionOutputCallbackRecord cb = {m2_decomp_callback, (__bridge void *)self};
            NSDictionary *attr = @{(id)kCVPixelBufferPixelFormatTypeKey: @(kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange)};
            OSStatus st = VTDecompressionSessionCreate(NULL, formatDesc, NULL, (__bridge CFDictionaryRef)attr, &cb, &_session);
            if (st != noErr) {
                M2_LOG("[Decode] ❌ 会话建立失败: %d", (int)st);
                free(data); return 1; // 🚨 建立失败，请求重传！
            }
        } else {
            M2_LOG("[Decode] ❌ formatDesc 描述文件生成失败！");
            free(data); return 1;
        }
    }
    
    if (!formatDesc || !_session) { free(data); return 1; }
    
    CMBlockBufferRef fbb, dbb;
    CMBlockBufferCreateWithMemoryBlock(NULL, data, length, kCFAllocatorDefault, NULL, 0, length, 0, &dbb);
    CMBlockBufferCreateEmpty(NULL, 0, 0, &fbb);
    
    // 🛡️ 精确的字节扫描逻辑
    int last = -1;
    for (int i = 0; i < length - 2; i++) {
        if (data[i] == 0 && data[i+1] == 0 && data[i+2] == 1) {
            int start_idx = i;
            if (i > 0 && data[i-1] == 0) start_idx = i - 1; // 兼容 4 字节前缀
            
            if (last != -1) {
                [self updateAnnexBBufferForRange:fbb dataBlock:dbb offset:last length:start_idx - last data:data];
            }
            last = start_idx;
            i += 2; // 跳过这几个字节，防止重复判断
        }
    }
    if (last != -1) {
        [self updateAnnexBBufferForRange:fbb dataBlock:dbb offset:last length:length - last data:data];
    } else {
        CMBlockBufferAppendBufferReference(fbb, dbb, 0, length, 0); // 找不到包头就全盘塞入
    }
    
    CMSampleBufferRef sb;
    CMSampleTimingInfo ti = {kCMTimeInvalid, CMTimeMake(du->presentationTimeMs, 1000), kCMTimeInvalid};
    OSStatus sbStatus = CMSampleBufferCreateReady(NULL, fbb, formatDesc, 1, 1, &ti, 0, NULL, &sb);
    
    if (sbStatus == noErr) {
        [displayLayer enqueueSampleBuffer:sb];
        VTDecompressionSessionDecodeFrame(_session, sb, kVTDecodeFrame_EnableAsynchronousDecompression, NULL, NULL);
    } else {
        M2_LOG("[Decode] ❌ SampleBuffer 构建失败: %d，请求服务器重发关键帧！", (int)sbStatus);
        CFRelease(dbb); CFRelease(fbb); free(data);
        return 1; // 🚨 这是终结黑屏死锁的最关键一行！
    }
    
    if (du->frameType == FRAME_TYPE_IDR) { 
        displayLayer.hidden = NO; 
        [_callbacks videoContentShown]; 
        M2_LOG("[Decode] 🚀 画面解码出炉！");
    }
    
    CFRelease(dbb); CFRelease(fbb); if (sb) CFRelease(sb);
    return 0;
}

- (void)setHdrMode:(BOOL)enabled {}
@end
