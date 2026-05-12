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
// 🧠 [M2 ANE] 异步非阻塞 AI 引擎 v4.0 (防黑屏稳定版)
// ==========================================================

static int m2_udp_sock = -1;
static struct sockaddr_in m2_pc_addr;
static VNCoreMLModel *m2_ai_model = nil;
static VNCoreMLRequest *m2_ai_request = nil;
static BOOL ai_engine_ready = NO;
static BOOL ai_failed_once = NO; // 防止重复报错死循环
static dispatch_queue_t m2_ai_queue = nil;

// 预热 AI 环境：只在后台运行，不占用主线程渲染
static void m2_warmup_ai(void) {
    if (ai_engine_ready || ai_failed_once) return;
    
    m2_ai_queue = dispatch_queue_create("com.m2.ai_tracker", DISPATCH_QUEUE_SERIAL);
    
    // 初始化网络
    m2_udp_sock = socket(AF_INET, SOCK_DGRAM, 0);
    fcntl(m2_udp_sock, F_SETFL, fcntl(m2_udp_sock, F_GETFL, 0) | O_NONBLOCK);
    m2_pc_addr.sin_family = AF_INET;
    m2_pc_addr.sin_port = htons(9999);
    // ⚠️ 检查：请确保 PC 端 Sunshine 运行在此 IP
    inet_pton(AF_INET, "10.0.0.1", &m2_pc_addr.sin_addr); 

    // 寻找模型
    NSURL *modelURL = [[NSBundle mainBundle] URLForResource:@"best" withExtension:@"mlmodelc"];
    if (!modelURL) {
        printf("⚠️ [M2 AI] 没找到 best.mlmodelc，进入纯流模式。\n");
        ai_failed_once = YES;
        return;
    }

    MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
    config.computeUnits = MLComputeUnitsAll; // 激活 M2 ANE
    
    NSError *err = nil;
    MLModel *mlModel = [MLModel modelWithContentsOfURL:modelURL configuration:config error:&err];
    if (mlModel) {
        m2_ai_model = [VNCoreMLModel modelForMLModel:mlModel error:&err];
        m2_ai_request = [[VNCoreMLRequest alloc] initWithModel:m2_ai_model];
        m2_ai_request.imageCropAndScaleOption = VNImageCropAndScaleOptionScaleFill;
        ai_engine_ready = YES;
        printf("✅ [M2 AI] 神经引擎点火成功！\n");
    } else {
        ai_failed_once = YES;
    }
}

static void m2_inference_async(CVImageBufferRef pixelBuffer) {
    if (!ai_engine_ready || !pixelBuffer) return;

    // ⚡ 核心：将图像数据保留（Retain），丢给后台队列处理
    CFRetain(pixelBuffer);
    dispatch_async(m2_ai_queue, ^{
        @autoreleasepool {
            int width = (int)CVPixelBufferGetWidth(pixelBuffer);
            int height = (int)CVPixelBufferGetHeight(pixelBuffer);
            
            VNImageRequestHandler *handler = [[VNImageRequestHandler alloc] initWithCVPixelBuffer:pixelBuffer options:@{}];
            [handler performRequests:@[m2_ai_request] error:nil];

            int best_x = -1, best_y = -1;
            long min_dist = 2000000000;
            int cx = width / 2;
            int cy = height / 2;

            if (m2_ai_request.results) {
                for (VNRecognizedObjectObservation *obs in m2_ai_request.results) {
                    if (obs.confidence > 0.45f) {
                        CGRect bbox = obs.boundingBox;
                        int bx = (int)(bbox.origin.x * width);
                        int bw = (int)(bbox.size.width * width);
                        int bh = (int)(bbox.size.height * height);
                        int by = (int)((1.0 - bbox.origin.y - bbox.size.height) * height);
                        
                        int tx = bx + bw / 2;
                        int ty = by + (int)(bh * 0.20f); // 瞄准点下压 20%
                        
                        long d = (tx-cx)*(tx-cx) + (ty-cy)*(ty-cy);
                        if (d < min_dist) { min_dist = d; best_x = tx; best_y = ty; }
                    }
                }
            }

            if (best_x != -1) {
                char msg[64];
                snprintf(msg, sizeof(msg), "{\"f\":1,\"dx\":%.1f,\"dy\":%.1f}", (float)(best_x - cx), (float)(best_y - cy));
                sendto(m2_udp_sock, msg, (int)strlen(msg), 0, (struct sockaddr *)&m2_pc_addr, sizeof(m2_pc_addr));
            } else {
                sendto(m2_udp_sock, "{\"f\":0}", 7, 0, (struct sockaddr *)&m2_pc_addr, sizeof(m2_pc_addr));
            }
            CFRelease(pixelBuffer); // 处理完释放内存
        }
    });
}

// 硬件解码回调（旁路模式）
static void m2_decomp_callback(void *ref, void *sf, OSStatus status, VTDecodeInfoFlags info, CVImageBufferRef image, CMTime pts, CMTime dur) {
    if (status == noErr && image) {
        // AI 逻辑仅作为旁路：失败了也不要紧
        m2_inference_async(image);
    }
}

// ==========================================================
// 🏗️ Moonlight 官方核心：确保显示绝对流畅
// ==========================================================

extern int ff_isom_write_av1c(AVIOContext *pb, const uint8_t *buf, int size, int write_seq_header);

@implementation VideoDecoderRenderer {
    StreamView* _view;
    id<ConnectionCallbacks> _callbacks;
    float _streamAspectRatio;
    AVSampleBufferDisplayLayer* displayLayer;
    int videoFormat;
    int frameRate;
    NSMutableArray *parameterSetBuffers;
    CMVideoFormatDescriptionRef formatDesc;
    CADisplayLink* _displayLink;
    BOOL framePacing;
    VTDecompressionSessionRef _m2Session; // AI 专用的解码会话
}

- (void)reinitializeDisplayLayer {
    if (displayLayer) [displayLayer removeFromSuperlayer];
    displayLayer = [[AVSampleBufferDisplayLayer alloc] init];
    displayLayer.backgroundColor = [UIColor blackColor].CGColor;
    
    CGSize vSize;
    if (_view.bounds.size.width > _view.bounds.size.height * _streamAspectRatio) {
        vSize = CGSizeMake(_view.bounds.size.height * _streamAspectRatio, _view.bounds.size.height);
    } else {
        vSize = CGSizeMake(_view.bounds.size.width, _view.bounds.size.width / _streamAspectRatio);
    }
    displayLayer.position = CGPointMake(CGRectGetMidX(_view.bounds), CGRectGetMidY(_view.bounds));
    displayLayer.bounds = CGRectMake(0, 0, vSize.width, vSize.height);
    displayLayer.videoGravity = AVLayerVideoGravityResize;
    displayLayer.hidden = YES;
    [_view.layer addSublayer:displayLayer];
    
    if (formatDesc) { CFRelease(formatDesc); formatDesc = nil; }
    if (_m2Session) { VTDecompressionSessionInvalidate(_m2Session); CFRelease(_m2Session); _m2Session = NULL; }
}

- (id)initWithView:(StreamView*)view callbacks:(id<ConnectionCallbacks>)callbacks streamAspectRatio:(float)aspectRatio useFramePacing:(BOOL)useFramePacing {
    self = [super init];
    _view = view; _callbacks = callbacks; _streamAspectRatio = aspectRatio; framePacing = useFramePacing;
    parameterSetBuffers = [[NSMutableArray alloc] init];
    [self reinitializeDisplayLayer];
    m2_warmup_ai(); // 启动时预热 AI
    return self;
}

- (void)setupWithVideoFormat:(int)videoFormat width:(int)videoWidth height:(int)videoHeight frameRate:(int)frameRate {
    self->videoFormat = videoFormat; self->frameRate = frameRate;
}

- (void)start {
    _displayLink = [CADisplayLink displayLinkWithTarget:self selector:@selector(displayLinkCallback:)];
    _displayLink.preferredFramesPerSecond = self->frameRate;
    [_displayLink addToRunLoop:[NSRunLoop mainRunLoop] forMode:NSDefaultRunLoopMode];
}

int DrSubmitDecodeUnit(PDECODE_UNIT du);

- (void)displayLinkCallback:(CADisplayLink *)sender {
    VIDEO_FRAME_HANDLE handle; PDECODE_UNIT du;
    while (LiPollNextVideoFrame(&handle, &du)) {
        LiCompleteVideoFrame(handle, DrSubmitDecodeUnit(du));
    }
}

- (void)stop {
    [_displayLink invalidate];
    if (_m2Session) { VTDecompressionSessionInvalidate(_m2Session); CFRelease(_m2Session); _m2Session = NULL; }
}

- (void)updateAnnexB:(CMBlockBufferRef)fbb data:(CMBlockBufferRef)dbb offset:(int)off length:(int)len {
    size_t old = CMBlockBufferGetDataLength(fbb);
    CMBlockBufferAppendMemoryBlock(fbb, NULL, 4, kCFAllocatorDefault, NULL, 0, 4, 0);
    const int dl = len - 3;
    const uint8_t lb[] = {(uint8_t)(dl >> 24), (uint8_t)(dl >> 16), (uint8_t)(dl >> 8), (uint8_t)dl};
    CMBlockBufferReplaceDataBytes(lb, fbb, old, 4);
    CMBlockBufferAppendBufferReference(fbb, dbb, off + 3, dl, 0);
}

- (int)submitDecodeBuffer:(unsigned char *)data length:(int)length bufferType:(int)bt decodeUnit:(PDECODE_UNIT)du {
    if (du->frameType == FRAME_TYPE_IDR) {
        if (bt != 4) { // 不是图片数据，是头数据
            if (bt >= 1 && bt <= 3) [parameterSetBuffers addObject:[NSData dataWithBytes:&data[data[2]==0x01?3:4] length:length-(data[2]==0x01?3:4)]];
            return 0;
        }
        if (formatDesc) { CFRelease(formatDesc); formatDesc = nil; }
        size_t pc = [parameterSetBuffers count];
        const uint8_t* pps[pc]; size_t pss[pc];
        for (int i=0; i<pc; i++) { NSData* p = parameterSetBuffers[i]; pps[i] = p.bytes; pss[i] = p.length; }
        if (videoFormat & 0x01) CMVideoFormatDescriptionCreateFromH264ParameterSets(NULL, pc, pps, pss, 4, &formatDesc);
        else if (videoFormat & 0x02) CMVideoFormatDescriptionCreateFromHEVCParameterSets(NULL, pc, pps, pss, 4, NULL, &formatDesc);
        [parameterSetBuffers removeAllObjects];
        
        // 重建 AI 旁路会话
        if (_m2Session) { VTDecompressionSessionInvalidate(_m2Session); CFRelease(_m2Session); _m2Session = NULL; }
        VTDecompressionOutputCallbackRecord cb = { .decompressionOutputCallback = m2_decomp_callback };
        NSDictionary *attr = @{ (id)kCVPixelBufferPixelFormatTypeKey: @(kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange) };
        VTDecompressionSessionCreate(NULL, formatDesc, NULL, (__bridge CFDictionaryRef)attr, &cb, &_m2Session);
    }

    if (!formatDesc) { free(data); return 1; }
    
    CMBlockBufferRef fbb, dbb;
    CMBlockBufferCreateWithMemoryBlock(NULL, data, length, kCFAllocatorDefault, NULL, 0, length, 0, &dbb);
    CMBlockBufferCreateEmpty(NULL, 0, 0, &fbb);
    int last = -1;
    for (int i=0; i<length-3; i++) {
        if (data[i]==0 && data[i+1]==0 && data[i+2]==1) {
            if (last != -1) [self updateAnnexB:fbb data:dbb offset:last length:i-last];
            last = i;
        }
    }
    if (last != -1) [self updateAnnexB:fbb data:dbb offset:last length:length-last];
    
    CMSampleBufferRef sb;
    CMSampleTimingInfo ti = {kCMTimeInvalid, CMTimeMake(du->presentationTimeMs, 1000), kCMTimeInvalid};
    CMSampleBufferCreateReady(NULL, fbb, formatDesc, 1, 1, &ti, 0, NULL, &sb);
    
    // 🛡️ 核心：无论 AI 如何，先给 displayLayer 渲染显示画面！
    [self->displayLayer enqueueSampleBuffer:sb];
    
    // 🧠 AI 推理仅仅是“顺带”执行
    if (_m2Session) {
        VTDecompressionSessionDecodeFrame(_m2Session, sb, kVTDecodeFrame_EnableAsynchronousDecompression, NULL, NULL);
    }
    
    if (du->frameType == FRAME_TYPE_IDR) { self->displayLayer.hidden = NO; [self->_callbacks videoContentShown]; }
    CFRelease(dbb); CFRelease(fbb); CFRelease(sb);
    return 0;
}

- (void)setHdrMode:(BOOL)enabled {}
@end
