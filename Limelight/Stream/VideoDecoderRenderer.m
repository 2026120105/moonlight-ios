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
// 🧠 [M2 ANE] 异步旁路 AI 引擎 (100% 官方兼容版)
// ==========================================================
static int m2_udp_sock = -1;
static struct sockaddr_in m2_pc_addr;
static VNCoreMLModel *m2_ai_model = nil;
static VNCoreMLRequest *m2_ai_request = nil;
static dispatch_queue_t m2_queue = nil;

// AI 插件静默启动
static void m2_init_plugin(void) {
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        m2_queue = dispatch_queue_create("com.m2.ai_logic", DISPATCH_QUEUE_SERIAL);
        m2_udp_sock = socket(AF_INET, SOCK_DGRAM, 0);
        fcntl(m2_udp_sock, F_SETFL, O_NONBLOCK);
        m2_pc_addr.sin_family = AF_INET;
        m2_pc_addr.sin_port = htons(9999);
        inet_pton(AF_INET, "10.0.0.1", &m2_pc_addr.sin_addr); // ⚠️ 确认你的 PC IP

        dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
            NSURL *url = [[NSBundle mainBundle] URLForResource:@"best" withExtension:@"mlmodelc"];
            if (url) {
                MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
                config.computeUnits = MLComputeUnitsAll; // 激活 M2 ANE
                MLModel *ml = [MLModel modelWithContentsOfURL:url configuration:config error:nil];
                if (ml) {
                    m2_ai_model = [VNCoreMLModel modelForMLModel:ml error:nil];
                    m2_ai_request = [[VNCoreMLRequest alloc] initWithModel:m2_ai_model];
                    m2_ai_request.imageCropAndScaleOption = VNImageCropAndScaleOptionScaleFill;
                    printf("✅ [M2 AI] ANE 引擎后台就绪！\n");
                }
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
                        // 坐标计算：包含 20% 的高度修正
                        int tx = (b.origin.x + b.size.width/2.0)*w, ty = (1.0-b.origin.y-b.size.height*0.8)*h_px;
                        float d = pow(tx-w/2, 2) + pow(ty-h_px/2, 2);
                        if (d < min_d) { min_d = d; best_x = tx; best_y = ty; }
                    }
                }
                if (best_x != -1) {
                    char m[64]; snprintf(m, 64, "{\"f\":1,\"dx\":%d,\"dy\":%d}", best_x-w/2, best_y-h_px/2);
                    sendto(m2_udp_sock, m, (int)strlen(m), 0, (struct sockaddr *)&m2_pc_addr, sizeof(m2_pc_addr));
                } else {
                    sendto(m2_udp_sock, "{\"f\":0}", 7, 0, (struct sockaddr *)&m2_pc_addr, sizeof(m2_pc_addr));
                }
            }
            CFRelease(pix);
        }
    });
}

// ==========================================================
// 🏗️ Moonlight 官方兼容层
// ==========================================================
extern int ff_isom_write_av1c(AVIOContext *pb, const uint8_t *buf, int size, int write_seq_header);

static void m2_decompression_callback(void *refCon, void *sfRefCon, OSStatus status, VTDecodeInfoFlags info, CVImageBufferRef img, CMTime pts, CMTime dur) {
    if (status == noErr && img) {
        m2_init_plugin();
        m2_run_ai(img);
    }
}

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
    [self reinitializeDisplayLayer];
    return self;
}

- (void)setupWithVideoFormat:(int)vf width:(int)vw height:(int)vh frameRate:(int)fr {
    self->videoFormat = vf; self->frameRate = fr;
}

- (void)start {
    _displayLink = [CADisplayLink displayLinkWithTarget:self selector:@selector(displayLinkCallback:)];
    _displayLink.preferredFramesPerSecond = self->frameRate;
    [_displayLink addToRunLoop:[NSRunLoop mainRunLoop] forMode:NSDefaultRunLoopMode];
}

- (void)displayLinkCallback:(CADisplayLink *)sender {
    VIDEO_FRAME_HANDLE h; PDECODE_UNIT du;
    while (LiPollNextVideoFrame(&h, &du)) {
        LiCompleteVideoFrame(h, DrSubmitDecodeUnit(du));
    }
}

- (void)stop {
    [_displayLink invalidate];
    if (_session) { VTDecompressionSessionInvalidate(_session); CFRelease(_session); _session = NULL; }
}

int DrSubmitDecodeUnit(PDECODE_UNIT du);

- (int)submitDecodeBuffer:(unsigned char *)data length:(int)length bufferType:(int)bt decodeUnit:(PDECODE_UNIT)du {
    if (du->frameType == FRAME_TYPE_IDR) {
        if (bt != 4) {
            if (bt >= 1 && bt <= 3) [parameterSetBuffers addObject:[NSData dataWithBytes:&data[data[2]==0x01?3:4] length:length-(data[2]==0x01?3:4)]];
            return 0;
        }
        [self reinitializeDisplayLayer];
        size_t pc = [parameterSetBuffers count];
        const uint8_t* pps[pc]; size_t pss[pc];
        for (int i=0; i<pc; i++) { NSData* p = parameterSetBuffers[i]; pps[i]=p.bytes; pss[i]=p.length; }
        if (videoFormat & 0x01) CMVideoFormatDescriptionCreateFromH264ParameterSets(NULL, pc, pps, pss, 4, &formatDesc);
        else if (videoFormat & 0x02) CMVideoFormatDescriptionCreateFromHEVCParameterSets(NULL, pc, pps, pss, 4, NULL, &formatDesc);
        [parameterSetBuffers removeAllObjects];
        
        VTDecompressionOutputCallbackRecord cb = {m2_decompression_callback, (__bridge void *)self};
        NSDictionary *attr = @{(id)kCVPixelBufferPixelFormatTypeKey: @(kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange)};
        VTDecompressionSessionCreate(NULL, formatDesc, NULL, (__bridge CFDictionaryRef)attr, &cb, &_session);
    }
    
    if (!formatDesc || !_session) { free(data); return 1; }
    
    CMBlockBufferRef fbb, dbb;
    CMBlockBufferCreateWithMemoryBlock(NULL, data, length, kCFAllocatorDefault, NULL, 0, length, 0, &dbb);
    CMBlockBufferCreateEmpty(NULL, 0, 0, &fbb);
    
    int last = -1;
    for (int i=0; i<length-3; i++) {
        if (data[i]==0 && data[i+1]==0 && data[i+2]==1) {
            if (last != -1) {
                size_t old = CMBlockBufferGetDataLength(fbb);
                CMBlockBufferAppendMemoryBlock(fbb, NULL, 4, kCFAllocatorDefault, NULL, 0, 4, 0);
                int dl = i - last - 3;
                uint8_t lb[] = {(uint8_t)(dl>>24),(uint8_t)(dl>>16),(uint8_t)(dl>>8),(uint8_t)dl};
                CMBlockBufferReplaceDataBytes(lb, fbb, old, 4);
                CMBlockBufferAppendBufferReference(fbb, dbb, last+3, dl, 0);
            }
            last = i;
        }
    }
    if (last != -1) {
        size_t old = CMBlockBufferGetDataLength(fbb);
        CMBlockBufferAppendMemoryBlock(fbb, NULL, 4, kCFAllocatorDefault, NULL, 0, 4, 0);
        int dl = length - last - 3;
        uint8_t lb[] = {(uint8_t)(dl>>24),(uint8_t)(dl>>16),(uint8_t)(dl>>8),(uint8_t)dl};
        CMBlockBufferReplaceDataBytes(lb, fbb, old, 4);
        CMBlockBufferAppendBufferReference(fbb, dbb, last+3, dl, 0);
    }

    CMSampleBufferRef sb;
    CMSampleTimingInfo ti = {kCMTimeInvalid, CMTimeMake(du->presentationTimeMs, 1000), kCMTimeInvalid};
    CMSampleBufferCreateReady(NULL, fbb, formatDesc, 1, 1, &ti, 0, NULL, &sb);
    
    // 🚀 核心修复：直接投递显示，绝不等待任何回调
    [displayLayer enqueueSampleBuffer:sb];
    // 后台并行触发 AI 解码
    VTDecompressionSessionDecodeFrame(_session, sb, 0, NULL, NULL);
    
    if (du->frameType == FRAME_TYPE_IDR) { displayLayer.hidden = NO; [_callbacks videoContentShown]; }
    CFRelease(dbb); CFRelease(fbb); CFRelease(sb);
    return 0;
}

- (void)setHdrMode:(BOOL)enabled {}
@end
