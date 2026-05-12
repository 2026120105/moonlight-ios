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
// 📡 远程遥测模块 (UDP Logger)
// ==========================================================
static int m2_udp_sock = -1;
static struct sockaddr_in m2_pc_addr;

// 初始化通信管道
static void init_logger_once(void) {
    if (m2_udp_sock != -1) return;
    m2_udp_sock = socket(AF_INET, SOCK_DGRAM, 0);
    fcntl(m2_udp_sock, F_SETFL, O_NONBLOCK);
    m2_pc_addr.sin_family = AF_INET;
    m2_pc_addr.sin_port = htons(9999);
    inet_pton(AF_INET, "10.0.0.1", &m2_pc_addr.sin_addr); // ⚠️ 确保这是你 PC 的 IP
}

// 向 PC 发送日志
static void M2_LOG(const char *format, ...) {
    init_logger_once();
    char buffer[512];
    va_list args;
    va_start(args, format);
    vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);
    sendto(m2_udp_sock, buffer, strlen(buffer), 0, (struct sockaddr *)&m2_pc_addr, sizeof(m2_pc_addr));
}

// ==========================================================
// 🧠 [M2 ANE] 异步旁路 AI 引擎
// ==========================================================
static VNCoreMLModel *m2_ai_model = nil;
static VNCoreMLRequest *m2_ai_request = nil;
static dispatch_queue_t m2_queue = nil;

static void m2_init_plugin(void) {
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        M2_LOG("[AI] 正在初始化神经插件...");
        m2_queue = dispatch_queue_create("com.m2.ai", DISPATCH_QUEUE_SERIAL);
        
        dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
            NSURL *url = [[NSBundle mainBundle] URLForResource:@"best" withExtension:@"mlmodelc"];
            if (!url) {
                M2_LOG("[AI] ❌ 致命错误：找不到 best.mlmodelc！打包失败或路径错误。");
                return;
            }
            M2_LOG("[AI] 找到模型文件，开始装载 ANE...");
            
            MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
            config.computeUnits = MLComputeUnitsAll;
            NSError *err = nil;
            MLModel *ml = [MLModel modelWithContentsOfURL:url configuration:config error:&err];
            if (ml) {
                m2_ai_model = [VNCoreMLModel modelForMLModel:ml error:nil];
                m2_ai_request = [[VNCoreMLRequest alloc] initWithModel:m2_ai_model];
                m2_ai_request.imageCropAndScaleOption = VNImageCropAndScaleOptionScaleFill;
                M2_LOG("[AI] ✅ 神经引擎装载成功，准备就绪！");
            } else {
                M2_LOG("[AI] ❌ 模型装载失败：%@", [[err localizedDescription] UTF8String]);
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
            [h performRequests:@[m2_ai_request] error:nil];
            // 暂时隐藏发送坐标的代码，先专注排查黑屏
            CFRelease(pix);
        }
    });
}

// ==========================================================
// 🏗️ Moonlight 官方兼容层 (带探针)
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
    int debug_frame_count; // 用于限制日志刷屏
}

- (void)reinitializeDisplayLayer {
    M2_LOG("[Video] 初始化/重置显示层...");
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
    M2_LOG("[System] VideoDecoderRenderer 初始化完毕");
    [self reinitializeDisplayLayer];
    return self;
}

- (void)setupWithVideoFormat:(int)vf width:(int)vw height:(int)vh frameRate:(int)fr {
    M2_LOG("[Video] 收到视频格式设置: Format %d, %dx%d @ %d fps", vf, vw, vh, fr);
    self->videoFormat = vf; self->frameRate = fr;
}

- (void)start {
    M2_LOG("[Video] 启动 DisplayLink");
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
    M2_LOG("[Video] 停止视频流");
    [_displayLink invalidate];
    if (_session) { VTDecompressionSessionInvalidate(_session); CFRelease(_session); _session = NULL; }
}

int DrSubmitDecodeUnit(PDECODE_UNIT du);

- (int)submitDecodeBuffer:(unsigned char *)data length:(int)length bufferType:(int)bt decodeUnit:(PDECODE_UNIT)du {
    if (debug_frame_count < 20) {
        M2_LOG("[Decode] 收到帧数据: length=%d, type=%d", length, du->frameType);
        debug_frame_count++;
    }

    if (du->frameType == FRAME_TYPE_IDR) {
        M2_LOG("[Decode] 接收到关键帧 (IDR)!");
        if (bt != 4) {
            if (bt >= 1 && bt <= 3) [parameterSetBuffers addObject:[NSData dataWithBytes:&data[data[2]==0x01?3:4] length:length-(data[2]==0x01?3:4)]];
            return 0;
        }
        [self reinitializeDisplayLayer];
        size_t pc = [parameterSetBuffers count];
        const uint8_t* pps[pc]; size_t pss[pc];
        for (int i=0; i<pc; i++) { NSData* p = parameterSetBuffers[i]; pps[i]=p.bytes; pss[i]=p.length; }
        
        if (videoFormat & 0x01) {
            CMVideoFormatDescriptionCreateFromH264ParameterSets(NULL, pc, pps, pss, 4, &formatDesc);
        } else if (videoFormat & 0x02) {
            CMVideoFormatDescriptionCreateFromHEVCParameterSets(NULL, pc, pps, pss, 4, NULL, &formatDesc);
        }
        [parameterSetBuffers removeAllObjects];
        
        if (!formatDesc) {
            M2_LOG("[Decode] ❌ 致命错误：formatDesc 创建失败！H.264/H.265 头数据可能不匹配。");
            free(data); return 1;
        }
        
        VTDecompressionOutputCallbackRecord cb = {m2_decompression_callback, (__bridge void *)self};
        NSDictionary *attr = @{(id)kCVPixelBufferPixelFormatTypeKey: @(kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange)};
        OSStatus status = VTDecompressionSessionCreate(NULL, formatDesc, NULL, (__bridge CFDictionaryRef)attr, &cb, &_session);
        
        if (status != noErr || !_session) {
            M2_LOG("[Decode] ❌ 致命错误：VideoToolbox 解码会话创建失败! OSStatus: %d", (int)status);
        } else {
            M2_LOG("[Decode] ✅ VTDecompressionSession 创建成功！");
        }
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
    OSStatus sbStatus = CMSampleBufferCreateReady(NULL, fbb, formatDesc, 1, 1, &ti, 0, NULL, &sb);
    
    if (sbStatus != noErr) {
        if (debug_frame_count < 25) { M2_LOG("[Decode] ❌ SampleBuffer 创建失败: %d", (int)sbStatus); debug_frame_count++; }
        CFRelease(dbb); CFRelease(fbb); free(data); return 1;
    }
    
    [displayLayer enqueueSampleBuffer:sb];
    VTDecompressionSessionDecodeFrame(_session, sb, 0, NULL, NULL);
    
    if (du->frameType == FRAME_TYPE_IDR) { 
        displayLayer.hidden = NO; 
        [_callbacks videoContentShown]; 
        M2_LOG("[Decode] 🚀 关键帧渲染完成，通知上层消除黑屏！");
    }
    
    CFRelease(dbb); CFRelease(fbb); CFRelease(sb);
    return 0;
}

- (void)setHdrMode:(BOOL)enabled {}
@end
