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
// 🧠 [M2 ANE] 异步纯旁路 AI 引擎 (零干涉架构)
// ==========================================================
static int m2_udp_sock = -1;
static struct sockaddr_in m2_pc_addr;
static VNCoreMLModel *m2_ai_model = nil;
static VNCoreMLRequest *m2_ai_request = nil;
static dispatch_queue_t m2_queue = nil;

static void m2_init_plugin(void) {
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        m2_queue = dispatch_queue_create("com.m2.ai", DISPATCH_QUEUE_SERIAL);
        m2_udp_sock = socket(AF_INET, SOCK_DGRAM, 0);
        fcntl(m2_udp_sock, F_SETFL, O_NONBLOCK);
        m2_pc_addr.sin_family = AF_INET;
        m2_pc_addr.sin_port = htons(9999);
        inet_pton(AF_INET, "10.0.0.1", &m2_pc_addr.sin_addr); // ⚠️ 确保是你的 PC IP

        dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
            NSURL *url = [[NSBundle mainBundle] URLForResource:@"best" withExtension:@"mlmodelc"];
            if (url) {
                MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
                config.computeUnits = MLComputeUnitsAll;
                MLModel *ml = [MLModel modelWithContentsOfURL:url configuration:config error:nil];
                if (ml) {
                    m2_ai_model = [VNCoreMLModel modelForMLModel:ml error:nil];
                    m2_ai_request = [[VNCoreMLRequest alloc] initWithModel:m2_ai_model];
                    m2_ai_request.imageCropAndScaleOption = VNImageCropAndScaleOptionScaleFill;
                    printf("✅ [M2 AI] 神经引擎后台点火成功！\n");
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
                        int tx = (b.origin.x + b.size.width/2.0)*w;
                        int ty = (1.0-b.origin.y-b.size.height*0.8)*h_px;
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

static void m2_decompression_callback(void *refCon, void *sfRefCon, OSStatus status, VTDecodeInfoFlags info, CVImageBufferRef img, CMTime pts, CMTime dur) {
    if (status == noErr && img) {
        m2_run_ai(img);
    }
}

// ==========================================================
// 🏗️ Moonlight 官方原汁原味渲染逻辑 (修复 NALU 破坏问题)
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
    VTDecompressionSessionRef _m2Session; // AI 专属窃听会话
}

- (void)reinitializeDisplayLayer {
    CALayer *oldLayer = displayLayer;
    displayLayer = [[AVSampleBufferDisplayLayer alloc] init];
    displayLayer.backgroundColor = [UIColor blackColor].CGColor;
    CGSize videoSize;
    if (_view.bounds.size.width > _view.bounds.size.height * _streamAspectRatio) {
        videoSize = CGSizeMake(_view.bounds.size.height * _streamAspectRatio, _view.bounds.size.height);
    } else {
        videoSize = CGSizeMake(_view.bounds.size.width, _view.bounds.size.width / _streamAspectRatio);
    }
    displayLayer.position = CGPointMake(CGRectGetMidX(_view.bounds), CGRectGetMidY(_view.bounds));
    displayLayer.bounds = CGRectMake(0, 0, videoSize.width, videoSize.height);
    displayLayer.videoGravity = AVLayerVideoGravityResize;
    displayLayer.hidden = YES;
    if (oldLayer != nil) [_view.layer replaceSublayer:oldLayer with:displayLayer];
    else [_view.layer addSublayer:displayLayer];

    if (formatDesc != nil) { CFRelease(formatDesc); formatDesc = nil; }
    if (_m2Session != NULL) { VTDecompressionSessionInvalidate(_m2Session); CFRelease(_m2Session); _m2Session = NULL; }
}

- (id)initWithView:(StreamView*)view callbacks:(id<ConnectionCallbacks>)callbacks streamAspectRatio:(float)aspectRatio useFramePacing:(BOOL)useFramePacing {
    self = [super init];
    _view = view; _callbacks = callbacks; _streamAspectRatio = aspectRatio; framePacing = useFramePacing;
    parameterSetBuffers = [[NSMutableArray alloc] init];
    [self reinitializeDisplayLayer];
    m2_init_plugin(); // 提前初始化 AI
    return self;
}

- (void)setupWithVideoFormat:(int)vf width:(int)vw height:(int)vh frameRate:(int)fr {
    self->videoFormat = vf; self->frameRate = fr;
}

- (void)start {
    _displayLink = [CADisplayLink displayLinkWithTarget:self selector:@selector(displayLinkCallback:)];
    if (@available(iOS 15.0, tvOS 15.0, *)) {
        _displayLink.preferredFrameRateRange = CAFrameRateRangeMake(self->frameRate, self->frameRate, self->frameRate);
    } else {
        _displayLink.preferredFramesPerSecond = self->frameRate;
    }
    [_displayLink addToRunLoop:[NSRunLoop mainRunLoop] forMode:NSDefaultRunLoopMode];
}

int DrSubmitDecodeUnit(PDECODE_UNIT decodeUnit);

- (void)displayLinkCallback:(CADisplayLink *)sender {
    VIDEO_FRAME_HANDLE handle; PDECODE_UNIT du;
    while (LiPollNextVideoFrame(&handle, &du)) {
        LiCompleteVideoFrame(handle, DrSubmitDecodeUnit(du));
        if (framePacing) {
            double displayRefreshRate = 1 / (_displayLink.targetTimestamp - _displayLink.timestamp);
            if (displayRefreshRate >= frameRate * 0.9f) {
                if (LiGetPendingVideoFrames() == 1) break;
            }
        }
    }
}

- (void)stop {
    [_displayLink invalidate];
    if (_m2Session != NULL) { VTDecompressionSessionInvalidate(_m2Session); CFRelease(_m2Session); _m2Session = NULL; }
}

#define NALU_START_PREFIX_SIZE 3
#define NAL_LENGTH_PREFIX_SIZE 4

// 🛡️ 原汁原味的封装函数，不破坏任何 H.264 结构
- (void)updateAnnexBBufferForRange:(CMBlockBufferRef)frameBuffer dataBlock:(CMBlockBufferRef)dataBuffer offset:(int)offset length:(int)nalLength {
    OSStatus status;
    size_t oldOffset = CMBlockBufferGetDataLength(frameBuffer);
    status = CMBlockBufferAppendMemoryBlock(frameBuffer, NULL, NAL_LENGTH_PREFIX_SIZE, kCFAllocatorDefault, NULL, 0, NAL_LENGTH_PREFIX_SIZE, 0);
    if (status != noErr) return;
    const int dataLength = nalLength - NALU_START_PREFIX_SIZE;
    const uint8_t lengthBytes[] = {(uint8_t)(dataLength >> 24), (uint8_t)(dataLength >> 16), (uint8_t)(dataLength >> 8), (uint8_t)dataLength};
    status = CMBlockBufferReplaceDataBytes(lengthBytes, frameBuffer, oldOffset, NAL_LENGTH_PREFIX_SIZE);
    if (status != noErr) return;
    status = CMBlockBufferAppendBufferReference(frameBuffer, dataBuffer, offset + NALU_START_PREFIX_SIZE, dataLength, 0);
}

- (int)submitDecodeBuffer:(unsigned char *)data length:(int)length bufferType:(int)bufferType decodeUnit:(PDECODE_UNIT)du {
    OSStatus status;
    if (du->frameType == FRAME_TYPE_IDR) {
        if (bufferType != 4) { // BUFFER_TYPE_PICDATA
            if (bufferType >= 1 && bufferType <= 3) {
                int startLen = data[2] == 0x01 ? 3 : 4;
                [parameterSetBuffers addObject:[NSData dataWithBytes:&data[startLen] length:length - startLen]];
            }
            return 0; // DR_OK
        }
        if (formatDesc != NULL) { CFRelease(formatDesc); formatDesc = NULL; }
        if (_m2Session != NULL) { VTDecompressionSessionInvalidate(_m2Session); CFRelease(_m2Session); _m2Session = NULL; }
        
        size_t parameterSetCount = [parameterSetBuffers count];
        const uint8_t* parameterSetPointers[parameterSetCount];
        size_t parameterSetSizes[parameterSetCount];
        for (int i = 0; i < parameterSetCount; i++) {
            NSData* parameterSet = parameterSetBuffers[i];
            parameterSetPointers[i] = parameterSet.bytes; parameterSetSizes[i] = parameterSet.length;
        }
        
        if (videoFormat & 0x01) { // H264
            status = CMVideoFormatDescriptionCreateFromH264ParameterSets(kCFAllocatorDefault, parameterSetCount, parameterSetPointers, parameterSetSizes, NAL_LENGTH_PREFIX_SIZE, &formatDesc);
        } else if (videoFormat & 0x02) { // HEVC
            status = CMVideoFormatDescriptionCreateFromHEVCParameterSets(kCFAllocatorDefault, parameterSetCount, parameterSetPointers, parameterSetSizes, NAL_LENGTH_PREFIX_SIZE, NULL, &formatDesc);
        }
        [parameterSetBuffers removeAllObjects];
        
        // 🧠 为 AI 旁路创建专门的解码会话
        if (formatDesc) {
            VTDecompressionOutputCallbackRecord cb = {0};
            cb.decompressionOutputCallback = m2_decompression_callback;
            NSDictionary *attrs = @{ (id)kCVPixelBufferPixelFormatTypeKey: @(kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange) };
            VTDecompressionSessionCreate(kCFAllocatorDefault, formatDesc, NULL, (__bridge CFDictionaryRef)attrs, &cb, &_m2Session);
        }
    }
    
    if (formatDesc == NULL) { free(data); return 1; } // DR_NEED_IDR
    
    CMBlockBufferRef frameBlockBuffer; CMBlockBufferRef dataBlockBuffer;
    status = CMBlockBufferCreateWithMemoryBlock(NULL, data, length, kCFAllocatorDefault, NULL, 0, length, 0, &dataBlockBuffer);
    if (status != noErr) { free(data); return 1; }
    status = CMBlockBufferCreateEmpty(NULL, 0, 0, &frameBlockBuffer);
    
    // 🛡️ 使用最原始的转换算法，确保底层解码器不罢工
    int lastOffset = -1;
    for (int i = 0; i < length - NALU_START_PREFIX_SIZE; i++) {
        if (data[i] == 0 && data[i+1] == 0 && data[i+2] == 1) {
            if (lastOffset != -1) [self updateAnnexBBufferForRange:frameBlockBuffer dataBlock:dataBlockBuffer offset:lastOffset length:i - lastOffset];
            lastOffset = i;
        }
    }
    if (lastOffset != -1) [self updateAnnexBBufferForRange:frameBlockBuffer dataBlock:dataBlockBuffer offset:lastOffset length:length - lastOffset];
    
    CMSampleBufferRef sampleBuffer;
    CMSampleTimingInfo sampleTiming = {kCMTimeInvalid, CMTimeMake(du->presentationTimeMs, 1000), kCMTimeInvalid};
    status = CMSampleBufferCreateReady(kCFAllocatorDefault, frameBlockBuffer, formatDesc, 1, 1, &sampleTiming, 0, NULL, &sampleBuffer);
    
    if (status == noErr) {
        // 🚀 原版渲染：直接丢给系统层，画面秒出！
        [self->displayLayer enqueueSampleBuffer:sampleBuffer];
        
        // 🧠 AI 旁路：喂给隐藏解码器去解析坐标
        if (_m2Session) {
            VTDecompressionSessionDecodeFrame(_m2Session, sampleBuffer, kVTDecodeFrame_EnableAsynchronousDecompression, NULL, NULL);
        }
    }
    
    if (du->frameType == FRAME_TYPE_IDR) {
        self->displayLayer.hidden = NO;
        [self->_callbacks videoContentShown];
    }
    
    CFRelease(dataBlockBuffer); CFRelease(frameBlockBuffer);
    if (sampleBuffer) CFRelease(sampleBuffer);
    return 0; // DR_OK
}

- (void)setHdrMode:(BOOL)enabled {}
@end
