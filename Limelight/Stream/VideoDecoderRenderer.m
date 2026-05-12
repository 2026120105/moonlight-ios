#import "VideoDecoderRenderer.h"
#import "StreamView.h"

#include <libavcodec/avcodec.h>
#include <libavcodec/cbs.h>
#include <libavcodec/cbs_av1.h>
#include <libavformat/avio.h>
#include <libavutil/mem.h>

#import <VideoToolbox/VideoToolbox.h>
// 🚨 核心：引入苹果视觉与机器学习框架
#import <Vision/Vision.h>
#import <CoreML/CoreML.h>

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <fcntl.h>

// ==========================================================
// 🧠 [M2 ANE 神经引擎] YOLOv8 语义视觉锁定单元
// ==========================================================

static int m2_udp_sock = -1;
static struct sockaddr_in m2_pc_addr;

// AI 神经网络全局缓存（保证只加载一次，绝对省电且响应极快）
static VNCoreMLModel *m2_ai_model = nil;
static VNCoreMLRequest *m2_ai_request = nil;
static BOOL ai_init_attempted = NO;

// 初始化 AI 引擎：在 App 内部搜索你通过 GitHub 云端注入的“大脑”
static void m2_init_ai_engine(void) {
    if (m2_ai_model || ai_init_attempted) return; 
    ai_init_attempted = YES;
    
    // 初始化物理专线：向 PC 发送雷达坐标
    m2_udp_sock = socket(AF_INET, SOCK_DGRAM, 0);
    fcntl(m2_udp_sock, F_SETFL, fcntl(m2_udp_sock, F_GETFL, 0) | O_NONBLOCK);
    m2_pc_addr.sin_family = AF_INET;
    m2_pc_addr.sin_port = htons(9999);
    // ⚠️ 依然走你的 0.1ms 直连专线 IP
    inet_pton(AF_INET, "10.0.0.1", &m2_pc_addr.sin_addr); 

    // 🔍 寻找编译后的二进制模型文件
    NSURL *modelURL = [[NSBundle mainBundle] URLForResource:@"best" withExtension:@"mlmodelc"];
    if (!modelURL) {
        printf("❌ [M2 AI] 严重警告：未能在 App 内找到 best.mlmodelc！请检查注入流程。\n");
        return;
    }
    
    // 🚀 强制拉响 M2 芯片里的 ANE 硬件加速警报
    MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
    config.computeUnits = MLComputeUnitsAll; 
    
    NSError *err = nil;
    MLModel *mlModel = [MLModel modelWithContentsOfURL:modelURL configuration:config error:&err];
    if (mlModel) {
        m2_ai_model = [VNCoreMLModel modelForMLModel:mlModel error:&err];
        m2_ai_request = [[VNCoreMLRequest alloc] initWithModel:m2_ai_model];
        // 缩放模式：强制匹配训练时的 416x416 视野
        m2_ai_request.imageCropAndScaleOption = VNImageCropAndScaleOptionScaleFill;
        printf("✅ [M2 AI] ANE 神经网络引擎点火成功！进入全息语义视觉模式！\n");
    }
}

static void m2_process_frame(CVImageBufferRef pixelBuffer) {
    if (!pixelBuffer) return;
    m2_init_ai_engine();
    
    if (!m2_ai_request) return;

    int width = (int)CVPixelBufferGetWidth(pixelBuffer);
    int height = (int)CVPixelBufferGetHeight(pixelBuffer);
    int cx = width / 2;
    int cy = height / 2;

    // ⚡️ 核心引爆：直接将底层视频流 CVPixelBuffer 喂给神经网络
    VNImageRequestHandler *handler = [[VNImageRequestHandler alloc] initWithCVPixelBuffer:pixelBuffer options:@{}];
    [handler performRequests:@[m2_ai_request] error:nil];

    int best_x = -1, best_y = -1;
    long min_dist = 2000000000;

    // 👁️ 解析 AI 处理结果
    if (m2_ai_request.results) {
        for (VNRecognizedObjectObservation *obs in m2_ai_request.results) {
            
            // 🎯 置信度阈值：只要 AI 觉得有 45% 的把握是目标就开锁
            if (obs.confidence > 0.45f) {
                
                // ⚠️ Vision 坐标系翻转：从左下角映射到屏幕坐标系
                CGRect bbox = obs.boundingBox;
                int box_x = (int)(bbox.origin.x * width);
                int box_w = (int)(bbox.size.width * width);
                int box_h = (int)(bbox.size.height * height);
                // 坐标翻转公式：(1.0 - y - height)
                int box_y = (int)((1.0 - bbox.origin.y - bbox.size.height) * height);
                
                // 💀 精准死锁：取宽度中点，高度向下压 20%（胸口绝对中心）
                int target_x = box_x + box_w / 2;
                int target_y = box_y + (int)(box_h * 0.20f);
                
                // 距离准星最近者为优先猎杀目标
                long dist = (target_x - cx)*(target_x - cx) + (target_y - cy)*(target_y - cy);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_x = target_x;
                    best_y = target_y;
                }
            }
        }
    }

    // 🚀 发送坐标数据包给 PC
    if (best_x != -1) {
        float dx = best_x - cx; float dy = best_y - cy;
        char msg[64];
        snprintf(msg, sizeof(msg), "{\"f\":1,\"dx\":%.1f,\"dy\":%.1f}", dx, dy);
        sendto(m2_udp_sock, msg, strlen(msg), 0, (struct sockaddr *)&m2_pc_addr, sizeof(m2_pc_addr));
    } else {
        char msg[] = "{\"f\":0}";
        sendto(m2_udp_sock, msg, strlen(msg), 0, (struct sockaddr *)&m2_pc_addr, sizeof(m2_pc_addr));
    }
}

static void m2_decompression_callback(
    void * CM_NULLABLE decompressionOutputRefCon,
    void * CM_NULLABLE sourceFrameRefCon,
    OSStatus status,
    VTDecodeInfoFlags infoFlags,
    CM_NULLABLE CVImageBufferRef imageBuffer,
    CMTime presentationTimeStamp,
    CMTime presentationDuration )
{
    if (status == noErr && imageBuffer) {
        m2_process_frame(imageBuffer);
    }
}

// ==========================================================
// 🏗️ Moonlight 官方接口兼容实现部分 (请保持原样)
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
    VTDecompressionSessionRef _m2Session;
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
    return self;
}

- (void)setupWithVideoFormat:(int)videoFormat width:(int)videoWidth height:(int)videoHeight frameRate:(int)frameRate {
    self->videoFormat = videoFormat; self->frameRate = frameRate;
}

- (void)start {
    _displayLink = [CADisplayLink displayLinkWithTarget:self selector:@selector(displayLinkCallback:)];
    if (@available(iOS 15.0, tvOS 15.0, *)) _displayLink.preferredFrameRateRange = CAFrameRateRangeMake(self->frameRate, self->frameRate, self->frameRate);
    else _displayLink.preferredFramesPerSecond = self->frameRate;
    [_displayLink addToRunLoop:[NSRunLoop mainRunLoop] forMode:NSDefaultRunLoopMode];
}

int DrSubmitDecodeUnit(PDECODE_UNIT decodeUnit);

- (void)displayLinkCallback:(CADisplayLink *)sender {
    VIDEO_FRAME_HANDLE handle;
    PDECODE_UNIT du;
    while (LiPollNextVideoFrame(&handle, &du)) {
        LiCompleteVideoFrame(handle, DrSubmitDecodeUnit(du));
        if (framePacing) {
            double displayRefreshRate = 1 / (_displayLink.targetTimestamp - _displayLink.timestamp);
            if (displayRefreshRate >= frameRate * 0.9f) { if (LiGetPendingVideoFrames() == 1) break; }
        }
    }
}

- (void)stop {
    [_displayLink invalidate];
    if (_m2Session != NULL) { VTDecompressionSessionInvalidate(_m2Session); CFRelease(_m2Session); _m2Session = NULL; }
}

- (void)updateAnnexBBufferForRange:(CMBlockBufferRef)frameBuffer dataBlock:(CMBlockBufferRef)dataBuffer offset:(int)offset length:(int)nalLength {
    OSStatus status;
    size_t oldOffset = CMBlockBufferGetDataLength(frameBuffer);
    status = CMBlockBufferAppendMemoryBlock(frameBuffer, NULL, 4, kCFAllocatorDefault, NULL, 0, 4, 0);
    if (status != noErr) return;
    const int dataLength = nalLength - 3;
    const uint8_t lengthBytes[] = {(uint8_t)(dataLength >> 24), (uint8_t)(dataLength >> 16), (uint8_t)(dataLength >> 8), (uint8_t)dataLength};
    status = CMBlockBufferReplaceDataBytes(lengthBytes, frameBuffer, oldOffset, 4);
    if (status != noErr) return;
    status = CMBlockBufferAppendBufferReference(frameBuffer, dataBuffer, offset + 3, dataLength, 0);
}

- (int)submitDecodeBuffer:(unsigned char *)data length:(int)length bufferType:(int)bufferType decodeUnit:(PDECODE_UNIT)du {
    OSStatus status;
    if (du->frameType == FRAME_TYPE_IDR) {
        if (bufferType != 4) { // BUFFER_TYPE_PICDATA
            if (bufferType == 1 || bufferType == 2 || bufferType == 3) [parameterSetBuffers addObject:[NSData dataWithBytes:&data[data[2]==0x01?3:4] length:length-(data[2]==0x01?3:4)]];
            return 0; // DR_OK
        }
        if (formatDesc != NULL) { CFRelease(formatDesc); formatDesc = nil; }
        if (_m2Session != NULL) { VTDecompressionSessionInvalidate(_m2Session); CFRelease(_m2Session); _m2Session = NULL; }
        size_t psCount = [parameterSetBuffers count];
        const uint8_t* psPointers[psCount]; size_t psSizes[psCount];
        for (int i=0; i<psCount; i++) { NSData* ps = parameterSetBuffers[i]; psPointers[i] = ps.bytes; psSizes[i] = ps.length; }
        if (videoFormat & 0x01) CMVideoFormatDescriptionCreateFromH264ParameterSets(kCFAllocatorDefault, psCount, psPointers, psSizes, 4, &formatDesc);
        else if (videoFormat & 0x02) CMVideoFormatDescriptionCreateFromHEVCParameterSets(kCFAllocatorDefault, psCount, psPointers, psSizes, 4, NULL, &formatDesc);
        [parameterSetBuffers removeAllObjects];
    }
    if (formatDesc == NULL) { free(data); return 1; } // DR_NEED_IDR
    
    // 🛠️ M2 硬件解压会话绑定：将 AI 回调挂载到硬件流上
    if (_m2Session == NULL) {
        VTDecompressionOutputCallbackRecord cb = {0};
        cb.decompressionOutputCallback = m2_decompression_callback;
        NSDictionary *attrs = @{ (id)kCVPixelBufferPixelFormatTypeKey: @(kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange) };
        VTDecompressionSessionCreate(kCFAllocatorDefault, formatDesc, NULL, (__bridge CFDictionaryRef)attrs, &cb, &_m2Session);
    }
    
    CMBlockBufferRef fbb; CMBlockBufferRef dbb;
    CMBlockBufferCreateWithMemoryBlock(NULL, data, length, kCFAllocatorDefault, NULL, 0, length, 0, &dbb);
    CMBlockBufferCreateEmpty(NULL, 0, 0, &fbb);
    int lastOff = -1;
    for (int i=0; i<length-3; i++) {
        if (data[i]==0 && data[i+1]==0 && data[i+2]==1) {
            if (lastOff != -1) [self updateAnnexBBufferForRange:fbb dataBlock:dbb offset:lastOff length:i-lastOff];
            lastOff = i;
        }
    }
    if (lastOff != -1) [self updateAnnexBBufferForRange:fbb dataBlock:dbb offset:lastOff length:length-lastOff];
    
    CMSampleBufferRef sb;
    CMSampleTimingInfo timing = {kCMTimeInvalid, CMTimeMake(du->presentationTimeMs, 1000), kCMTimeInvalid};
    CMSampleBufferCreateReady(kCFAllocatorDefault, fbb, formatDesc, 1, 1, &timing, 0, NULL, &sb);
    
    [self->displayLayer enqueueSampleBuffer:sb];
    
    // 🧠 触发 AI 神经处理
    if (_m2Session) VTDecompressionSessionDecodeFrame(_m2Session, sb, kVTDecodeFrame_EnableAsynchronousDecompression, NULL, NULL);
    
    if (du->frameType == FRAME_TYPE_IDR) { self->displayLayer.hidden = NO; [self->_callbacks videoContentShown]; }
    CFRelease(dbb); CFRelease(fbb); CFRelease(sb);
    return 0; // DR_OK
}

- (void)setHdrMode:(BOOL)enabled {}
@end
