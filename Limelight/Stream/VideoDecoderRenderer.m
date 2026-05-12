//
//  VideoDecoderRenderer.m
//  Moonlight
//
//  Created by Cameron Gutman on 10/18/14.
//  Copyright (c) 2014 Moonlight Stream. All rights reserved.
//

#import "VideoDecoderRenderer.h"
#import "StreamView.h"

#include <libavcodec/avcodec.h>
#include <libavcodec/cbs.h>
#include <libavcodec/cbs_av1.h>
#include <libavformat/avio.h>
#include <libavutil/mem.h>

#import <VideoToolbox/VideoToolbox.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/time.h>
#include <math.h>
#include <fcntl.h>
#include <stdlib.h>

// ==========================================================
// 🚀 [终极力学轨道] M2 动态质心引力 + 纯网格密度引擎 (完美专线版)
// ==========================================================

static int m2_udp_sock = -1;
static struct sockaddr_in m2_pc_addr;

// 存储包围盒、像素数量，以及用于计算“物理质心”的坐标总和
typedef struct {
    int min_x, max_x, min_y, max_y;
    int pixel_count; 
    long long sum_x; // 累加所有像素的 X 坐标，用于求质心
    long long sum_y; // 累加所有像素的 Y 坐标，用于求质心
} TargetBlob;

static void m2_process_frame(CVImageBufferRef pixelBuffer) {
    if (!pixelBuffer) return;
    
    if (m2_udp_sock == -1) {
        m2_udp_sock = socket(AF_INET, SOCK_DGRAM, 0);
        fcntl(m2_udp_sock, F_SETFL, fcntl(m2_udp_sock, F_GETFL, 0) | O_NONBLOCK);
        m2_pc_addr.sin_family = AF_INET;
        m2_pc_addr.sin_port = htons(9999);
        
        // 🚨 终极专线：确保这是你电脑的有线专线 IP！
        inet_pton(AF_INET, "10.0.0.1", &m2_pc_addr.sin_addr); 
    }

    CVPixelBufferLockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
    if (CVPixelBufferGetPlaneCount(pixelBuffer) >= 2) {
        int width = (int)CVPixelBufferGetWidth(pixelBuffer);
        int height = (int)CVPixelBufferGetHeight(pixelBuffer);
        
        uint8_t *yPlane = (uint8_t *)CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0);
        size_t yBytesPerRow = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 0);
        uint8_t *uvPlane = (uint8_t *)CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 1);
        size_t uvBytesPerRow = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 1);
        
        int cx = width / 2;
        int cy = height / 2;
        
        TargetBlob blobs[32];
        int blob_count = 0;

        int start_y = (int)(height * 0.1); int end_y = (int)(height * 0.9);
        int start_x = (int)(width * 0.1);  int end_x = (int)(width * 0.9);

        // 1. 🔍 第一层：磁力收集与【质量累加】 (步长2，极速穿插网格条纹)
        int step = 2; 
        for (int y = start_y; y < end_y; y += step) {
            uint8_t *yRow = yPlane + y * yBytesPerRow;
            uint8_t *uvRow = uvPlane + (y / 2) * uvBytesPerRow;
            
            for (int x = start_x; x < end_x; x += step) {
                // 屏蔽准星的自干扰
                if (abs(x - cx) < 30 && abs(y - cy) < 30) continue;

                uint8_t v = uvRow[(x / 2) * 2 + 1];
                
                // 🎯 完美基因锁：基于你提取的8组色彩样本严密推导
                if (v > 145) { 
                    uint8_t y_val = yRow[x];
                    uint8_t u = uvRow[(x / 2) * 2];
                    
                    if (y_val > 30 && u < 125) {
                        int added = 0;
                        for (int i = 0; i < blob_count; i++) {
                            // 🧲 磁力圈：55像素的强力融合，无视肢体在高速运动中产生残影断裂
                            if (x >= blobs[i].min_x - 55 && x <= blobs[i].max_x + 55 &&
                                y >= blobs[i].min_y - 55 && y <= blobs[i].max_y + 55) {
                                
                                if (x < blobs[i].min_x) blobs[i].min_x = x;
                                if (x > blobs[i].max_x) blobs[i].max_x = x;
                                if (y < blobs[i].min_y) blobs[i].min_y = y;
                                if (y > blobs[i].max_y) blobs[i].max_y = y;
                                
                                blobs[i].pixel_count++;
                                blobs[i].sum_x += x; // 💀 叠加 X 质量
                                blobs[i].sum_y += y; // 💀 叠加 Y 质量
                                added = 1;
                                break;
                            }
                        }
                        if (!added && blob_count < 32) {
                            blobs[blob_count].min_x = x; blobs[blob_count].max_x = x;
                            blobs[blob_count].min_y = y; blobs[blob_count].max_y = y;
                            blobs[blob_count].pixel_count = 1;
                            blobs[blob_count].sum_x = x;
                            blobs[blob_count].sum_y = y;
                            blob_count++;
                        }
                    }
                }
            }
        }
        
        // 2. ⚖️ 第二层：网格透视法与【质心瞄准】
        int best_x = -1, best_y = -1;
        long min_dist = 2000000000;
        
        for (int i = 0; i < blob_count; i++) {
            int w = blobs[i].max_x - blobs[i].min_x;
            int h = blobs[i].max_y - blobs[i].min_y;
            
            // 🔪 锁一：物理尺寸包容 (防单点小火花干扰)
            if (w < 12 || h < 12 || w > 800 || h > 800) continue;
            
            // 🔪 锁二：狂野姿态解禁 (0.35 ~ 6.0 包容滑铲飞踢、高空下落拉伸)
            float aspect = (float)h / (float)w;
            if (aspect < 0.35f || aspect > 6.0f) continue;
            
            // 🔪 锁三：纯网格密度查杀 (无情秒杀实心红墙、红箱子)
            float max_pixels = (float)(w / step + 1) * (float)(h / step + 1);
            float density = (float)blobs[i].pixel_count / max_pixels;
            
            float max_allowed_density = 0.60f; // 正常网格最高 60% 密度
            
            // 目标越远，马赛克越重，越容易糊成实心块。微小目标放宽密度上限到 95%
            if (w * h < 4000) {
                max_allowed_density = 0.95f; 
            }
            if (density < 0.02f || density > max_allowed_density) continue;

            // ==========================================
            // 🎯 目标确定，启动【物理质心牵引】！
            // ==========================================
            // 将框内成千上万个红点坐标求平均，得出躯干最密集的绝对物理重心！
            // 彻底解决挥手、抬枪导致准星猛跳的终极问题！
            int com_x = (int)(blobs[i].sum_x / blobs[i].pixel_count);
            int com_y = (int)(blobs[i].sum_y / blobs[i].pixel_count);
            
            int target_x = com_x;
            int target_y;
            
            if (aspect > 1.2f) {
                // 🧍‍♂️ 站立跑动姿态：质心位于肚子，向上提拔身高的 12%，精准死锁胸膛
                target_y = com_y - (int)(h * 0.12f);
            } else {
                // 🛷 滑铲横飞姿态：躯干横摆，质心直接暴露在胸腹核心，无需拉升原位锁死
                target_y = com_y;
            }
            
            long dist = (target_x - cx)*(target_x - cx) + (target_y - cy)*(target_y - cy);
            if (dist < min_dist) {
                min_dist = dist;
                best_x = target_x;
                best_y = target_y;
            }
        }
        
        // 3. 将极其平滑的质心坐标，极速顺着网线发往 PC 雷达
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
    CVPixelBufferUnlockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
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

// 后续 Moonlight 官方接口完美兼容部分
extern int ff_isom_write_av1c(AVIOContext *pb, const uint8_t *buf, int size,
                              int write_seq_header);

@implementation VideoDecoderRenderer {
    StreamView* _view;
    id<ConnectionCallbacks> _callbacks;
    float _streamAspectRatio;
    
    AVSampleBufferDisplayLayer* displayLayer;
    int videoFormat;
    int frameRate;
    
    NSMutableArray *parameterSetBuffers;
    NSData *masteringDisplayColorVolume;
    NSData *contentLightLevelInfo;
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
    
    if (oldLayer != nil) {
        [_view.layer replaceSublayer:oldLayer with:displayLayer];
    } else {
        [_view.layer addSublayer:displayLayer];
    }
    
    if (formatDesc != nil) {
        CFRelease(formatDesc);
        formatDesc = nil;
    }
    
    if (_m2Session != NULL) {
        VTDecompressionSessionInvalidate(_m2Session);
        CFRelease(_m2Session);
        _m2Session = NULL;
    }
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
    if (@available(iOS 15.0, tvOS 15.0, *)) {
        _displayLink.preferredFrameRateRange = CAFrameRateRangeMake(self->frameRate, self->frameRate, self->frameRate);
    } else {
        _displayLink.preferredFramesPerSecond = self->frameRate;
    }
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
            if (displayRefreshRate >= frameRate * 0.9f) {
                if (LiGetPendingVideoFrames() == 1) break;
            }
        }
    }
}

- (void)stop {
    [_displayLink invalidate];
    if (_m2Session != NULL) {
        VTDecompressionSessionInvalidate(_m2Session);
        CFRelease(_m2Session);
        _m2Session = NULL;
    }
}

#define NALU_START_PREFIX_SIZE 3
#define NAL_LENGTH_PREFIX_SIZE 4

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

- (NSData*)getAv1CodecConfigurationBox:(NSData*)frameData {
    AVIOContext* ioctx = NULL; int err = avio_open_dyn_buf(&ioctx);
    if (err < 0) return nil;
    err = ff_isom_write_av1c(ioctx, (uint8_t*)frameData.bytes, (int)frameData.length, 1);
    uint8_t* av1cBuf = NULL; int av1cBufLen = avio_close_dyn_buf(ioctx, &av1cBuf);
    NSData* data = (err >= 0 && av1cBufLen > 0) ? [NSData dataWithBytes:av1cBuf length:av1cBufLen] : nil;
    av_free(av1cBuf); return data;
}

- (CMVideoFormatDescriptionRef)createAV1FormatDescriptionForIDRFrame:(NSData*)frameData {
    NSMutableDictionary* extensions = [[NSMutableDictionary alloc] init];
    CodedBitstreamContext* cbsCtx = NULL;
    int err = ff_cbs_init(&cbsCtx, AV_CODEC_ID_AV1, NULL);
    if (err < 0) return nil;
    AVPacket avPacket = {}; avPacket.data = (uint8_t*)frameData.bytes; avPacket.size = (int)frameData.length;
    CodedBitstreamFragment cbsFrag = {};
    err = ff_cbs_read_packet(cbsCtx, &cbsFrag, &avPacket);
    if (err < 0) { ff_cbs_close(&cbsCtx); return nil; }
#define SET_CFSTR_EXTENSION(key, value) extensions[(__bridge NSString*)key] = (__bridge NSString*)(value)
#define SET_EXTENSION(key, value) extensions[(__bridge NSString*)key] = (value)
    SET_EXTENSION(kCMFormatDescriptionExtension_FormatName, @"av01");
    SET_EXTENSION(kCMFormatDescriptionExtension_Depth, @24);
    CodedBitstreamAV1Context* bitstreamCtx = (CodedBitstreamAV1Context*)cbsCtx->priv_data;
    AV1RawSequenceHeader* seqHeader = bitstreamCtx->sequence_header;
    if (seqHeader == NULL) { ff_cbs_fragment_free(&cbsFrag); ff_cbs_close(&cbsCtx); return nil; }
    SET_EXTENSION(kCMFormatDescriptionExtension_FullRangeVideo, @(seqHeader->color_config.color_range == 1));
    SET_EXTENSION(kCMFormatDescriptionExtension_FieldCount, @(1));
    extensions[(__bridge NSString*)kCMFormatDescriptionExtension_SampleDescriptionExtensionAtoms] = @{ @"av1C" : [self getAv1CodecConfigurationBox:frameData] };
    extensions[@"BitsPerComponent"] = @(bitstreamCtx->bit_depth);
#undef SET_EXTENSION
#undef SET_CFSTR_EXTENSION
    CMVideoFormatDescriptionRef formatDesc = NULL;
    OSStatus status = CMVideoFormatDescriptionCreate(kCFAllocatorDefault, kCMVideoCodecType_AV1, bitstreamCtx->frame_width, bitstreamCtx->frame_height, (__bridge CFDictionaryRef)extensions, &formatDesc);
    if (status != noErr) formatDesc = NULL;
    ff_cbs_fragment_free(&cbsFrag); ff_cbs_close(&cbsCtx);
    return formatDesc;
}

- (int)submitDecodeBuffer:(unsigned char *)data length:(int)length bufferType:(int)bufferType decodeUnit:(PDECODE_UNIT)du {
    OSStatus status;
    if (du->frameType == FRAME_TYPE_IDR) {
        if (bufferType != BUFFER_TYPE_PICDATA) {
            if (bufferType == BUFFER_TYPE_VPS || bufferType == BUFFER_TYPE_SPS || bufferType == BUFFER_TYPE_PPS) {
                int startLen = data[2] == 0x01 ? 3 : 4;
                [parameterSetBuffers addObject:[NSData dataWithBytes:&data[startLen] length:length - startLen]];
            }
            return DR_OK;
        }
        if (formatDesc != NULL) { CFRelease(formatDesc); formatDesc = NULL; }
        if (_m2Session != NULL) { VTDecompressionSessionInvalidate(_m2Session); CFRelease(_m2Session); _m2Session = NULL; }
        if (videoFormat & VIDEO_FORMAT_MASK_H264) {
            size_t parameterSetCount = [parameterSetBuffers count];
            const uint8_t* parameterSetPointers[parameterSetCount];
            size_t parameterSetSizes[parameterSetCount];
            for (int i = 0; i < parameterSetCount; i++) {
                NSData* parameterSet = parameterSetBuffers[i];
                parameterSetPointers[i] = parameterSet.bytes; parameterSetSizes[i] = parameterSet.length;
            }
            status = CMVideoFormatDescriptionCreateFromH264ParameterSets(kCFAllocatorDefault, parameterSetCount, parameterSetPointers, parameterSetSizes, NAL_LENGTH_PREFIX_SIZE, &formatDesc);
            [parameterSetBuffers removeAllObjects];
        }
        else if (videoFormat & VIDEO_FORMAT_MASK_H265) {
            size_t parameterSetCount = [parameterSetBuffers count];
            const uint8_t* parameterSetPointers[parameterSetCount];
            size_t parameterSetSizes[parameterSetCount];
            for (int i = 0; i < parameterSetCount; i++) {
                NSData* parameterSet = parameterSetBuffers[i];
                parameterSetPointers[i] = parameterSet.bytes; parameterSetSizes[i] = parameterSet.length;
            }
            NSMutableDictionary* videoFormatParams = [[NSMutableDictionary alloc] init];
            status = CMVideoFormatDescriptionCreateFromHEVCParameterSets(kCFAllocatorDefault, parameterSetCount, parameterSetPointers, parameterSetSizes, NAL_LENGTH_PREFIX_SIZE, (__bridge CFDictionaryRef)videoFormatParams, &formatDesc);
            [parameterSetBuffers removeAllObjects];
        }
        else if (videoFormat & VIDEO_FORMAT_MASK_AV1) {
            NSData* fullFrameData = [NSData dataWithBytesNoCopy:data length:length freeWhenDone:NO];
            formatDesc = [self createAV1FormatDescriptionForIDRFrame:fullFrameData];
        }
    }
    if (formatDesc == NULL) { free(data); return DR_NEED_IDR; }
    
    // 挂载 M2 硬件旁路窃听
    if (_m2Session == NULL) {
        VTDecompressionOutputCallbackRecord cb = {0};
        cb.decompressionOutputCallback = m2_decompression_callback;
        NSDictionary *attrs = @{ (id)kCVPixelBufferPixelFormatTypeKey: @(kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange) };
        VTDecompressionSessionCreate(kCFAllocatorDefault, formatDesc, NULL, (__bridge CFDictionaryRef)attrs, &cb, &_m2Session);
    }
    
    if (displayLayer.status == AVQueuedSampleBufferRenderingStatusFailed) {
        [self reinitializeDisplayLayer]; free(data); return DR_NEED_IDR;
    }
    CMBlockBufferRef frameBlockBuffer; CMBlockBufferRef dataBlockBuffer;
    status = CMBlockBufferCreateWithMemoryBlock(NULL, data, length, kCFAllocatorDefault, NULL, 0, length, 0, &dataBlockBuffer);
    if (status != noErr) { free(data); return DR_NEED_IDR; }
    status = CMBlockBufferCreateEmpty(NULL, 0, 0, &frameBlockBuffer);
    
    if (videoFormat & (VIDEO_FORMAT_MASK_H264 | VIDEO_FORMAT_MASK_H265)) {
        int lastOffset = -1;
        for (int i = 0; i < length - NALU_START_PREFIX_SIZE; i++) {
            if (data[i] == 0 && data[i+1] == 0 && data[i+2] == 1) {
                if (lastOffset != -1) [self updateAnnexBBufferForRange:frameBlockBuffer dataBlock:dataBlockBuffer offset:lastOffset length:i - lastOffset];
                lastOffset = i;
            }
        }
        if (lastOffset != -1) [self updateAnnexBBufferForRange:frameBlockBuffer dataBlock:dataBlockBuffer offset:lastOffset length:length - lastOffset];
    } else {
        CMBlockBufferAppendBufferReference(frameBlockBuffer, dataBlockBuffer, 0, length, 0);
    }
        
    CMSampleBufferRef sampleBuffer;
    CMSampleTimingInfo sampleTiming = {kCMTimeInvalid, CMTimeMake(du->presentationTimeMs, 1000), kCMTimeInvalid};
    status = CMSampleBufferCreateReady(kCFAllocatorDefault, frameBlockBuffer, formatDesc, 1, 1, &sampleTiming, 0, NULL, &sampleBuffer);
    
    // 原版正常显示
    [self->displayLayer enqueueSampleBuffer:sampleBuffer];
    
    // 注入雷达解码执行暗杀锁定
    if (_m2Session) {
        VTDecompressionSessionDecodeFrame(_m2Session, sampleBuffer, kVTDecodeFrame_EnableAsynchronousDecompression, NULL, NULL);
    }
    
    if (du->frameType == FRAME_TYPE_IDR) {
        self->displayLayer.hidden = NO;
        [self->_callbacks videoContentShown];
    }
    CFRelease(dataBlockBuffer); CFRelease(frameBlockBuffer); CFRelease(sampleBuffer);
    return DR_OK;
}

- (void)setHdrMode:(BOOL)enabled {}
@end

