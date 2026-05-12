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
// 🚀 [终极力学轨道] M2 动态质心引力 + 网格密度透视引擎
// ==========================================================

// 升级结构体：加入 sum_x 和 sum_y 来累加所有红点，计算物理质心！
typedef struct {
    int min_x, max_x, min_y, max_y;
    int pixel_count; 
    long long sum_x; 
    long long sum_y; 
} TargetBlob;

static void m2_process_frame(CVImageBufferRef pixelBuffer) {
    if (!pixelBuffer) return;
    
    if (m2_udp_sock == -1) {
        m2_udp_sock = socket(AF_INET, SOCK_DGRAM, 0);
        fcntl(m2_udp_sock, F_SETFL, fcntl(m2_udp_sock, F_GETFL, 0) | O_NONBLOCK);
        m2_pc_addr.sin_family = AF_INET;
        m2_pc_addr.sin_port = htons(9999);
        inet_pton(AF_INET, "10.0.0.1", &m2_pc_addr.sin_addr); // ⚠️ 确保是电脑的专线 IP
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

        // 1. 🔍 第一层：磁力收集与【质量累加】 (步长缩至2，极速精细扫描)
        int step = 2; 
        for (int y = start_y; y < end_y; y += step) {
            uint8_t *yRow = yPlane + y * yBytesPerRow;
            uint8_t *uvRow = uvPlane + (y / 2) * uvBytesPerRow;
            
            for (int x = start_x; x < end_x; x += step) {
                if (abs(x - cx) < 30 && abs(y - cy) < 30) continue;

                uint8_t v = uvRow[(x / 2) * 2 + 1];
                
                // 🧬 基因锁微调：稍微放宽亮度(Y>25)，完美包容运动模糊产生的暗沉！
                if (v > 140) { 
                    uint8_t y_val = yRow[x];
                    uint8_t u = uvRow[(x / 2) * 2];
                    
                    // 💡 紫光抗性拉满：U 放宽到 <135，强行包容图3的紫色闪光，绝不跟丢！
                    if (y_val > 25 && u < 135) {
                        int added = 0;
                        for (int i = 0; i < blob_count; i++) {
                            // 🧲 磁力圈拉大到 55 像素！
                            // 就算假人在高速运动中手脚被残影拉断，也能像强力胶一样重新缝合成一个人！
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
        
        // 2. ⚖️ 第二层：网格密度透视法与【质心瞄准】
        int best_x = -1, best_y = -1;
        long min_dist = 2000000000;
        
        for (int i = 0; i < blob_count; i++) {
            int w = blobs[i].max_x - blobs[i].min_x;
            int h = blobs[i].max_y - blobs[i].min_y;
            
            // 🔪 锁一：物理尺寸包容
            if (w < 12 || h < 12 || w > 800 || h > 800) continue;
            
            // 🔪 锁二：狂野姿态解禁 (兼容极度拉伸的动作)
            // 将长宽比放宽到极端的 0.40 ~ 5.5！不管是贴地滑铲/飞踢，还是垂直站立，通杀！
            float aspect = (float)h / (float)w;
            if (aspect < 0.40f || aspect > 5.5f) continue;
            
            // 🔪 锁三：纯网格密度查杀 (The Wall Killer)
            float max_pixels = (float)(w / step + 1) * (float)(h / step + 1);
            float density = (float)blobs[i].pixel_count / max_pixels;
            
            float max_allowed_density = 0.55f; // 近战网格默认最高密度 55% (无情过滤红墙)
            
            // 远距容错：如果目标极小(图5)，视频压缩会让它糊成实心红斑！动态放宽密度至 95%
            if (w * h < 4000) {
                max_allowed_density = 0.95f; 
            }
            
            // 实心红砖墙、微小红光噪点，全部在此被判死刑！
            if (density < 0.02f || density > max_allowed_density) continue;

            // ==========================================
            // 🎯 目标确定，启动【物理质心牵引】！
            // ==========================================
            // 将所有红点的坐标求平均，得出躯干质量最密集的“绝对物理重心”
            int com_x = (int)(blobs[i].sum_x / blobs[i].pixel_count);
            int com_y = (int)(blobs[i].sum_y / blobs[i].pixel_count);
            
            int target_x = com_x;
            int target_y;
            
            // 根据姿态自适应锁胸点：
            if (aspect > 1.2f) {
                // 🧍‍♂️ 站立/奔跑状态：质心通常在肚子。我们将其往上提拔身高的 12%，精准锁定胸锁骨！
                target_y = com_y - (int)(h * 0.12f);
            } else {
                // 🛷 横飞/蹲伏状态：身体是横向的，质心本来就处于胸腹之间。原位锁定，枪枪到肉！
                target_y = com_y;
            }
            
            long dist = (target_x - cx)*(target_x - cx) + (target_y - cy)*(target_y - cy);
            if (dist < min_dist) {
                min_dist = dist;
                best_x = target_x;
                best_y = target_y;
            }
        }
        
        // 3. 将极其平滑的坐标极速发往 PC 雷达
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

// 后面的 @implementation VideoDecoderRenderer { ... } 及苹果官方源码部分完全保持不变，确保无缝接入。
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
    
    // 挂载旁路解码回调
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
    
    // 正常显示
    [self->displayLayer enqueueSampleBuffer:sampleBuffer];
    
    // 旁路解码
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

