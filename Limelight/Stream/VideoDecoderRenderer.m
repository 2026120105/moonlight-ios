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
// 🚀 [终极轨道] M2 AABB 2D包围盒 + 切角拓扑双材质光谱引擎
// ==========================================================

static int m2_udp_sock = -1;
static struct sockaddr_in m2_pc_addr;

typedef struct {
    int min_x, max_x, min_y, max_y;
    int pixel_count; 
} TargetBlob;

static void m2_process_frame(CVImageBufferRef pixelBuffer) {
    if (!pixelBuffer) return;
    
    if (m2_udp_sock == -1) {
        m2_udp_sock = socket(AF_INET, SOCK_DGRAM, 0);
        fcntl(m2_udp_sock, F_SETFL, fcntl(m2_udp_sock, F_GETFL, 0) | O_NONBLOCK);
        m2_pc_addr.sin_family = AF_INET;
        m2_pc_addr.sin_port = htons(9999);
        
        // 🚨 终极专线：死死绑定你电脑的以太网物理 IP！
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

        // 1. 🔍 第一层：粗略网格磁力聚类 (快速圈出所有红光可疑物)
        int step = 3; 
        for (int y = start_y; y < end_y; y += step) {
            uint8_t *yRow = yPlane + y * yBytesPerRow;
            uint8_t *uvRow = uvPlane + (y / 2) * uvBytesPerRow;
            
            for (int x = start_x; x < end_x; x += step) {
                if (abs(x - cx) < 30 && abs(y - cy) < 30) continue;

                uint8_t v = uvRow[(x / 2) * 2 + 1];
                
                // 基因锁：红/紫 V通道狂暴极值
                if (v > 150) { 
                    uint8_t y_val = yRow[x];
                    uint8_t u = uvRow[(x / 2) * 2];
                    
                    if (y_val > 35 && u < 115) {
                        int added = 0;
                        for (int i = 0; i < blob_count; i++) {
                            // 磁力吸附半径：45像素，强行缝合被打碎的假人网格
                            if (x >= blobs[i].min_x - 45 && x <= blobs[i].max_x + 45 &&
                                y >= blobs[i].min_y - 45 && y <= blobs[i].max_y + 45) {
                                
                                if (x < blobs[i].min_x) blobs[i].min_x = x;
                                if (x > blobs[i].max_x) blobs[i].max_x = x;
                                if (y < blobs[i].min_y) blobs[i].min_y = y;
                                if (y > blobs[i].max_y) blobs[i].max_y = y;
                                blobs[i].pixel_count++;
                                added = 1;
                                break;
                            }
                        }
                        if (!added && blob_count < 32) {
                            blobs[blob_count].min_x = x; blobs[blob_count].max_x = x;
                            blobs[blob_count].min_y = y; blobs[blob_count].max_y = y;
                            blobs[blob_count].pixel_count = 1;
                            blob_count++;
                        }
                    }
                }
            }
        }
        
        // 2. ⚖️ 第二层：开启显微镜 - 切角拓扑 + 双材质光谱过滤
        int best_x = -1, best_y = -1;
        long min_dist = 2000000000;
        
        for (int i = 0; i < blob_count; i++) {
            int w = blobs[i].max_x - blobs[i].min_x;
            int h = blobs[i].max_y - blobs[i].min_y;
            
            // 🔪 锁一：物理大小与长宽比限制
            if (w < 10 || h < 15 || w > 600 || h > 900) continue;
            float aspect = (float)h / (float)w;
            if (aspect < 0.85f || aspect > 4.5f) continue;
            
            int head_y_end = blobs[i].min_y + (int)(h * 0.20f); 
            int head_min_x = 9999, head_max_x = -1;
            int torso_min_x = 9999, torso_max_x = -1;
            
            int corner_violation = 0;
            int corner_w = (int)(w * 0.25f);
            
            int bright_flesh_count = 0;
            int total_pixels = 0;
            
            // 深度精密扫描盒子内部特征
            for (int py = blobs[i].min_y; py <= blobs[i].max_y; py += step) {
                uint8_t *yRow = yPlane + py * yBytesPerRow;
                uint8_t *uvRow = uvPlane + (py / 2) * uvBytesPerRow;
                
                for (int px = blobs[i].min_x; px <= blobs[i].max_x; px += step) {
                    uint8_t v = uvRow[(px / 2) * 2 + 1];
                    uint8_t y_val = yRow[px];
                    uint8_t u = uvRow[(px / 2) * 2];
                    
                    total_pixels++;
                    
                    if (v > 150 && y_val > 35 && u < 115) {
                        if (py < head_y_end) {
                            if (px < head_min_x) head_min_x = px;
                            if (px > head_max_x) head_max_x = px;
                            
                            // 肩部上空探测：防矩形红墙
                            if (px < blobs[i].min_x + corner_w || px > blobs[i].max_x - corner_w) {
                                corner_violation++;
                            }
                        } else {
                            if (px < torso_min_x) torso_min_x = px;
                            if (px > torso_max_x) torso_max_x = px;
                        }
                    }
                    // 肉体高光底色探测：防死红色的油桶 (极其耀眼的偏肉色灰白点)
                    else if (y_val > 110 && v >= 125 && v <= 150 && u > 100 && u < 135) {
                        bright_flesh_count++;
                    }
                }
            }
            
            // 🔪 锁二：切角矩形查杀 (假人的头两侧绝不能有太多红点)
            if (corner_violation > 8) continue;
            
            // 🔪 锁三：材质共生查杀 (近距离下，必须带有高反光的肉体底色特征)
            if (h > 40) {
                float flesh_ratio = (float)bright_flesh_count / (float)total_pixels;
                if (flesh_ratio < 0.015f) continue;
            }

            // 🔪 锁四：人类骨架比例查杀 (防直筒型电线杆)
            int head_width = (head_max_x != -1 && head_max_x > head_min_x) ? (head_max_x - head_min_x) : w;
            int torso_width = (torso_max_x != -1 && torso_max_x > torso_min_x) ? (torso_max_x - torso_min_x) : w;
            if (head_width >= (int)(torso_width * 0.85f)) continue; 

            // 🎯 完美幸存者，执行核心锁定：宽度取中，高度等比向下取 22% 锁死颈部/胸口
            int target_x = blobs[i].min_x + w / 2;
            int target_y = blobs[i].min_y + (int)(h * 0.22); 
            
            long dist = (target_x - cx)*(target_x - cx) + (target_y - cy)*(target_y - cy);
            if (dist < min_dist) {
                min_dist = dist;
                best_x = target_x;
                best_y = target_y;
            }
        }
        
        // 3. 将计算完成的光滑坐标极速发送到 PC
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

