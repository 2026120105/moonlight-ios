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

// ==========================================
// 🚀 [外挂轨道] M2 旁路影子硬件解码与视觉雷达
// ==========================================
#import <VideoToolbox/VideoToolbox.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/time.h>
#include <math.h>
#include <fcntl.h>
#include <stdlib.h>

static int m2_udp_sock = -1;
static struct sockaddr_in m2_pc_addr;
static float m2_prev_cx = -1, m2_prev_cy = -1;
static double m2_prev_time = 0;

static double m2_get_time_sec(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + (tv.tv_usec / 1000000.0);
}

// 🎯 并行解码回调：内存一解压出画面，立马执行扫描
static void m2_process_frame(CVImageBufferRef pixelBuffer) {
    if (!pixelBuffer) return;
    
    if (m2_udp_sock == -1) {
        m2_udp_sock = socket(AF_INET, SOCK_DGRAM, 0);
        fcntl(m2_udp_sock, F_SETFL, fcntl(m2_udp_sock, F_GETFL, 0) | O_NONBLOCK);
        m2_pc_addr.sin_family = AF_INET;
        m2_pc_addr.sin_port = htons(9999);
        // ⚠️⚠️⚠️ 极其重要：替换为你 PC 的热点 IP ⚠️⚠️⚠️
        inet_pton(AF_INET, "192.168.137.1", &m2_pc_addr.sin_addr); 
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
        int best_x = -1, best_y = -1;
        long min_dist = 2000000000;

        int start_y = (int)(height * 0.2);
        int end_y = (int)(height * 0.8);
        int start_x = (int)(width * 0.2);
        int end_x = (int)(width * 0.8);

        for (int y = start_y; y < end_y; y += 2) {
            uint8_t *yRow = yPlane + y * yBytesPerRow;
            uint8_t *uvRow = uvPlane + (y / 2) * uvBytesPerRow;
            
            for (int x = start_x; x < end_x; x += 2) {
                // 过滤屏幕中心的准星
                if (abs(x - cx) < 25 && abs(y - cy) < 25) continue;

                if (yRow[x] > 240) { 
                    uint8_t u = uvRow[(x / 2) * 2];
                    uint8_t v = uvRow[(x / 2) * 2 + 1];
                    
                    if (u > 115 && u < 140 && v > 115 && v < 140) {
                        int is_line = 1;
                        for (int k = 1; k < 4; k++) {
                            if (x + k * 2 < end_x && yRow[x + k * 2] < 200) {
                                is_line = 0; break;
                            }
                        }
                        if (is_line) {
                            long dist = (x - cx)*(x - cx) + (y - cy)*(y - cy);
                            if (dist < min_dist) {
                                min_dist = dist;
                                best_x = x;
                                best_y = y;
                            }
                            x += 30; // 找到后跳出干扰像素
                        }
                    }
                }
            }
        }
        
        if (best_x != -1) {
            float dx_left = best_x - cx;
            float dy_left = best_y - cy;
            double current_time = m2_get_time_sec();
            float speed = 0, angle = 0;
            
            if (m2_prev_cx != -1) {
                double dt = current_time - m2_prev_time;
                if (dt > 0) {
                    float vx = (best_x - m2_prev_cx) / dt; 
                    float vy = (best_y - m2_prev_cy) / dt;
                    speed = sqrt(vx*vx + vy*vy); 
                    angle = atan2(vy, vx) * (180.0 / M_PI);
                }
            }
            m2_prev_cx = best_x; m2_prev_cy = best_y; m2_prev_time = current_time;
            
            char msg[128];
            snprintf(msg, sizeof(msg), "{\"f\":1,\"dx\":%.1f,\"dy\":%.1f,\"spd\":%.1f,\"ang\":%.1f}", dx_left, dy_left, speed, angle);
            sendto(m2_udp_sock, msg, strlen(msg), 0, (struct sockaddr *)&m2_pc_addr, sizeof(m2_pc_addr));
        } else {
            m2_prev_cx = -1;
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
// ==========================================

// Private libavformat API for writing the AV1 Codec Configuration Box
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
    
    // 🔥 新增：M2 雷达影子硬件解码会话
    VTDecompressionSessionRef _m2Session;
}

- (void)reinitializeDisplayLayer
{
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
    }
    else {
        [_view.layer addSublayer:displayLayer];
    }
    
    if (formatDesc != nil) {
        CFRelease(formatDesc);
        formatDesc = nil;
    }
    
    // 🔥 清理旧的影子解码器
    if (_m2Session != NULL) {
        VTDecompressionSessionInvalidate(_m2Session);
        CFRelease(_m2Session);
        _m2Session = NULL;
    }
}

- (id)initWithView:(StreamView*)view callbacks:(id<ConnectionCallbacks>)callbacks streamAspectRatio:(float)aspectRatio useFramePacing:(BOOL)useFramePacing
{
    self = [super init];
    
    _view = view;
    _callbacks = callbacks;
    _streamAspectRatio = aspectRatio;
    framePacing = useFramePacing;
    
    parameterSetBuffers = [[NSMutableArray alloc] init];
    
    [self reinitializeDisplayLayer];
    
    return self;
}

- (void)setupWithVideoFormat:(int)videoFormat width:(int)videoWidth height:(int)videoHeight frameRate:(int)frameRate
{
    self->videoFormat = videoFormat;
    self->frameRate = frameRate;
}

- (void)start
{
    _displayLink = [CADisplayLink displayLinkWithTarget:self selector:@selector(displayLinkCallback:)];
    if (@available(iOS 15.0, tvOS 15.0, *)) {
        _displayLink.preferredFrameRateRange = CAFrameRateRangeMake(self->frameRate, self->frameRate, self->frameRate);
    }
    else {
        _displayLink.preferredFramesPerSecond = self->frameRate;
    }
    [_displayLink addToRunLoop:[NSRunLoop mainRunLoop] forMode:NSDefaultRunLoopMode];
}

// TODO: Refactor this
int DrSubmitDecodeUnit(PDECODE_UNIT decodeUnit);

- (void)displayLinkCallback:(CADisplayLink *)sender
{
    VIDEO_FRAME_HANDLE handle;
    PDECODE_UNIT du;
    
    while (LiPollNextVideoFrame(&handle, &du)) {
        LiCompleteVideoFrame(handle, DrSubmitDecodeUnit(du));
        
        if (framePacing) {
            double displayRefreshRate = 1 / (_displayLink.targetTimestamp - _displayLink.timestamp);
            if (displayRefreshRate >= frameRate * 0.9f) {
                if (LiGetPendingVideoFrames() == 1) {
                    break;
                }
            }
        }
    }
}

- (void)stop
{
    [_displayLink invalidate];
    
    // 🔥 释放解码器
    if (_m2Session != NULL) {
        VTDecompressionSessionInvalidate(_m2Session);
        CFRelease(_m2Session);
        _m2Session = NULL;
    }
}

#define NALU_START_PREFIX_SIZE 3
#define NAL_LENGTH_PREFIX_SIZE 4

- (void)updateAnnexBBufferForRange:(CMBlockBufferRef)frameBuffer dataBlock:(CMBlockBufferRef)dataBuffer offset:(int)offset length:(int)nalLength
{
    OSStatus status;
    size_t oldOffset = CMBlockBufferGetDataLength(frameBuffer);
    
    status = CMBlockBufferAppendMemoryBlock(frameBuffer, NULL,
                                            NAL_LENGTH_PREFIX_SIZE,
                                            kCFAllocatorDefault, NULL, 0,
                                            NAL_LENGTH_PREFIX_SIZE, 0);
    if (status != noErr) {
        return;
    }
    
    const int dataLength = nalLength - NALU_START_PREFIX_SIZE;
    const uint8_t lengthBytes[] = {(uint8_t)(dataLength >> 24), (uint8_t)(dataLength >> 16),
        (uint8_t)(dataLength >> 8), (uint8_t)dataLength};
    status = CMBlockBufferReplaceDataBytes(lengthBytes, frameBuffer,
                                           oldOffset, NAL_LENGTH_PREFIX_SIZE);
    if (status != noErr) {
        return;
    }
    
    status = CMBlockBufferAppendBufferReference(frameBuffer, dataBuffer, offset + NALU_START_PREFIX_SIZE, dataLength, 0);
    if (status != noErr) {
        return;
    }
}

- (NSData*)getAv1CodecConfigurationBox:(NSData*)frameData  {
    AVIOContext* ioctx = NULL;
    int err;
    
    err = avio_open_dyn_buf(&ioctx);
    if (err < 0) {
        return nil;
    }

    err = ff_isom_write_av1c(ioctx, (uint8_t*)frameData.bytes, (int)frameData.length, 1);
    
    uint8_t* av1cBuf = NULL;
    int av1cBufLen = avio_close_dyn_buf(ioctx, &av1cBuf);
    
    NSData* data = nil;
    if (err >= 0 && av1cBufLen > 0) {
        data = [NSData dataWithBytes:av1cBuf length:av1cBufLen];
    }
    else {
        data = nil;
    }
    
    av_free(av1cBuf);
    return data;
}

- (CMVideoFormatDescriptionRef)createAV1FormatDescriptionForIDRFrame:(NSData*)frameData {
    NSMutableDictionary* extensions = [[NSMutableDictionary alloc] init];

    CodedBitstreamContext* cbsCtx = NULL;
    int err = ff_cbs_init(&cbsCtx, AV_CODEC_ID_AV1, NULL);
    if (err < 0) {
        return nil;
    }
    
    AVPacket avPacket = {};
    avPacket.data = (uint8_t*)frameData.bytes;
    avPacket.size = (int)frameData.length;
    
    CodedBitstreamFragment cbsFrag = {};
    err = ff_cbs_read_packet(cbsCtx, &cbsFrag, &avPacket);
    if (err < 0) {
        ff_cbs_close(&cbsCtx);
        return nil;
    }
    
#define SET_CFSTR_EXTENSION(key, value) extensions[(__bridge NSString*)key] = (__bridge NSString*)(value)
#define SET_EXTENSION(key, value) extensions[(__bridge NSString*)key] = (value)

    SET_EXTENSION(kCMFormatDescriptionExtension_FormatName, @"av01");
    SET_EXTENSION(kCMFormatDescriptionExtension_Depth, @24);
    
    CodedBitstreamAV1Context* bitstreamCtx = (CodedBitstreamAV1Context*)cbsCtx->priv_data;
    AV1RawSequenceHeader* seqHeader = bitstreamCtx->sequence_header;
    if (seqHeader == NULL) {
        ff_cbs_fragment_free(&cbsFrag);
        ff_cbs_close(&cbsCtx);
        return nil;
    }
    
    switch (seqHeader->color_config.color_primaries) {
        case 1: SET_CFSTR_EXTENSION(kCMFormatDescriptionExtension_ColorPrimaries, kCMFormatDescriptionColorPrimaries_ITU_R_709_2); break;
        case 6: SET_CFSTR_EXTENSION(kCMFormatDescriptionExtension_ColorPrimaries, kCMFormatDescriptionColorPrimaries_SMPTE_C); break;
        case 9: SET_CFSTR_EXTENSION(kCMFormatDescriptionExtension_ColorPrimaries, kCMFormatDescriptionColorPrimaries_ITU_R_2020); break;
    }
    
    switch (seqHeader->color_config.transfer_characteristics) {
        case 1: case 6: SET_CFSTR_EXTENSION(kCMFormatDescriptionExtension_TransferFunction, kCMFormatDescriptionTransferFunction_ITU_R_709_2); break;
        case 7: SET_CFSTR_EXTENSION(kCMFormatDescriptionExtension_TransferFunction, kCMFormatDescriptionTransferFunction_SMPTE_240M_1995); break;
        case 8: SET_CFSTR_EXTENSION(kCMFormatDescriptionExtension_TransferFunction, kCMFormatDescriptionTransferFunction_Linear); break;
        case 14: case 15: SET_CFSTR_EXTENSION(kCMFormatDescriptionExtension_TransferFunction, kCMFormatDescriptionTransferFunction_ITU_R_2020); break;
        case 16: SET_CFSTR_EXTENSION(kCMFormatDescriptionExtension_TransferFunction, kCMFormatDescriptionTransferFunction_SMPTE_ST_2084_PQ); break;
        case 17: SET_CFSTR_EXTENSION(kCMFormatDescriptionExtension_TransferFunction, kCMFormatDescriptionTransferFunction_ITU_R_2100_HLG); break;
    }
    
    switch (seqHeader->color_config.matrix_coefficients) {
        case 1: SET_CFSTR_EXTENSION(kCMFormatDescriptionExtension_YCbCrMatrix, kCMFormatDescriptionYCbCrMatrix_ITU_R_709_2); break;
        case 6: SET_CFSTR_EXTENSION(kCMFormatDescriptionExtension_YCbCrMatrix, kCMFormatDescriptionYCbCrMatrix_ITU_R_601_4); break;
        case 7: SET_CFSTR_EXTENSION(kCMFormatDescriptionExtension_YCbCrMatrix, kCMFormatDescriptionYCbCrMatrix_SMPTE_240M_1995); break;
        case 9: SET_CFSTR_EXTENSION(kCMFormatDescriptionExtension_YCbCrMatrix, kCMFormatDescriptionYCbCrMatrix_ITU_R_2020); break;
    }
    
    SET_EXTENSION(kCMFormatDescriptionExtension_FullRangeVideo, @(seqHeader->color_config.color_range == 1));
    SET_EXTENSION(kCMFormatDescriptionExtension_FieldCount, @(1));
    
    switch (seqHeader->color_config.chroma_sample_position) {
        case 1: SET_CFSTR_EXTENSION(kCMFormatDescriptionExtension_ChromaLocationTopField, kCMFormatDescriptionChromaLocation_Left); break;
        case 2: SET_CFSTR_EXTENSION(kCMFormatDescriptionExtension_ChromaLocationTopField, kCMFormatDescriptionChromaLocation_TopLeft); break;
    }
    
    if (contentLightLevelInfo) {
        SET_EXTENSION(kCMFormatDescriptionExtension_ContentLightLevelInfo, contentLightLevelInfo);
    }
    
    if (masteringDisplayColorVolume) {
        SET_EXTENSION(kCMFormatDescriptionExtension_MasteringDisplayColorVolume, masteringDisplayColorVolume);
    }
    
    extensions[(__bridge NSString*)kCMFormatDescriptionExtension_SampleDescriptionExtensionAtoms] =
    @{
        @"av1C" : [self getAv1CodecConfigurationBox:frameData],
    };
    extensions[@"BitsPerComponent"] = @(bitstreamCtx->bit_depth);
    
#undef SET_EXTENSION
#undef SET_CFSTR_EXTENSION
    
    CMVideoFormatDescriptionRef formatDesc = NULL;
    OSStatus status = CMVideoFormatDescriptionCreate(kCFAllocatorDefault, kCMVideoCodecType_AV1,
                                                     bitstreamCtx->frame_width, bitstreamCtx->frame_height,
                                                     (__bridge CFDictionaryRef)extensions,
                                                     &formatDesc);
    if (status != noErr) {
        formatDesc = NULL;
    }
    
    ff_cbs_fragment_free(&cbsFrag);
    ff_cbs_close(&cbsCtx);
    return formatDesc;
}

- (int)submitDecodeBuffer:(unsigned char *)data length:(int)length bufferType:(int)bufferType decodeUnit:(PDECODE_UNIT)du
{
    OSStatus status;
    
    if (du->frameType == FRAME_TYPE_IDR) {
        if (bufferType != BUFFER_TYPE_PICDATA) {
            if (bufferType == BUFFER_TYPE_VPS || bufferType == BUFFER_TYPE_SPS || bufferType == BUFFER_TYPE_PPS) {
                int startLen = data[2] == 0x01 ? 3 : 4;
                [parameterSetBuffers addObject:[NSData dataWithBytes:&data[startLen] length:length - startLen]];
            }
            return DR_OK;
        }
        
        // Free the old format description
        if (formatDesc != NULL) {
            CFRelease(formatDesc);
            formatDesc = NULL;
        }
        
        // 🔥 释放旧的旁路解码器
        if (_m2Session != NULL) {
            VTDecompressionSessionInvalidate(_m2Session);
            CFRelease(_m2Session);
            _m2Session = NULL;
        }
        
        if (videoFormat & VIDEO_FORMAT_MASK_H264) {
            size_t parameterSetCount = [parameterSetBuffers count];
            const uint8_t* parameterSetPointers[parameterSetCount];
            size_t parameterSetSizes[parameterSetCount];
            for (int i = 0; i < parameterSetCount; i++) {
                NSData* parameterSet = parameterSetBuffers[i];
                parameterSetPointers[i] = parameterSet.bytes;
                parameterSetSizes[i] = parameterSet.length;
            }
            
            status = CMVideoFormatDescriptionCreateFromH264ParameterSets(kCFAllocatorDefault,
                                                                         parameterSetCount,
                                                                         parameterSetPointers,
                                                                         parameterSetSizes,
                                                                         NAL_LENGTH_PREFIX_SIZE,
                                                                         &formatDesc);
            if (status != noErr) {
                formatDesc = NULL;
            }
            
            [parameterSetBuffers removeAllObjects];
        }
        else if (videoFormat & VIDEO_FORMAT_MASK_H265) {
            size_t parameterSetCount = [parameterSetBuffers count];
            const uint8_t* parameterSetPointers[parameterSetCount];
            size_t parameterSetSizes[parameterSetCount];
            for (int i = 0; i < parameterSetCount; i++) {
                NSData* parameterSet = parameterSetBuffers[i];
                parameterSetPointers[i] = parameterSet.bytes;
                parameterSetSizes[i] = parameterSet.length;
            }
            
            NSMutableDictionary* videoFormatParams = [[NSMutableDictionary alloc] init];
            
            if (contentLightLevelInfo) {
                [videoFormatParams setObject:contentLightLevelInfo forKey:(__bridge NSString*)kCMFormatDescriptionExtension_ContentLightLevelInfo];
            }
            
            if (masteringDisplayColorVolume) {
                [videoFormatParams setObject:masteringDisplayColorVolume forKey:(__bridge NSString*)kCMFormatDescriptionExtension_MasteringDisplayColorVolume];
            }
            
            status = CMVideoFormatDescriptionCreateFromHEVCParameterSets(kCFAllocatorDefault,
                                                                         parameterSetCount,
                                                                         parameterSetPointers,
                                                                         parameterSetSizes,
                                                                         NAL_LENGTH_PREFIX_SIZE,
                                                                         (__bridge CFDictionaryRef)videoFormatParams,
                                                                         &formatDesc);
            
            if (status != noErr) {
                formatDesc = NULL;
            }
            
            [parameterSetBuffers removeAllObjects];
        }
        else if (videoFormat & VIDEO_FORMAT_MASK_AV1) {
            NSData* fullFrameData = [NSData dataWithBytesNoCopy:data length:length freeWhenDone:NO];
            formatDesc = [self createAV1FormatDescriptionForIDRFrame:fullFrameData];
        }
    }
    
    if (formatDesc == NULL) {
        free(data);
        return DR_NEED_IDR;
    }
    
    // 🔥 挂载 M2 旁路硬件解码器
    if (_m2Session == NULL) {
        VTDecompressionOutputCallbackRecord cb = {0};
        cb.decompressionOutputCallback = m2_decompression_callback;
        cb.decompressionOutputRefCon = NULL;
        
        NSDictionary *attrs = @{
            (id)kCVPixelBufferPixelFormatTypeKey: @(kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange)
        };
        
        VTDecompressionSessionCreate(kCFAllocatorDefault,
                                     formatDesc,
                                     NULL,
                                     (__bridge CFDictionaryRef)attrs,
                                     &cb,
                                     &_m2Session);
    }
    
    if (displayLayer.status == AVQueuedSampleBufferRenderingStatusFailed) {
        [self reinitializeDisplayLayer];
        free(data);
        return DR_NEED_IDR;
    }
    
    CMBlockBufferRef frameBlockBuffer;
    CMBlockBufferRef dataBlockBuffer;
    
    status = CMBlockBufferCreateWithMemoryBlock(NULL, data, length, kCFAllocatorDefault, NULL, 0, length, 0, &dataBlockBuffer);
    if (status != noErr) {
        free(data);
        return DR_NEED_IDR;
    }
    
    status = CMBlockBufferCreateEmpty(NULL, 0, 0, &frameBlockBuffer);
    if (status != noErr) {
        CFRelease(dataBlockBuffer);
        return DR_NEED_IDR;
    }
    
    if (videoFormat & (VIDEO_FORMAT_MASK_H264 | VIDEO_FORMAT_MASK_H265)) {
        int lastOffset = -1;
        for (int i = 0; i < length - NALU_START_PREFIX_SIZE; i++) {
            if (data[i] == 0 && data[i+1] == 0 && data[i+2] == 1) {
                if (lastOffset != -1) {
                    [self updateAnnexBBufferForRange:frameBlockBuffer dataBlock:dataBlockBuffer offset:lastOffset length:i - lastOffset];
                }
                lastOffset = i;
            }
        }
        
        if (lastOffset != -1) {
            [self updateAnnexBBufferForRange:frameBlockBuffer dataBlock:dataBlockBuffer offset:lastOffset length:length - lastOffset];
        }
    }
    else {
        status = CMBlockBufferAppendBufferReference(frameBlockBuffer, dataBlockBuffer, 0, length, 0);
        if (status != noErr) {
            return DR_NEED_IDR;
        }
    }
        
    CMSampleBufferRef sampleBuffer;
    
    CMSampleTimingInfo sampleTiming = {kCMTimeInvalid, CMTimeMake(du->presentationTimeMs, 1000), kCMTimeInvalid};
    
    status = CMSampleBufferCreateReady(kCFAllocatorDefault,
                                  frameBlockBuffer,
                                  formatDesc, 1, 1,
                                  &sampleTiming, 0, NULL,
                                  &sampleBuffer);
    if (status != noErr) {
        CFRelease(dataBlockBuffer);
        CFRelease(frameBlockBuffer);
        return DR_NEED_IDR;
    }

    // Enqueue the next frame (原代码：无损0延迟送显)
    [self->displayLayer enqueueSampleBuffer:sampleBuffer];
    
    // 🔥 暗杀轨道：把同样的压缩包喂给我们的影子解码器
    if (_m2Session) {
        VTDecodeFrameFlags flags = kVTDecodeFrame_EnableAsynchronousDecompression;
        VTDecompressionSessionDecodeFrame(_m2Session, sampleBuffer, flags, NULL, NULL);
    }
    
    if (du->frameType == FRAME_TYPE_IDR) {
        self->displayLayer.hidden = NO;
        [self->_callbacks videoContentShown];
    }
    
    CFRelease(dataBlockBuffer);
    CFRelease(frameBlockBuffer);
    CFRelease(sampleBuffer);
    
    return DR_OK;
}

- (void)setHdrMode:(BOOL)enabled {
    SS_HDR_METADATA hdrMetadata;
    
    BOOL hasMetadata = enabled && LiGetHdrMetadata(&hdrMetadata);
    BOOL metadataChanged = NO;
    
    if (hasMetadata && hdrMetadata.displayPrimaries[0].x != 0 && hdrMetadata.maxDisplayLuminance != 0) {
        struct {
          vector_ushort2 primaries[3];
          vector_ushort2 white_point;
          uint32_t luminance_max;
          uint32_t luminance_min;
        } __attribute__((packed, aligned(4))) mdcv;

        mdcv.primaries[0].x = __builtin_bswap16(hdrMetadata.displayPrimaries[1].x);
        mdcv.primaries[0].y = __builtin_bswap16(hdrMetadata.displayPrimaries[1].y);
        mdcv.primaries[1].x = __builtin_bswap16(hdrMetadata.displayPrimaries[2].x);
        mdcv.primaries[1].y = __builtin_bswap16(hdrMetadata.displayPrimaries[2].y);
        mdcv.primaries[2].x = __builtin_bswap16(hdrMetadata.displayPrimaries[0].x);
        mdcv.primaries[2].y = __builtin_bswap16(hdrMetadata.displayPrimaries[0].y);

        mdcv.white_point.x = __builtin_bswap16(hdrMetadata.whitePoint.x);
        mdcv.white_point.y = __builtin_bswap16(hdrMetadata.whitePoint.y);

        mdcv.luminance_max = __builtin_bswap32((uint32_t)hdrMetadata.maxDisplayLuminance * 10000);
        mdcv.luminance_min = __builtin_bswap32(hdrMetadata.minDisplayLuminance);

        NSData* newMdcv = [NSData dataWithBytes:&mdcv length:sizeof(mdcv)];
        if (masteringDisplayColorVolume == nil || ![newMdcv isEqualToData:masteringDisplayColorVolume]) {
            masteringDisplayColorVolume = newMdcv;
            metadataChanged = YES;
        }
    }
    else if (masteringDisplayColorVolume != nil) {
        masteringDisplayColorVolume = nil;
        metadataChanged = YES;
    }
    
    if (hasMetadata && hdrMetadata.maxContentLightLevel != 0 && hdrMetadata.maxFrameAverageLightLevel != 0) {
        struct {
            uint16_t max_content_light_level;
            uint16_t max_frame_average_light_level;
        } __attribute__((packed, aligned(2))) cll;

        cll.max_content_light_level = __builtin_bswap16(hdrMetadata.maxContentLightLevel);
        cll.max_frame_average_light_level = __builtin_bswap16(hdrMetadata.maxFrameAverageLightLevel);

        NSData* newCll = [NSData dataWithBytes:&cll length:sizeof(cll)];
        if (contentLightLevelInfo == nil || ![newCll isEqualToData:contentLightLevelInfo]) {
            contentLightLevelInfo = newCll;
            metadataChanged = YES;
        }
    }
    else if (contentLightLevelInfo != nil) {
        contentLightLevelInfo = nil;
        metadataChanged = YES;
    }
    
    if (metadataChanged) {
        LiRequestIdrFrame();
    }
}

@end

