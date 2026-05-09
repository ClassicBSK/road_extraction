import { Component, OnDestroy, signal } from '@angular/core';

@Component({
  selector: 'live-view',
  templateUrl: './liveView.html',
  styleUrl: './liveView.css'
})
export class LiveView {
  private readonly apiBaseUrl = 'http://127.0.0.1:8000';
  private fpsWindowStartedAt = 0;
  private fpsFrameCount = 0;
  private segmentedAbortController: AbortController | null = null;
  private segmentedObjectUrl: string | null = null;

  protected videoUrlInput = signal('');
  protected activeVideoUrl = signal('');
  protected activeOriginalImgUrl = signal('');
  protected useImageForOriginal = signal(false);
  protected segmentedStreamUrl = signal('');
  protected segmentedDisplayUrl = signal('');
  protected segmentedFps = signal(0);
  protected readonly materialLegend = [
    { name: 'Background', rgb: '0, 0, 0', hex: '#000000' },
    { name: 'Asphalt', rgb: '220, 20, 60', hex: '#DC143C' },
    { name: 'Meadows', rgb: '34, 139, 34', hex: '#228B22' },
    { name: 'Gravel', rgb: '255, 165, 0', hex: '#FFA500' },
    { name: 'Trees', rgb: '0, 128, 0', hex: '#008000' },
    { name: 'Metal', rgb: '169, 169, 169', hex: '#A9A9A9' },
    { name: 'Soil', rgb: '139, 69, 19', hex: '#8B4513' },
    { name: 'Bitumen', rgb: '75, 0, 130', hex: '#4B0082' },
    { name: 'Bricks', rgb: '178, 34, 34', hex: '#B22222' }
  ] as const;

  private isLikelyImageStream(url: string): boolean {
    const lower = url.toLowerCase();
    return (
      lower.includes('/video-feed') ||
      lower.includes('multipart') ||
      lower.includes('/stream') ||
      lower.includes('mjpeg')
    ) && !lower.endsWith('.mp4');
  }

  protected applyVideoUrl(): void {
    const trimmedUrl = this.videoUrlInput().trim();
    if (!trimmedUrl) {
      this.activeVideoUrl.set('');
      this.activeOriginalImgUrl.set('');
      this.useImageForOriginal.set(false);
      this.segmentedStreamUrl.set('');
      this.stopSegmentedStream();
      this.resetSegmentedFps();
      return;
    }

    this.activeVideoUrl.set(trimmedUrl);
    this.activeOriginalImgUrl.set(`${trimmedUrl}${trimmedUrl.includes('?') ? '&' : '?'}t=${Date.now()}`);
    this.useImageForOriginal.set(this.isLikelyImageStream(trimmedUrl));

    const encodedVideoUrl = encodeURIComponent(trimmedUrl);
    const cacheBuster = Date.now();
    const streamUrl = `${this.apiBaseUrl}/segment/stream-url?video_url=${encodedVideoUrl}&frame_skip=1&confidence_threshold=0&t=${cacheBuster}`;
    this.segmentedStreamUrl.set(streamUrl);
    this.startSegmentedStream(streamUrl);
    this.resetSegmentedFps();
  }

  protected clearVideoUrl(): void {
    this.videoUrlInput.set('');
    this.activeVideoUrl.set('');
    this.activeOriginalImgUrl.set('');
    this.useImageForOriginal.set(false);
    this.segmentedStreamUrl.set('');
    this.stopSegmentedStream();
    this.resetSegmentedFps();
  }

  protected onInputChange(event: Event): void {
    const input = event.target as HTMLInputElement | null;
    this.videoUrlInput.set(input?.value ?? '');
  }

  protected onOriginalVideoError(): void {
    if (this.activeOriginalImgUrl()) {
      this.useImageForOriginal.set(true);
    }
  }

  ngOnDestroy(): void {
    this.stopSegmentedStream();
  }

  private startSegmentedStream(streamUrl: string): void {
    this.stopSegmentedStream();
    if (!streamUrl) {
      return;
    }

    const controller = new AbortController();
    this.segmentedAbortController = controller;

    void this.consumeSegmentedStream(streamUrl, controller.signal);
  }

  private stopSegmentedStream(): void {
    this.segmentedAbortController?.abort();
    this.segmentedAbortController = null;

    if (this.segmentedObjectUrl) {
      URL.revokeObjectURL(this.segmentedObjectUrl);
      this.segmentedObjectUrl = null;
    }

    this.segmentedDisplayUrl.set('');
  }

  private async consumeSegmentedStream(streamUrl: string, signal: AbortSignal): Promise<void> {
    try {
      const response = await fetch(streamUrl, {
        method: 'GET',
        cache: 'no-store',
        mode: 'cors',
        signal
      });

      if (!response.ok || !response.body) {
        throw new Error(`Segmented stream request failed (${response.status})`);
      }

      const reader = response.body.getReader();
      let buffer: Uint8Array<ArrayBufferLike> = new Uint8Array(0);

      while (!signal.aborted) {
        const { done, value } = await reader.read();
        if (done) {
          break;
        }
        if (!value || value.length === 0) {
          continue;
        }

        buffer = this.concatBytes(buffer, value);
        const { frames, remainder } = this.extractJpegFrames(buffer);
        buffer = remainder;

        for (const frame of frames) {
          this.pushSegmentedFrame(frame);
        }
      }
    } catch (error) {
      if (!signal.aborted) {
        console.error('Segmented stream reader stopped:', error);
      }
    }
  }

  private pushSegmentedFrame(frameBytes: Uint8Array<ArrayBufferLike>): void {
    const normalizedFrame = new Uint8Array(frameBytes);
    const nextUrl = URL.createObjectURL(new Blob([normalizedFrame], { type: 'image/jpeg' }));
    const previousUrl = this.segmentedObjectUrl;

    this.segmentedObjectUrl = nextUrl;
    this.segmentedDisplayUrl.set(nextUrl);

    if (previousUrl) {
      URL.revokeObjectURL(previousUrl);
    }

    this.onSegmentedFrameDecoded();
  }

  private onSegmentedFrameDecoded(): void {
    const now = performance.now();
    if (this.fpsWindowStartedAt === 0) {
      this.fpsWindowStartedAt = now;
      this.fpsFrameCount = 1;
      return;
    }

    this.fpsFrameCount += 1;
    const elapsedMs = now - this.fpsWindowStartedAt;
    if (elapsedMs >= 1000) {
      const fps = (this.fpsFrameCount * 1000) / elapsedMs;
      this.segmentedFps.set(Number.isFinite(fps) ? Number(fps.toFixed(1)) : 0);
      this.fpsWindowStartedAt = now;
      this.fpsFrameCount = 0;
    }
  }

  private extractJpegFrames(data: Uint8Array<ArrayBufferLike>): { frames: Uint8Array<ArrayBufferLike>[]; remainder: Uint8Array<ArrayBufferLike> } {
    const frames: Uint8Array<ArrayBufferLike>[] = [];
    let cursor = 0;

    while (cursor < data.length - 1) {
      const start = this.indexOfMarker(data, 0xff, 0xd8, cursor);
      if (start === -1) {
        return { frames, remainder: new Uint8Array(0) };
      }

      const end = this.indexOfMarker(data, 0xff, 0xd9, start + 2);
      if (end === -1) {
        return { frames, remainder: data.slice(start) };
      }

      frames.push(data.slice(start, end + 2));
      cursor = end + 2;
    }

    return { frames, remainder: data.slice(cursor) };
  }

  private indexOfMarker(data: Uint8Array<ArrayBufferLike>, first: number, second: number, startAt: number): number {
    for (let i = startAt; i < data.length - 1; i += 1) {
      if (data[i] === first && data[i + 1] === second) {
        return i;
      }
    }
    return -1;
  }

  private concatBytes(a: Uint8Array<ArrayBufferLike>, b: Uint8Array<ArrayBufferLike>): Uint8Array<ArrayBufferLike> {
    const merged = new Uint8Array(a.length + b.length);
    merged.set(a, 0);
    merged.set(b, a.length);
    return merged;
  }

  private resetSegmentedFps(): void {
    this.segmentedFps.set(0);
    this.fpsWindowStartedAt = 0;
    this.fpsFrameCount = 0;
  }
}