import { Component, signal } from '@angular/core';

@Component({
  selector: 'live-view',
  templateUrl: './liveView.html',
  styleUrl: './liveView.css'
})
export class LiveView {
  private readonly apiBaseUrl = 'http://127.0.0.1:8000';

  protected videoUrlInput = signal('');
  protected activeVideoUrl = signal('');
  protected activeOriginalImgUrl = signal('');
  protected useImageForOriginal = signal(false);
  protected segmentedStreamUrl = signal('');

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
      return;
    }

    this.activeVideoUrl.set(trimmedUrl);
    this.activeOriginalImgUrl.set(`${trimmedUrl}${trimmedUrl.includes('?') ? '&' : '?'}t=${Date.now()}`);
    this.useImageForOriginal.set(this.isLikelyImageStream(trimmedUrl));

    const encodedVideoUrl = encodeURIComponent(trimmedUrl);
    const cacheBuster = Date.now();
    this.segmentedStreamUrl.set(
      `${this.apiBaseUrl}/segment/stream-url?video_url=${encodedVideoUrl}&frame_skip=1&confidence_threshold=0&t=${cacheBuster}`
    );
  }

  protected clearVideoUrl(): void {
    this.videoUrlInput.set('');
    this.activeVideoUrl.set('');
    this.activeOriginalImgUrl.set('');
    this.useImageForOriginal.set(false);
    this.segmentedStreamUrl.set('');
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
}