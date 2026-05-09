import { Component, OnDestroy, signal } from '@angular/core';

@Component({
  selector: 'photo-upload',
  templateUrl: './photoUpload.html',
  styleUrl: './photoUpload.css'
})
export class PhotoUpload implements OnDestroy {
  private readonly apiBaseUrl = 'http://127.0.0.1:8000';
  private previewObjectUrl: string | null = null;
  private outputObjectUrl: string | null = null;

  protected selectedFile = signal<File | null>(null);
  protected selectedFileName = signal('');
  protected previewUrl = signal('');
  protected outputUrl = signal('');
  protected isLoading = signal(false);
  protected errorMessage = signal('');
  protected previewUnsupported = signal(false);
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

  protected async onFileSelected(event: Event): Promise<void> {
    const input = event.target as HTMLInputElement | null;
    const file = input?.files?.[0] ?? null;

    this.clearError();
    this.clearOutput();
    this.previewUnsupported.set(false);

    if (!file) {
      this.selectedFile.set(null);
      this.selectedFileName.set('');
      this.setPreview('');
      return;
    }

    if (!file.type.startsWith('image/')) {
      this.setError('Please choose a valid image file.');
      this.selectedFile.set(null);
      this.selectedFileName.set('');
      this.setPreview('');
      return;
    }

    this.selectedFile.set(file);
    this.selectedFileName.set(file.name);
    if (this.isPreviewSupported(file)) {
      this.setPreview(URL.createObjectURL(file));
    } else {
      try {
        const previewUrl = await this.createTiffPreview(file);
        this.setPreview(previewUrl);
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Failed to render TIFF preview.';
        this.setError(message);
        this.previewUnsupported.set(true);
        this.setPreview('');
      }
    }
  }

  protected async uploadAndSegment(): Promise<void> {
    const file = this.selectedFile();
    if (!file) {
      this.setError('Choose an image first.');
      return;
    }

    this.clearError();
    this.isLoading.set(true);

    try {
      const formData = new FormData();
      formData.append('image', file, file.name);

      const response = await fetch(
        `${this.apiBaseUrl}/segment/image`,
        {
          method: 'POST',
          body: formData
        }
      );

      if (!response.ok) {
        const message = await response.text();
        throw new Error(message || `Request failed (${response.status})`);
      }

      const blob = await response.blob();
      this.setOutput(URL.createObjectURL(blob));
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Upload failed.';
      this.setError(message);
    } finally {
      this.isLoading.set(false);
    }
  }

  protected clearSelection(): void {
    this.selectedFile.set(null);
    this.selectedFileName.set('');
    this.setPreview('');
    this.clearOutput();
    this.clearError();
    this.previewUnsupported.set(false);
  }

  ngOnDestroy(): void {
    this.revokePreview();
    this.revokeOutput();
  }

  private setPreview(url: string): void {
    this.revokePreview();
    this.previewObjectUrl = url || null;
    this.previewUrl.set(url);
  }

  private setOutput(url: string): void {
    this.revokeOutput();
    this.outputObjectUrl = url || null;
    this.outputUrl.set(url);
  }

  private clearOutput(): void {
    this.setOutput('');
  }

  private revokePreview(): void {
    if (this.previewObjectUrl) {
      URL.revokeObjectURL(this.previewObjectUrl);
      this.previewObjectUrl = null;
    }
  }

  private revokeOutput(): void {
    if (this.outputObjectUrl) {
      URL.revokeObjectURL(this.outputObjectUrl);
      this.outputObjectUrl = null;
    }
  }

  private setError(message: string): void {
    this.errorMessage.set(message);
  }

  private clearError(): void {
    this.errorMessage.set('');
  }

  private isPreviewSupported(file: File): boolean {
    const lowerName = file.name.toLowerCase();
    if (file.type === 'image/tiff' || lowerName.endsWith('.tif') || lowerName.endsWith('.tiff')) {
      return false;
    }
    return true;
  }

  private async createTiffPreview(file: File): Promise<string> {
    const module = await import('utif');
    const UTIF = module as unknown as {
      decode: (buffer: ArrayBuffer) => Array<{ width: number; height: number }>;
      decodeImage: (buffer: ArrayBuffer, ifd: object) => void;
      toRGBA8: (ifd: object) => Uint8Array;
    };

    const buffer = await file.arrayBuffer();
    const ifds = UTIF.decode(buffer);
    if (!ifds.length) {
      throw new Error('Unable to parse TIFF file.');
    }

    const first = ifds[0];
    UTIF.decodeImage(buffer, first);
    const rgba = UTIF.toRGBA8(first);
    const width = first.width;
    const height = first.height;

    if (!width || !height) {
      throw new Error('Invalid TIFF dimensions.');
    }

    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      throw new Error('Canvas is not available for preview.');
    }

    const imageData = new ImageData(new Uint8ClampedArray(rgba), width, height);
    ctx.putImageData(imageData, 0, 0);
    return canvas.toDataURL('image/png');
  }
}
