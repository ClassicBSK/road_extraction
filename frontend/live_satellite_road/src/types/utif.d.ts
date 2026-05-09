declare module 'utif' {
  export function decode(buffer: ArrayBuffer): Array<{ width: number; height: number }>;
  export function decodeImage(buffer: ArrayBuffer, ifd: object): void;
  export function toRGBA8(ifd: object): Uint8Array;
}
