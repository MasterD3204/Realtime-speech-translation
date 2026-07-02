// AudioWorkletProcessor: capture raw PCM liên tục, gom đủ 512 sample (32ms @ 16kHz)
// rồi post ra main thread. Chạy trên audio rendering thread — không block bởi
// main thread, khác với ScriptProcessorNode (deprecated) trước đây.

class PCMCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.frameSize = 512;
    this.buffer = new Float32Array(0);
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0) return true;
    const channel = input[0];
    if (!channel || channel.length === 0) return true;

    const merged = new Float32Array(this.buffer.length + channel.length);
    merged.set(this.buffer);
    merged.set(channel, this.buffer.length);
    this.buffer = merged;

    while (this.buffer.length >= this.frameSize) {
      const frame = this.buffer.slice(0, this.frameSize);
      this.buffer = this.buffer.slice(this.frameSize);
      // Transfer underlying buffer — tránh copy thêm lần nữa qua postMessage.
      this.port.postMessage(frame, [frame.buffer]);
    }

    return true;
  }
}

registerProcessor("pcm-capture-processor", PCMCaptureProcessor);
