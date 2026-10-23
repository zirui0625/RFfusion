"""Optional FP16 VAE inference with a full-precision high-resolution decoder tail."""

def configure_safe_fp16(autoencoder):
    """Call after loading original FP32 weights, outside an autocast context."""
    autoencoder.encoder.half()
    autoencoder.quant_conv.half()
    autoencoder.quantize.half()
    autoencoder.post_quant_conv.half()
    decoder = autoencoder.decoder
    decoder.conv_in.half()
    decoder.mid.half()
    for level in list(decoder.up)[1:]:
        level.half()
    decoder.up[0].float()
    decoder.norm_out.float()
    decoder.conv_out.float()

    def enter_tail(_module, inputs):
        return (inputs[0].float(), *inputs[1:])

    decoder.up[0].block[0].register_forward_pre_hook(enter_tail)
    return autoencoder
