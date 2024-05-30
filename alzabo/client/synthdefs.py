from supriya import DoneAction, SynthDef, SynthDefBuilder, synthdef
from supriya.ugens import (
    LPF,
    AllpassL,
    BufDur,
    CombL,
    CompanderD,
    DelayN,
    ExpRand,
    In,
    LFDNoise3,
    LFNoise2,
    LeakDC,
    Limiter,
    Line,
    Linen,
    Mix,
    Out,
    PanAz,
    PlayBuf,
    ReplaceOut,
    UGenOperable,
    Warp1,
)


def build_aux_send(channel_count=2) -> SynthDef:
    with SynthDefBuilder(aux_out=0, gate=1, mix=0.5, out=0) as builder:
        in_ = In.ar(channel_count=channel_count, bus=builder["out"])
        envelope = Linen.kr(
            attack_time=0.25,
            done_action=DoneAction.FREE_SYNTH,
            gate=builder["gate"],
            release_time=5.0,
            sustain_level=1.0,
        )
        source = in_ * envelope
        Out.ar(bus=builder["out"], source=source * (1.0 - builder["mix"]))
        Out.ar(bus=builder["aux_out"], source=source * builder["mix"])
    return builder.build(name="aux-send")


def build_basic_playback(channel_count=2) -> SynthDef:
    with SynthDefBuilder(buffer_id=0, gain=0.0, out=0, panning=0) as builder:
        in_ = PlayBuf.ar(buffer_id=builder["buffer_id"], channel_count=1, loop=0)
        envelope = Line.kr(
            duration=BufDur.kr(buffer_id=builder["buffer_id"]),
            done_action=DoneAction.FREE_SYNTH,
        ).hanning_window()
        source = in_ * envelope * builder["gain"].db_to_amplitude()
        panned = PanAz.ar(
            channel_count=channel_count, source=source, position=builder["panning"]
        )
        Out.ar(bus=builder["out"], source=panned)
    return builder.build(name="basic-playback")


def build_warp_playback(channel_count=2) -> SynthDef:
    with SynthDefBuilder(
        buffer_id=0,
        gain=0.0,
        out=0,
        overlaps=4,
        panning=0.0,
        start=0.0,
        stop=1.0,
        time_scaling=1.0,
        transposition=0.0,
    ) as builder:
        duration = BufDur.kr(buffer_id=builder["buffer_id"]) * builder["time_scaling"]
        window = Line.kr(
            duration=duration, done_action=DoneAction.FREE_SYNTH
        ).hanning_window()
        window *= builder["gain"].db_to_amplitude() / builder["overlaps"]
        pointer = Line.kr(
            start=builder["start"], stop=builder["stop"], duration=duration
        )
        signals = []
        layers = 2
        for _ in range(layers):
            window_size = LFDNoise3.kr(
                frequency=ExpRand.ir(minimum=0.01, maximum=0.1)
            ).scale(-1, 1, 0.05, 0.5)
            position = builder["panning"] * LFNoise2.kr(frequency=0.5).scale(
                -1, 1, 0.5, 1
            )
            frequency_scaling = (
                builder["transposition"] + (LFNoise2.kr(frequency=0.1) * 0.25)
            ).semitones_to_ratio()
            signal = Warp1.ar(
                buffer_id=builder["buffer_id"],
                frequency_scaling=frequency_scaling,
                interpolation=4,
                overlaps=builder["overlaps"],
                pointer=(
                    (pointer + LFNoise2.kr(frequency=1.0) * 0.05).clip(0.0, 1.0)
                    * ((duration - window_size) / duration)
                ),
                window_rand_ratio=0.15,
                window_size=window_size,
            )
            signal *= window
            signal = PanAz.ar(
                channel_count=channel_count, source=signal, position=position
            )
            signals.append(signal)
        signal = LeakDC.ar(source=Mix.multichannel(signals, channel_count)) / layers
        Out.ar(bus=builder["out"], source=signal)
    return builder.build(name="warp-playback")


@synthdef()
def hdverb(in_=0, out=0, decay=3.5, lpf1=2000, lpf2=6000, predelay=0.025) -> None:
    comb_count = 16 // 2
    allpass_count = 8
    source = In.ar(bus=in_, channel_count=1)
    source = DelayN.ar(
        source=source, maximum_delay_time=0.5, delay_time=predelay.clip(0.0001, 0.5)
    )
    source = (
        Mix.new(
            LPF.ar(
                source=CombL.ar(
                    source=source,
                    maximum_delay_time=0.1,
                    delay_time=LFNoise2.kr(
                        frequency=ExpRand.ir(minimum=0.02, maximum=0.04)
                    ).scale(-1, 1, 0.02, 0.099, exponential=True),
                    decay_time=decay,
                ),
                frequency=lpf1,
            )
            for _ in range(comb_count)
        )
        * 0.25
    )
    for _ in range(allpass_count):
        source = AllpassL.ar(
            source=source,
            maximum_delay_time=0.1,
            delay_time=LFNoise2.kr(
                frequency=ExpRand.ir(minimum=0.02, maximum=0.04)
            ).scale(-1, 1, 0.02, 0.099, exponential=True),
            decay_time=decay,
        )
    source = LeakDC.ar(source=source)
    source = LPF.ar(source=source, frequency=lpf2) * 0.5
    Out.ar(bus=out, source=source)


@synthdef()
def limiter(in_: float, out: float) -> None:
    source = In.ar(bus=in_, channel_count=1)
    bands: list[UGenOperable] = []
    rest = source
    for frequency in [150, 1500, 6000]:
        band = LPF.ar(source=rest, frequency=frequency)
        bands.append(band)
        rest = rest - band
    bands.append(rest)
    source = Mix.new(
        sources=[
            CompanderD.ar(source=band, slope_above=0.25, threshold=0.25)
            for band in bands
        ]
    )
    source = Limiter.ar(source=source, duration=0.1)
    ReplaceOut.ar(bus=out, source=source)
