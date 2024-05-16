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
    with SynthDefBuilder(amplitude=1.0, buffer_id=0, out=0, panning=0) as builder:
        in_ = PlayBuf.ar(builder["buffer_id"], channel_count=1, loop=0)
        envelope = Line.kr(
            duration=BufDur.kr(buffer_id=builder["buffer_id"]),
            done_action=DoneAction.FREE_SYNTH,
        ).hanning_window()
        source = in_ * envelope * builder["amplitude"]
        panned = PanAz.ar(
            channel_count=channel_count, source=source, position=builder["panning"]
        )
        Out.ar(bus=builder["out"], source=panned)
    return builder.build(name="basic-playback")


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
