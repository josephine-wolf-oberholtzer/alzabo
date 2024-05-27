import logging
from pathlib import Path

import pytest
from supriya import SynthDef
from uqbar.strings import normalize

from alzabo.constants import SCSYNTH_ANALYSIS_SIZE
from alzabo.core.scsynth import (
    aggregate,
    analyze,
    build_offline_analysis_synthdef,
    build_online_analysis_synthdef,
)


@pytest.mark.asyncio
async def test_aggregate(recordings_path: Path) -> None:
    """
    Given a numpy array (or slice thereof), aggregate the data into one value
    per row.
    """
    audio_path = recordings_path / "ibn-arabi-44100-1s.wav"
    analysis = await analyze(audio_path)
    assert aggregate(analysis[10:20]) == {
        "is_voiced": False,
        "r:centroid:mean": 102.17546615600585,
        "r:centroid:std": 4.6465817744641615,
        "r:chroma": [
            0.058108895528130235,
            0.05983194620348513,
            0.05957096866331994,
            0.06540839513763785,
            0.09433190505951643,
            0.13969308882951736,
            0.16502613127231597,
            0.11585419476032258,
            0.06454430948942899,
            0.05822751689702273,
            0.06162195838987827,
            0.05778068909421563,
        ],
        "r:f0:mean": -1.0,
        "r:f0:std": 0.0,
        "r:flatness:mean": 0.12626039693132043,
        "r:flatness:std": 0.11192989810564866,
        "r:mfcc": [
            0.5500117391347885,
            0.32335798889398576,
            0.3053797483444214,
            0.009281852841377258,
            0.2670589178800583,
            0.29174797534942626,
            0.23289807587862016,
            0.2481079339981079,
            0.3160476952791214,
            0.3116189897060394,
            0.2348788186907768,
            0.28578830808401107,
            0.14814276844263077,
            0.2448696032166481,
            0.15299261808395387,
            0.2195752054452896,
            0.2123452514410019,
            0.2703291043639183,
            0.3108777284622192,
            0.2301008000969887,
            0.30138839930295946,
            0.24768556356430055,
            0.2619121611118317,
            0.2634056001901627,
            0.21913592666387557,
            0.197379432618618,
            0.22120984345674516,
            0.242153337597847,
            0.2665415570139885,
            0.2662082463502884,
            0.24349459260702133,
            0.2671607226133347,
            0.2485161080956459,
            0.22621844410896302,
            0.23960513174533843,
            0.2600105956196785,
            0.2579686462879181,
            0.2525347426533699,
            0.24745482206344604,
            0.2594869628548622,
            0.25294138193130494,
            0.25,
        ],
        "r:onsets": 0.1,
        "r:peak:mean": -39.71778841018677,
        "r:peak:std": 24.74014081451421,
        "r:rms:mean": -41.7574541091919,
        "r:rms:std": 23.85892211791406,
        "r:rolloff:mean": 88.51670455932617,
        "r:rolloff:std": 7.237117549925431,
        "w:centroid:mean": 102.17546615600585,
        "w:centroid:std": 4.6465817744641615,
        "w:chroma": [
            0.058108895528130235,
            0.05983194620348513,
            0.05957096866331994,
            0.06540839513763785,
            0.09433190505951643,
            0.13969308882951736,
            0.16502613127231597,
            0.11585419476032258,
            0.06454430948942899,
            0.05822751689702273,
            0.06162195838987827,
            0.05778068909421563,
        ],
        "w:f0:mean": -1.0,
        "w:f0:std": 0.0,
        "w:flatness:mean": 0.12626039693132043,
        "w:flatness:std": 0.11192989810564866,
        "w:mfcc": [
            0.5500117391347885,
            0.32335798889398576,
            0.3053797483444214,
            0.009281852841377258,
            0.2670589178800583,
            0.29174797534942626,
            0.23289807587862016,
            0.2481079339981079,
            0.3160476952791214,
            0.3116189897060394,
            0.2348788186907768,
            0.28578830808401107,
            0.14814276844263077,
            0.2448696032166481,
            0.15299261808395387,
            0.2195752054452896,
            0.2123452514410019,
            0.2703291043639183,
            0.3108777284622192,
            0.2301008000969887,
            0.30138839930295946,
            0.24768556356430055,
            0.2619121611118317,
            0.2634056001901627,
            0.21913592666387557,
            0.197379432618618,
            0.22120984345674516,
            0.242153337597847,
            0.2665415570139885,
            0.2662082463502884,
            0.24349459260702133,
            0.2671607226133347,
            0.2485161080956459,
            0.22621844410896302,
            0.23960513174533843,
            0.2600105956196785,
            0.2579686462879181,
            0.2525347426533699,
            0.24745482206344604,
            0.2594869628548622,
            0.25294138193130494,
            0.25,
        ],
        "w:onsets": 0.5623413251903491,
        "w:peak:mean": -39.71778841018677,
        "w:peak:std": 24.74014081451421,
        "w:rms:mean": -41.7574541091919,
        "w:rms:std": 23.85892211791406,
        "w:rolloff:mean": 88.51670455932617,
        "w:rolloff:std": 7.237117549925431,
    }
    assert aggregate(analysis[:40]) == {
        "is_voiced": True,
        "r:centroid:mean": 89.98462104797363,
        "r:centroid:std": 26.896250040055225,
        "r:chroma": [
            0.02857129691819864,
            0.03427787481705309,
            0.030756118156568847,
            0.04668607334606349,
            0.11743937754072249,
            0.21052453480660915,
            0.21170435100793839,
            0.09836374083533883,
            0.03570144157783943,
            0.029965405073653528,
            0.030173810700534886,
            0.025835976968755857,
        ],
        "r:f0:mean": 64.39348137896994,
        "r:f0:std": 4.007876386002851,
        "r:flatness:mean": 0.1502397577278316,
        "r:flatness:std": 0.2213624785613459,
        "r:mfcc": [
            0.4267571233212948,
            0.18711134530603885,
            0.20196336768567563,
            -0.05220579504966736,
            0.09526116587221622,
            0.21329422742128373,
            0.2671147618442774,
            0.21624561995267869,
            0.32516602240502834,
            0.37777928188443183,
            0.2981520812958479,
            0.36062665469944477,
            0.22268481515347957,
            0.24111236035823821,
            0.07285813577473163,
            0.18924205452203752,
            0.23654527701437472,
            0.2931366734206676,
            0.38021478466689584,
            0.2620264749974012,
            0.25570895709097385,
            0.22550652250647546,
            0.2439420934766531,
            0.3021530512720346,
            0.2534282587468624,
            0.19028708413243295,
            0.18993747159838675,
            0.24341083467006683,
            0.2825418543070555,
            0.2768420748412609,
            0.23589200675487518,
            0.2520163211971521,
            0.260795671865344,
            0.2399986658245325,
            0.23174107521772386,
            0.24228834994137288,
            0.2598283741623163,
            0.25414316579699514,
            0.218731377273798,
            0.2510281529277563,
            0.2637407198548317,
            0.25,
        ],
        "r:onsets": 0.075,
        "r:peak:mean": -102.68271009922027,
        "r:peak:std": 222.02114050153008,
        "r:rms:mean": -102.61906691789628,
        "r:rms:std": 222.10698239442502,
        "r:rolloff:mean": 76.79694499969483,
        "r:rolloff:std": 20.69229787878831,
        "w:centroid:mean": 89.98462104797363,
        "w:centroid:std": 26.896250040055225,
        "w:chroma": [
            0.02857129691819864,
            0.03427787481705309,
            0.030756118156568847,
            0.04668607334606349,
            0.11743937754072249,
            0.21052453480660915,
            0.21170435100793839,
            0.09836374083533883,
            0.03570144157783943,
            0.029965405073653528,
            0.030173810700534886,
            0.025835976968755857,
        ],
        "w:f0:mean": 64.39348137896994,
        "w:f0:std": 4.007876386002851,
        "w:flatness:mean": 0.1502397577278316,
        "w:flatness:std": 0.2213624785613459,
        "w:mfcc": [
            0.4267571233212948,
            0.18711134530603885,
            0.20196336768567563,
            -0.05220579504966736,
            0.09526116587221622,
            0.21329422742128373,
            0.2671147618442774,
            0.21624561995267869,
            0.32516602240502834,
            0.37777928188443183,
            0.2981520812958479,
            0.36062665469944477,
            0.22268481515347957,
            0.24111236035823821,
            0.07285813577473163,
            0.18924205452203752,
            0.23654527701437472,
            0.2931366734206676,
            0.38021478466689584,
            0.2620264749974012,
            0.25570895709097385,
            0.22550652250647546,
            0.2439420934766531,
            0.3021530512720346,
            0.2534282587468624,
            0.19028708413243295,
            0.18993747159838675,
            0.24341083467006683,
            0.2825418543070555,
            0.2768420748412609,
            0.23589200675487518,
            0.2520163211971521,
            0.260795671865344,
            0.2399986658245325,
            0.23174107521772386,
            0.24228834994137288,
            0.2598283741623163,
            0.25414316579699514,
            0.218731377273798,
            0.2510281529277563,
            0.2637407198548317,
            0.25,
        ],
        "w:onsets": 0.5233175696960528,
        "w:peak:mean": -102.68271009922027,
        "w:peak:std": 222.02114050153008,
        "w:rms:mean": -102.61906691789628,
        "w:rms:std": 222.10698239442502,
        "w:rolloff:mean": 76.79694499969483,
        "w:rolloff:std": 20.69229787878831,
    }


@pytest.mark.asyncio
async def test_analyze(caplog, recordings_path: Path) -> None:
    caplog.set_level(logging.DEBUG, "supriya")
    audio_path = recordings_path / "ibn-arabi-44100-1s.wav"
    analysis = await analyze(audio_path)
    assert analysis.shape == (86, SCSYNTH_ANALYSIS_SIZE)
    for frame in analysis:
        print(frame)
        assert frame.max() > 0
        assert frame.min() < 0


def test_build_offline_analysis_synthdef() -> None:
    synthdef = build_offline_analysis_synthdef()
    assert isinstance(synthdef, SynthDef)
    assert str(synthdef) == normalize(
        """
        synthdef:
            name: analysis
            ugens:
            -   Control.kr:
                    buffer_id: 0.0
                    duration: 0.0
                    in_: 0.0
            -   In.ar:
                    bus: Control.kr[2:in_]
            -   Amplitude.ar:
                    source: In.ar[0]
                    attack_time: 0.01
                    release_time: 0.01
            -   UnaryOpUGen(AMPLITUDE_TO_DB).ar/0:
                    source: Amplitude.ar[0]
            -   BinaryOpUGen(MULTIPLICATION).ar:
                    left: In.ar[0]
                    right: In.ar[0]
            -   LPF.ar:
                    source: BinaryOpUGen(MULTIPLICATION).ar[0]
                    frequency: 10.0
            -   UnaryOpUGen(SQUARE_ROOT).ar:
                    source: LPF.ar[0]
            -   UnaryOpUGen(AMPLITUDE_TO_DB).ar/1:
                    source: UnaryOpUGen(SQUARE_ROOT).ar[0]
            -   Pitch.kr:
                    source: In.ar[0]
                    initial_frequency: 440.0
                    min_frequency: 60.0
                    max_frequency: 3000.0
                    exec_frequency: 100.0
                    max_bins_per_octave: 16.0
                    median: 1.0
                    amplitude_threshold: 0.01
                    peak_threshold: 0.5
                    down_sample_factor: 1.0
                    clarity: 0.0
            -   BufFrames.kr:
                    buffer_id: Control.kr[0:buffer_id]
            -   BinaryOpUGen(SUBTRACTION).kr:
                    left: BufFrames.kr[0]
                    right: 1.0
            -   Line.kr:
                    start: 0.0
                    stop: BinaryOpUGen(SUBTRACTION).kr[0]
                    duration: Control.kr[1:duration]
                    done_action: 0.0
            -   UnaryOpUGen(FLOOR).kr:
                    source: Line.kr[0]
            -   MaxLocalBufs.ir:
                    maximum: 1.0
            -   LocalBuf.ir:
                    channel_count: 1.0
                    frame_count: 2048.0
            -   FFT.kr:
                    buffer_id: LocalBuf.ir[0]
                    source: In.ar[0]
                    hop: 0.25
                    window_type: 0.0
                    active: 1.0
                    window_size: 2048.0
            -   Onsets.kr:
                    pv_chain: FFT.kr[0]
                    threshold: 0.01
                    odftype: 5.0
                    relaxtime: 0.1
                    floor_: 1e-06
                    mingap: 10.0
                    medianspan: 11.0
                    whtype: 1.0
                    rawodf: 0.0
            -   SpecCentroid.kr:
                    pv_chain: FFT.kr[0]
            -   UnaryOpUGen(HZ_TO_MIDI).kr/0:
                    source: SpecCentroid.kr[0]
            -   Sanitize.kr/0:
                    source: UnaryOpUGen(HZ_TO_MIDI).kr/0[0]
                    replace: 0.0
            -   SpecFlatness.kr:
                    pv_chain: FFT.kr[0]
            -   SpecPcile.kr:
                    pv_chain: FFT.kr[0]
                    fraction: 0.5
                    interpolate: 0.0
            -   UnaryOpUGen(HZ_TO_MIDI).kr/1:
                    source: SpecPcile.kr[0]
            -   Sanitize.kr/1:
                    source: UnaryOpUGen(HZ_TO_MIDI).kr/1[0]
                    replace: 0.0
            -   MFCC.kr:
                    pv_chain: FFT.kr[0]
                    coeff_count: 42.0
            -   UnaryOpUGen(HZ_TO_MIDI).kr/2:
                    source: Pitch.kr[0]
            -   SampleRate.ir: null
            -   BinaryOpUGen(FLOAT_DIVISION).ir:
                    left: SampleRate.ir[0]
                    right: 2.0
            -   FluidChroma.kr:
                    source: In.ar[0]
                    chroma_count: 12.0
                    max_chroma_count: 12.0
                    reference: 1.0
                    normalize: 1.0
                    min_frequency: 0.0
                    max_frequency: BinaryOpUGen(FLOAT_DIVISION).ir[0]
                    window_size: 2048.0
                    hop_size: 512.0
                    fft_size: 2048.0
                    max_fft_size: 2048.0
            -   BufWr.kr:
                    buffer_id: Control.kr[0:buffer_id]
                    phase: UnaryOpUGen(FLOOR).kr[0]
                    loop: 1.0
                    source[0]: UnaryOpUGen(AMPLITUDE_TO_DB).ar/0[0]
                    source[1]: UnaryOpUGen(AMPLITUDE_TO_DB).ar/1[0]
                    source[2]: UnaryOpUGen(HZ_TO_MIDI).kr/2[0]
                    source[3]: Pitch.kr[1]
                    source[4]: Onsets.kr[0]
                    source[5]: Sanitize.kr/0[0]
                    source[6]: SpecFlatness.kr[0]
                    source[7]: Sanitize.kr/1[0]
                    source[8]: MFCC.kr[0]
                    source[9]: MFCC.kr[1]
                    source[10]: MFCC.kr[2]
                    source[11]: MFCC.kr[3]
                    source[12]: MFCC.kr[4]
                    source[13]: MFCC.kr[5]
                    source[14]: MFCC.kr[6]
                    source[15]: MFCC.kr[7]
                    source[16]: MFCC.kr[8]
                    source[17]: MFCC.kr[9]
                    source[18]: MFCC.kr[10]
                    source[19]: MFCC.kr[11]
                    source[20]: MFCC.kr[12]
                    source[21]: MFCC.kr[13]
                    source[22]: MFCC.kr[14]
                    source[23]: MFCC.kr[15]
                    source[24]: MFCC.kr[16]
                    source[25]: MFCC.kr[17]
                    source[26]: MFCC.kr[18]
                    source[27]: MFCC.kr[19]
                    source[28]: MFCC.kr[20]
                    source[29]: MFCC.kr[21]
                    source[30]: MFCC.kr[22]
                    source[31]: MFCC.kr[23]
                    source[32]: MFCC.kr[24]
                    source[33]: MFCC.kr[25]
                    source[34]: MFCC.kr[26]
                    source[35]: MFCC.kr[27]
                    source[36]: MFCC.kr[28]
                    source[37]: MFCC.kr[29]
                    source[38]: MFCC.kr[30]
                    source[39]: MFCC.kr[31]
                    source[40]: MFCC.kr[32]
                    source[41]: MFCC.kr[33]
                    source[42]: MFCC.kr[34]
                    source[43]: MFCC.kr[35]
                    source[44]: MFCC.kr[36]
                    source[45]: MFCC.kr[37]
                    source[46]: MFCC.kr[38]
                    source[47]: MFCC.kr[39]
                    source[48]: MFCC.kr[40]
                    source[49]: MFCC.kr[41]
                    source[50]: FluidChroma.kr[0]
                    source[51]: FluidChroma.kr[1]
                    source[52]: FluidChroma.kr[2]
                    source[53]: FluidChroma.kr[3]
                    source[54]: FluidChroma.kr[4]
                    source[55]: FluidChroma.kr[5]
                    source[56]: FluidChroma.kr[6]
                    source[57]: FluidChroma.kr[7]
                    source[58]: FluidChroma.kr[8]
                    source[59]: FluidChroma.kr[9]
                    source[60]: FluidChroma.kr[10]
                    source[61]: FluidChroma.kr[11]
        """
    )


def test_build_online_analysis_synthdef_scsynth() -> None:
    synthdef = build_online_analysis_synthdef("scsynth")
    assert isinstance(synthdef, SynthDef)
    assert str(synthdef) == normalize(
        """
        synthdef:
            name: analysis
            ugens:
            -   Control.kr:
                    in_: 0.0
            -   In.ar:
                    bus: Control.kr[0:in_]
            -   Amplitude.ar:
                    source: In.ar[0]
                    attack_time: 0.01
                    release_time: 0.01
            -   UnaryOpUGen(AMPLITUDE_TO_DB).ar/0:
                    source: Amplitude.ar[0]
            -   BinaryOpUGen(MULTIPLICATION).ar:
                    left: In.ar[0]
                    right: In.ar[0]
            -   LPF.ar:
                    source: BinaryOpUGen(MULTIPLICATION).ar[0]
                    frequency: 10.0
            -   UnaryOpUGen(SQUARE_ROOT).ar:
                    source: LPF.ar[0]
            -   UnaryOpUGen(AMPLITUDE_TO_DB).ar/1:
                    source: UnaryOpUGen(SQUARE_ROOT).ar[0]
            -   Pitch.kr:
                    source: In.ar[0]
                    initial_frequency: 440.0
                    min_frequency: 60.0
                    max_frequency: 3000.0
                    exec_frequency: 100.0
                    max_bins_per_octave: 16.0
                    median: 1.0
                    amplitude_threshold: 0.01
                    peak_threshold: 0.5
                    down_sample_factor: 1.0
                    clarity: 0.0
            -   SampleRate.ir/0: null
            -   BinaryOpUGen(FLOAT_DIVISION).ir/0:
                    left: SampleRate.ir/0[0]
                    right: 512.0
            -   Impulse.kr:
                    frequency: BinaryOpUGen(FLOAT_DIVISION).ir/0[0]
                    phase: 0.0
            -   MaxLocalBufs.ir:
                    maximum: 1.0
            -   LocalBuf.ir:
                    channel_count: 1.0
                    frame_count: 2048.0
            -   FFT.kr:
                    buffer_id: LocalBuf.ir[0]
                    source: In.ar[0]
                    hop: 0.25
                    window_type: 0.0
                    active: 1.0
                    window_size: 2048.0
            -   Onsets.kr:
                    pv_chain: FFT.kr[0]
                    threshold: 0.01
                    odftype: 5.0
                    relaxtime: 0.1
                    floor_: 1e-06
                    mingap: 10.0
                    medianspan: 11.0
                    whtype: 1.0
                    rawodf: 0.0
            -   SpecCentroid.kr:
                    pv_chain: FFT.kr[0]
            -   UnaryOpUGen(HZ_TO_MIDI).kr/0:
                    source: SpecCentroid.kr[0]
            -   Sanitize.kr/0:
                    source: UnaryOpUGen(HZ_TO_MIDI).kr/0[0]
                    replace: 0.0
            -   SpecFlatness.kr:
                    pv_chain: FFT.kr[0]
            -   SpecPcile.kr:
                    pv_chain: FFT.kr[0]
                    fraction: 0.5
                    interpolate: 0.0
            -   UnaryOpUGen(HZ_TO_MIDI).kr/1:
                    source: SpecPcile.kr[0]
            -   Sanitize.kr/1:
                    source: UnaryOpUGen(HZ_TO_MIDI).kr/1[0]
                    replace: 0.0
            -   MFCC.kr:
                    pv_chain: FFT.kr[0]
                    coeff_count: 42.0
            -   UnaryOpUGen(HZ_TO_MIDI).kr/2:
                    source: Pitch.kr[0]
            -   SampleRate.ir/1: null
            -   BinaryOpUGen(FLOAT_DIVISION).ir/1:
                    left: SampleRate.ir/1[0]
                    right: 2.0
            -   FluidChroma.kr:
                    source: In.ar[0]
                    chroma_count: 12.0
                    max_chroma_count: 12.0
                    reference: 1.0
                    normalize: 1.0
                    min_frequency: 0.0
                    max_frequency: BinaryOpUGen(FLOAT_DIVISION).ir/1[0]
                    window_size: 2048.0
                    hop_size: 512.0
                    fft_size: 2048.0
                    max_fft_size: 2048.0
            -   SendReply.kr:
                    trigger: Impulse.kr[0]
                    reply_id: -1.0
                    character_count: 9.0
                    character[0]: 47.0
                    character[1]: 97.0
                    character[2]: 110.0
                    character[3]: 97.0
                    character[4]: 108.0
                    character[5]: 121.0
                    character[6]: 115.0
                    character[7]: 105.0
                    character[8]: 115.0
                    source[0]: UnaryOpUGen(AMPLITUDE_TO_DB).ar/0[0]
                    source[1]: UnaryOpUGen(AMPLITUDE_TO_DB).ar/1[0]
                    source[2]: UnaryOpUGen(HZ_TO_MIDI).kr/2[0]
                    source[3]: Pitch.kr[1]
                    source[4]: Onsets.kr[0]
                    source[5]: Sanitize.kr/0[0]
                    source[6]: SpecFlatness.kr[0]
                    source[7]: Sanitize.kr/1[0]
                    source[8]: MFCC.kr[0]
                    source[9]: MFCC.kr[1]
                    source[10]: MFCC.kr[2]
                    source[11]: MFCC.kr[3]
                    source[12]: MFCC.kr[4]
                    source[13]: MFCC.kr[5]
                    source[14]: MFCC.kr[6]
                    source[15]: MFCC.kr[7]
                    source[16]: MFCC.kr[8]
                    source[17]: MFCC.kr[9]
                    source[18]: MFCC.kr[10]
                    source[19]: MFCC.kr[11]
                    source[20]: MFCC.kr[12]
                    source[21]: MFCC.kr[13]
                    source[22]: MFCC.kr[14]
                    source[23]: MFCC.kr[15]
                    source[24]: MFCC.kr[16]
                    source[25]: MFCC.kr[17]
                    source[26]: MFCC.kr[18]
                    source[27]: MFCC.kr[19]
                    source[28]: MFCC.kr[20]
                    source[29]: MFCC.kr[21]
                    source[30]: MFCC.kr[22]
                    source[31]: MFCC.kr[23]
                    source[32]: MFCC.kr[24]
                    source[33]: MFCC.kr[25]
                    source[34]: MFCC.kr[26]
                    source[35]: MFCC.kr[27]
                    source[36]: MFCC.kr[28]
                    source[37]: MFCC.kr[29]
                    source[38]: MFCC.kr[30]
                    source[39]: MFCC.kr[31]
                    source[40]: MFCC.kr[32]
                    source[41]: MFCC.kr[33]
                    source[42]: MFCC.kr[34]
                    source[43]: MFCC.kr[35]
                    source[44]: MFCC.kr[36]
                    source[45]: MFCC.kr[37]
                    source[46]: MFCC.kr[38]
                    source[47]: MFCC.kr[39]
                    source[48]: MFCC.kr[40]
                    source[49]: MFCC.kr[41]
                    source[50]: FluidChroma.kr[0]
                    source[51]: FluidChroma.kr[1]
                    source[52]: FluidChroma.kr[2]
                    source[53]: FluidChroma.kr[3]
                    source[54]: FluidChroma.kr[4]
                    source[55]: FluidChroma.kr[5]
                    source[56]: FluidChroma.kr[6]
                    source[57]: FluidChroma.kr[7]
                    source[58]: FluidChroma.kr[8]
                    source[59]: FluidChroma.kr[9]
                    source[60]: FluidChroma.kr[10]
                    source[61]: FluidChroma.kr[11]
        """
    )


def test_build_online_analysis_synthdef_supernova() -> None:
    synthdef = build_online_analysis_synthdef("supernova")
    assert isinstance(synthdef, SynthDef)
    assert str(synthdef) == normalize(
        """
        synthdef:
            name: analysis
            ugens:
            -   Control.kr:
                    in_: 0.0
            -   In.ar:
                    bus: Control.kr[0:in_]
            -   Amplitude.ar:
                    source: In.ar[0]
                    attack_time: 0.01
                    release_time: 0.01
            -   UnaryOpUGen(AMPLITUDE_TO_DB).ar/0:
                    source: Amplitude.ar[0]
            -   BinaryOpUGen(MULTIPLICATION).ar:
                    left: In.ar[0]
                    right: In.ar[0]
            -   LPF.ar:
                    source: BinaryOpUGen(MULTIPLICATION).ar[0]
                    frequency: 10.0
            -   UnaryOpUGen(SQUARE_ROOT).ar:
                    source: LPF.ar[0]
            -   UnaryOpUGen(AMPLITUDE_TO_DB).ar/1:
                    source: UnaryOpUGen(SQUARE_ROOT).ar[0]
            -   Pitch.kr:
                    source: In.ar[0]
                    initial_frequency: 440.0
                    min_frequency: 60.0
                    max_frequency: 3000.0
                    exec_frequency: 100.0
                    max_bins_per_octave: 16.0
                    median: 1.0
                    amplitude_threshold: 0.01
                    peak_threshold: 0.5
                    down_sample_factor: 1.0
                    clarity: 0.0
            -   SampleRate.ir: null
            -   BinaryOpUGen(FLOAT_DIVISION).ir:
                    left: SampleRate.ir[0]
                    right: 512.0
            -   Impulse.kr:
                    frequency: BinaryOpUGen(FLOAT_DIVISION).ir[0]
                    phase: 0.0
            -   MaxLocalBufs.ir:
                    maximum: 1.0
            -   LocalBuf.ir:
                    channel_count: 1.0
                    frame_count: 2048.0
            -   FFT.kr:
                    buffer_id: LocalBuf.ir[0]
                    source: In.ar[0]
                    hop: 0.25
                    window_type: 0.0
                    active: 1.0
                    window_size: 2048.0
            -   Onsets.kr:
                    pv_chain: FFT.kr[0]
                    threshold: 0.01
                    odftype: 5.0
                    relaxtime: 0.1
                    floor_: 1e-06
                    mingap: 10.0
                    medianspan: 11.0
                    whtype: 1.0
                    rawodf: 0.0
            -   SpecCentroid.kr:
                    pv_chain: FFT.kr[0]
            -   UnaryOpUGen(HZ_TO_MIDI).kr/0:
                    source: SpecCentroid.kr[0]
            -   Sanitize.kr/0:
                    source: UnaryOpUGen(HZ_TO_MIDI).kr/0[0]
                    replace: 0.0
            -   SpecFlatness.kr:
                    pv_chain: FFT.kr[0]
            -   SpecPcile.kr:
                    pv_chain: FFT.kr[0]
                    fraction: 0.5
                    interpolate: 0.0
            -   UnaryOpUGen(HZ_TO_MIDI).kr/1:
                    source: SpecPcile.kr[0]
            -   Sanitize.kr/1:
                    source: UnaryOpUGen(HZ_TO_MIDI).kr/1[0]
                    replace: 0.0
            -   MFCC.kr:
                    pv_chain: FFT.kr[0]
                    coeff_count: 42.0
            -   UnaryOpUGen(HZ_TO_MIDI).kr/2:
                    source: Pitch.kr[0]
            -   SendReply.kr:
                    trigger: Impulse.kr[0]
                    reply_id: -1.0
                    character_count: 9.0
                    character[0]: 47.0
                    character[1]: 97.0
                    character[2]: 110.0
                    character[3]: 97.0
                    character[4]: 108.0
                    character[5]: 121.0
                    character[6]: 115.0
                    character[7]: 105.0
                    character[8]: 115.0
                    source[0]: UnaryOpUGen(AMPLITUDE_TO_DB).ar/0[0]
                    source[1]: UnaryOpUGen(AMPLITUDE_TO_DB).ar/1[0]
                    source[2]: UnaryOpUGen(HZ_TO_MIDI).kr/2[0]
                    source[3]: Pitch.kr[1]
                    source[4]: Onsets.kr[0]
                    source[5]: Sanitize.kr/0[0]
                    source[6]: SpecFlatness.kr[0]
                    source[7]: Sanitize.kr/1[0]
                    source[8]: MFCC.kr[0]
                    source[9]: MFCC.kr[1]
                    source[10]: MFCC.kr[2]
                    source[11]: MFCC.kr[3]
                    source[12]: MFCC.kr[4]
                    source[13]: MFCC.kr[5]
                    source[14]: MFCC.kr[6]
                    source[15]: MFCC.kr[7]
                    source[16]: MFCC.kr[8]
                    source[17]: MFCC.kr[9]
                    source[18]: MFCC.kr[10]
                    source[19]: MFCC.kr[11]
                    source[20]: MFCC.kr[12]
                    source[21]: MFCC.kr[13]
                    source[22]: MFCC.kr[14]
                    source[23]: MFCC.kr[15]
                    source[24]: MFCC.kr[16]
                    source[25]: MFCC.kr[17]
                    source[26]: MFCC.kr[18]
                    source[27]: MFCC.kr[19]
                    source[28]: MFCC.kr[20]
                    source[29]: MFCC.kr[21]
                    source[30]: MFCC.kr[22]
                    source[31]: MFCC.kr[23]
                    source[32]: MFCC.kr[24]
                    source[33]: MFCC.kr[25]
                    source[34]: MFCC.kr[26]
                    source[35]: MFCC.kr[27]
                    source[36]: MFCC.kr[28]
                    source[37]: MFCC.kr[29]
                    source[38]: MFCC.kr[30]
                    source[39]: MFCC.kr[31]
                    source[40]: MFCC.kr[32]
                    source[41]: MFCC.kr[33]
                    source[42]: MFCC.kr[34]
                    source[43]: MFCC.kr[35]
                    source[44]: MFCC.kr[36]
                    source[45]: MFCC.kr[37]
                    source[46]: MFCC.kr[38]
                    source[47]: MFCC.kr[39]
                    source[48]: MFCC.kr[40]
                    source[49]: MFCC.kr[41]
        """
    )
