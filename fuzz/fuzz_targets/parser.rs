#![no_main]

use libfuzzer_sys::fuzz_target;
use vqa::*;

fuzz_target!(|data: &[u8]| {
    // Every public parser must fail cleanly on arbitrary input.
    let _ = vqa_version(data);
    let _ = frame_info(data);
    let _ = snd2_chunk(data);
    let _ = vqfr_chunk(data);
    let _ = cbf_chunk(data);
    let _ = vqa_header(data);
    let _ = finf_chunk(data);
    let _ = raw_chunk(data);
    let _ = cbp_chunk(data);
    let _ = cpl_chunk(data);
    let _ = vpt_chunk(data);
    let _ = vptr_chunk(data);
    let _ = vqfl_chunk(data);
    let _ = sn2j_chunk(data);

    // The high-level API must also hold up: parse, decode a bounded number
    // of video frames and convert them to RGB, and decode the soundtrack.
    if let Ok(vqa) = VQA::parse(data) {
        if let Ok(frames) = vqa.frames() {
            for frame in frames.take(16) {
                let Ok(frame) = frame else {
                    break;
                };
                let _ = frame.to_rgb888();
            }
        }
        let _ = vqa.decode_audio();
    }

    // Walk the container the way a real consumer does: FORM header, VQA
    // header, then FINF (scanning past any LINF/CINF chunks before it), and
    // finally the frame data each decoded FINF offset points at.
    let Ok((rest, _)) = form_chunk(data) else {
        return;
    };
    let Ok((rest, _)) = vqa_header(rest) else {
        return;
    };

    let Some(finf_pos) = rest.windows(4).position(|w| w == b"FINF") else {
        return;
    };
    let Ok((_, finf)) = finf_chunk(&rest[finf_pos..]) else {
        return;
    };

    for frame in finf.frames {
        let Some(frame_data) = data.get(frame.offset as usize..) else {
            continue;
        };
        let _ = snd2_chunk(frame_data);
        let _ = vqfr_chunk(frame_data);
        let _ = cbf_chunk(frame_data);
    }
});
